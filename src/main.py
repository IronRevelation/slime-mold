# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path

import slangpy as spy

from simulation_parameters import SimulationParameters


SHADER_DIR = Path(__file__).parent / "shaders"
MAX_DELTA_TIME = 0.05


class App:
    def __init__(self):
        self.window = spy.Window(
            width=1200,
            height=720,
            title="Slime mold simulation",
            resizable=True,
        )
        self.device = spy.Device(
            enable_debug_layers=True,
            compiler_options={"include_paths": [SHADER_DIR]},
        )
        self.surface = self.device.create_surface(self.window)
        self.surface.configure(
            width=self.window.width,
            height=self.window.height,
            vsync=False,
        )
        self.ui = spy.ui.Context(self.device)

        self.diffuse_kernel = self._load_kernel("diffuse_decay")
        self.update_agents_kernel = self._load_kernel("update_agents")
        self.render_kernel = self._load_kernel("render_image")

        self.parameters = SimulationParameters()
        self.agent_data = None
        self.trail_a = None
        self.trail_b = None
        self.output_texture = None
        self.resource_signature = None

        self.playing = True
        self.reset_requested = True
        self.simulation_frame = 0
        self.fps_average = 0.0

        self.window.on_keyboard_event = self.on_keyboard_event
        self.window.on_mouse_event = self.on_mouse_event
        self.window.on_resize = self.on_resize

        self.setup_ui()

    def _load_kernel(self, entry_point: str) -> spy.ComputeKernel:
        program = self.device.load_program("slime_mold", [entry_point])
        return self.device.create_compute_kernel(program)

    def setup_ui(self) -> None:
        window = spy.ui.Window(
            self.ui.screen,
            "Settings",
            size=spy.float2(320, 440),
        )
        self.fps_text = spy.ui.Text(window, "FPS: 0")
        spy.ui.Button(window, "Start", callback=self.start)
        spy.ui.Button(window, "Stop", callback=self.stop)
        spy.ui.Button(window, "Reset", callback=self.request_reset)
        self.parameters.create_ui(window, self.request_reset)

    def start(self) -> None:
        self.playing = True

    def stop(self) -> None:
        self.playing = False

    def request_reset(self) -> None:
        self.reset_requested = True

    def create_simulation_resources(self, width: int, height: int) -> None:
        texture_usage = spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access
        self.agent_data = self.device.create_buffer(
            size=self.parameters.agent_count * 3 * 4,
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
            label="agent_data",
        )
        self.trail_a = self.device.create_texture(
            format=spy.Format.r32_float,
            width=width,
            height=height,
            usage=texture_usage,
            label="trail_a",
        )
        self.trail_b = self.device.create_texture(
            format=spy.Format.r32_float,
            width=width,
            height=height,
            usage=texture_usage,
            label="trail_b",
        )
        self.output_texture = self.device.create_texture(
            format=spy.Format.rgba16_float,
            width=width,
            height=height,
            usage=texture_usage,
            label="output_texture",
        )
        self.resource_signature = (width, height, self.parameters.agent_count)
        self.reset_requested = True

    def reset_simulation(self, command_encoder: spy.CommandEncoder) -> None:
        command_encoder.clear_buffer(self.agent_data)
        command_encoder.clear_texture_float(self.trail_a)
        command_encoder.clear_texture_float(self.trail_b)
        self.simulation_frame = 0
        self.reset_requested = False

    def dispatch_simulation(
        self,
        command_encoder: spy.CommandEncoder,
        delta_time: float,
    ) -> None:
        shader_parameters = self.parameters.shader_values()
        common_vars = {
            "g_params": shader_parameters,
            "g_frame": self.simulation_frame,
            "g_trail_a": self.trail_a,
            "g_trail_b": self.trail_b,
        }
        self.diffuse_kernel.dispatch(
            thread_count=[self.trail_a.width, self.trail_a.height, 1],
            vars=common_vars,
            command_encoder=command_encoder,
        )
        self.update_agents_kernel.dispatch(
            thread_count=[self.parameters.agent_count, 1, 1],
            vars={
                **common_vars,
                "g_delta_time": delta_time,
                "g_agent_data": self.agent_data,
            },
            command_encoder=command_encoder,
        )
        self.simulation_frame += 1

    def dispatch_render(self, command_encoder: spy.CommandEncoder) -> None:
        self.render_kernel.dispatch(
            thread_count=[self.output_texture.width, self.output_texture.height, 1],
            vars={
                "g_frame": self.simulation_frame,
                "g_trail_a": self.trail_a,
                "g_trail_b": self.trail_b,
                "g_output": self.output_texture,
            },
            command_encoder=command_encoder,
        )

    def on_keyboard_event(self, event: spy.KeyboardEvent) -> None:
        self.ui.handle_keyboard_event(event)

    def on_mouse_event(self, event: spy.MouseEvent) -> None:
        self.ui.handle_mouse_event(event)

    def on_resize(self, width: int, height: int) -> None:
        self.device.wait()
        if width > 0 and height > 0:
            self.surface.configure(width=width, height=height, vsync=False)
            self.request_reset()
        else:
            self.surface.unconfigure()

    def run(self) -> None:
        timer = spy.Timer()

        while not self.window.should_close():
            self.window.process_events()

            elapsed = timer.elapsed_s()
            timer.reset()
            delta_time = min(elapsed, MAX_DELTA_TIME)

            if elapsed > 0.0:
                current_fps = 1.0 / elapsed
                self.fps_average = 0.95 * self.fps_average + 0.05 * current_fps
                self.fps_text.text = f"FPS: {self.fps_average:.2f}"

            if not self.surface.config:
                continue

            surface_texture = self.surface.acquire_next_image()
            if not surface_texture:
                continue

            self.ui.begin_frame(surface_texture.width, surface_texture.height)

            signature = (
                surface_texture.width,
                surface_texture.height,
                self.parameters.agent_count,
            )
            if self.resource_signature != signature:
                self.create_simulation_resources(
                    surface_texture.width,
                    surface_texture.height,
                )

            command_encoder = self.device.create_command_encoder()
            if self.reset_requested:
                self.reset_simulation(command_encoder)
            if self.playing:
                self.dispatch_simulation(command_encoder, delta_time)
            self.dispatch_render(command_encoder)

            command_encoder.blit(surface_texture, self.output_texture)
            self.ui.end_frame(surface_texture, command_encoder)
            self.device.submit_command_buffer(command_encoder.finish())
            del surface_texture
            self.surface.present()


if __name__ == "__main__":
    App().run()
