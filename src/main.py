from pathlib import Path
import slangpy as spy
from simulation_parameters import SimulationParameters

SHADER_DIR = Path(__file__).parent / "shaders"
MAX_DELTA_TIME = 0.05

# Number of species
SPECIES_COUNT = 2

# Every agent is characterized by 4 parameters: x position, y position, angle, species
N_PARAMS_PER_AGENT = 4

class App:
    def __init__(self):
        self.window = spy.Window(
            width=1280,
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
        self.paint_kernel = self._load_kernel("paint_field")
        self.render_kernel = self._load_kernel("render_image")

        self.parameters = SimulationParameters()
        self.agent_data = None
        self.trail_a = None
        self.trail_b = None
        self.food = None
        self.repellent = None
        self.output_texture = None
        self.resource_signature = None

        self.playing = True
        self.reset_requested = True
        self.simulation_frame = 0
        self.fps_average = 0.0
        self.brush_mode = "food"
        self.brush_radius = 5.0
        self.mouse_position = spy.float2(0.0, 0.0)
        self.previous_brush_position = None
        self.mouse_down = False
        self.mouse_over_ui = False

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
            size=spy.float2(320, 560),
        )
        self.fps_text = spy.ui.Text(window, "FPS: 0")

        # == Buttons ==
        spy.ui.Button(window, "Start", callback=self.start)
        spy.ui.Button(window, "Stop", callback=self.stop)
        spy.ui.Button(window, "Reset", callback=self.request_reset)
        self.brush_text = spy.ui.Text(window, "Brush: Food")
        spy.ui.Button(window, "Food Brush", callback=lambda: self.select_brush("food"))
        spy.ui.Button(
            window,
            "Repellent Brush",
            callback=lambda: self.select_brush("repellent"),
        )
        # =============

        # == Sliders ==
        spy.ui.SliderFloat(
            window,
            "Brush Radius",
            value=self.brush_radius,
            min=1.0,
            max=10.0,
            format="%.0f px",
            callback=self.set_brush_radius,
        )
        self.parameters.create_ui(window, self.request_reset)
        # == Buttons ==

    def start(self) -> None:
        self.playing = True

    def stop(self) -> None:
        self.playing = False

    def request_reset(self) -> None:
        self.reset_requested = True

    def select_brush(self, mode: str) -> None:
        self.brush_mode = mode
        self.brush_text.text = f"Brush: {mode.title()}"

    def set_brush_radius(self, radius: float) -> None:
        self.brush_radius = radius

    def create_simulation_resources(self, width: int, height: int) -> None:
        texture_usage = spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access

        self.agent_data = self.device.create_buffer(
            size=self.parameters.agent_count * N_PARAMS_PER_AGENT * 4,
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
            label="agent_data",
        )

        #trails are 2D textures with a layer per species
        self.trail_a = self.device.create_texture(
            type=spy.TextureType.texture_2d_array,
            format=spy.Format.r32_float,
            width=width,
            height=height,
            array_length=SPECIES_COUNT,
            usage=texture_usage,
            label="trail_a",
        )
        self.trail_b = self.device.create_texture(
            type=spy.TextureType.texture_2d_array,
            format=spy.Format.r32_float,
            width=width,
            height=height,
            array_length=SPECIES_COUNT,
            usage=texture_usage,
            label="trail_b",
        )

        # Food and repellent are simple 2D texture, shared by all species
        self.food = self.device.create_texture(
            format=spy.Format.r32_float,
            width=width,
            height=height,
            usage=texture_usage,
            label="food",
        )
        self.repellent = self.device.create_texture(
            format=spy.Format.r32_float,
            width=width,
            height=height,
            usage=texture_usage,
            label="repellent",
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
        command_encoder.clear_texture_float(self.food)
        command_encoder.clear_texture_float(self.repellent)
        self.simulation_frame = 0
        self.reset_requested = False

    def dispatch_paint(self, command_encoder: spy.CommandEncoder) -> None:
        brush_start = (
            self.previous_brush_position
            if self.previous_brush_position is not None
            else self.mouse_position
        )
        self.paint_kernel.dispatch(
            thread_count=[self.food.width, self.food.height, 1],
            vars={
                "g_food": self.food,
                "g_repellent": self.repellent,
                "g_brush_start_position": brush_start,
                "g_brush_position": self.mouse_position,
                "g_brush_radius": self.brush_radius,
                "g_brush_mode": 0 if self.brush_mode == "food" else 1,
            },
            command_encoder=command_encoder,
        )
        self.previous_brush_position = self.mouse_position

    def dispatch_simulation(
        self,
        command_encoder: spy.CommandEncoder,
        delta_time: float,
    ) -> None:
        shader_parameters = self.parameters.shader_values()
        # Invariato: i nomi ("g_trail_a", "g_trail_b") restano gli stessi,
        # slangpy passa semplicemente l'oggetto texture (ora array) allo
        # stesso identico modo con cui passava la texture 2D prima.
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
                "g_food": self.food,
                "g_repellent": self.repellent,
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
                "g_food": self.food,
                "g_repellent": self.repellent,
                "g_output": self.output_texture,
            },
            command_encoder=command_encoder,
        )

    def on_keyboard_event(self, event: spy.KeyboardEvent) -> None:
        self.ui.handle_keyboard_event(event)

    def on_mouse_event(self, event: spy.MouseEvent) -> None:
        consumed_by_ui = self.ui.handle_mouse_event(event)

        if event.type in (
            spy.MouseEventType.move,
            spy.MouseEventType.button_down,
            spy.MouseEventType.button_up,
        ):
            self.mouse_position = spy.float2(float(event.pos.x), float(event.pos.y))
            self.mouse_over_ui = consumed_by_ui
            if consumed_by_ui:
                self.previous_brush_position = None

        if event.type == spy.MouseEventType.button_up:
            if event.button == spy.MouseButton.left:
                self.mouse_down = False
                self.previous_brush_position = None
        elif (
            event.type == spy.MouseEventType.button_down
            and event.button == spy.MouseButton.left
            and not consumed_by_ui
        ):
            self.mouse_down = True
            self.previous_brush_position = self.mouse_position

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
            if self.mouse_down and not self.mouse_over_ui:
                self.dispatch_paint(command_encoder)
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