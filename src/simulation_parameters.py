from dataclasses import dataclass, field, fields
from typing import Callable
import slangpy as spy

@dataclass(frozen=True)
class SliderSpec:
    label: str
    minimum: int | float
    maximum: int | float
    format: str
    requires_reset: bool = False


def slider(
    default: int | float,
    label: str,
    minimum: int | float,
    maximum: int | float,
    format: str,
    *,
    requires_reset: bool = False,
):
    return field(
        default=default,
        metadata={
            "slider": SliderSpec(
                label=label,
                minimum=minimum,
                maximum=maximum,
                format=format,
                requires_reset=requires_reset,
            )
        },
    )


@dataclass
class SimulationParameters:
    agent_count: int = slider(10000, "Agent Count", 256, 100000, "%d", requires_reset=True)
    agent_speed: float = slider(100.0, "Agent Speed", 10.0, 200.0, "%.1f")
    sensor_angle: float = slider(0.95, "Sensor Angle", 0.05, 1.2, "%.2f")
    sensor_distance: float = slider(10.0, "Sensor Distance", 2.0, 100.0, "%.1f")
    turn_speed: float = slider(7.5, "Turn Speed", 0.5, 10.0, "%.1f")
    random_turn_amount: float = slider(1.0, "Random Turn", 0.0, 1.0, "%.2f")
    deposit_amount: float = slider(5.0, "Deposit Amount", 1, 20.0, "%.1f")
    food_strength: float = slider(10.0, "Food Strength", 0.0, 100.0, "%.1f")
    repellent_strength: float = slider(10.0, "Repellent Strength", 0.0, 100.0, "%.1f")
    decay_rate: float = slider(0.95, "Decay Rate", 0.80, 0.999, "%.3f")
    diffuse_rate: float = slider(0.6, "Diffuse Rate", 0.0, 1.0, "%.2f")
    species_repellent: float = slider(100.0, "Species Repellent", 0.0, 100.0, "%.2f")


    def create_ui(self, parent: spy.ui.Widget, reset_callback: Callable[[], None]) -> None:
        for parameter in fields(self):
            spec: SliderSpec = parameter.metadata["slider"]

            def update(value: int | float, name=parameter.name, slider_spec=spec) -> None:
                setattr(self, name, value)
                if slider_spec.requires_reset:
                    reset_callback()

            value = getattr(self, parameter.name)
            if parameter.type is int:
                spy.ui.SliderInt(
                    parent,
                    spec.label,
                    value=value,
                    min=int(spec.minimum),
                    max=int(spec.maximum),
                    format=spec.format,
                    callback=update,
                )
            elif parameter.type is float:
                spy.ui.SliderFloat(
                    parent,
                    spec.label,
                    value=value,
                    min=float(spec.minimum),
                    max=float(spec.maximum),
                    format=spec.format,
                    callback=update,
                )
            else:
                raise TypeError(f"Unsupported simulation parameter type: {parameter.type}")

    def shader_values(self) -> dict[str, int | float]:
        return {parameter.name: getattr(self, parameter.name) for parameter in fields(self)}
