# Slime Mold Simulation

A small GPU slime-mold simulation built with Python, SlangPy, and three compute
shader passes:

1. Diffuse and decay the existing trail map.
2. Sense nearby trails, move the agents, and deposit new trails.
3. Convert trail intensity into the displayed image.

## Run

Install the locked dependencies and launch the application with:

```shell
uv run src/main.py
```

## Project structure

- `src/main.py` owns the window, GPU resources, simulation lifecycle, and the
  three-pass dispatch sequence.
- `src/simulation_parameters.py` declares adjustable values and generates their
  UI sliders and shader parameter dictionary.
- `src/shaders/test1.slang` contains the simulation and rendering kernels.

## Add a parameter

1. Add a typed field with `slider(...)` metadata to `SimulationParameters`.
2. Add the same field to the shader's `SimulationParameters` struct and use it.

The Python declaration supplies the default, label, range, format, UI control,
and shader binding automatically.

Reset clears the trails and respawns agents without changing slider values.
Changing **Agent Count** or resizing the window also resets the simulation,
because those changes require new GPU resources.
