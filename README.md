# Slime Mold Simulation

A small GPU slime-mold simulation built with Python, SlangPy, and four compute
shader passes:

1. Paint food or repellent fields from mouse input.
2. Diffuse and decay the existing trail map.
3. Sense nearby trails and painted fields, move the agents, consume fields, and
   deposit new trails.
4. Convert trails and painted fields into the displayed image.

## Painting

Use **Food Brush** or **Repellent Brush** in the settings window, then hold and
drag the left mouse button over the simulation. **Brush Radius** changes the
stroke size, while **Food Strength** and **Repellent Strength** independently
control how strongly agents react without changing the paint's appearance.

Paint remains in place until agents sense and consume it. Painting works while
the simulation is stopped, and consumption resumes when it starts again.
**Reset**, resizing the window, or changing the agent count clears all paint.

## Run

Install the locked dependencies and launch the application with:

```shell
uv run src/main.py
```

## Project structure

- `src/main.py` owns the window, GPU resources, simulation lifecycle, and the
  four-pass dispatch sequence.
- `src/simulation_parameters.py` declares adjustable values and generates their
  UI sliders and shader parameter dictionary.
- `src/shaders/slime_mold.slang` contains the simulation and rendering kernels.

## Add a parameter

1. Add a typed field with `slider(...)` metadata to `SimulationParameters`.
2. Add the same field to the shader's `SimulationParameters` struct and use it.

The Python declaration supplies the default, label, range, format, UI control,
and shader binding automatically.

Reset clears the trails and painted fields, and respawns agents without changing
slider values.
Changing **Agent Count** or resizing the window also resets the simulation,
because those changes require new GPU resources.
