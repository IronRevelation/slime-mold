# Slime Mold Simulation

A small GPU slime-mold simulation built with Python, SlangPy, and four compute
shader passes:

1. Paint food or repellent fields from mouse input.
2. Diffuse and decay the existing trail map.
3. Sense nearby trails and painted fields, move the agents, consume fields, and
   deposit new trails.
4. Convert trails and painted fields into the displayed image.

## Compile-time configurable parameters
In ```slime_molg.slang``` it is possible to set the following constants to change the agents behavior during the simulation:
* ```SPECIES_COUNT```: The number of species initialized. IMPORTANT: this parameter must be changed accordingly also in ```main.py```.
* ```RIGID_WALLS```: If 1, agents bounces back when hitting the border of the frame. Otherwise, a wrap around is performed and the agent hitting the right/bottom side of the window will appear at left/top side in the next frame.
* ```INIT_MODE```: Defines the initial location of the species.
  * If '1', every agent is initialized in a ''almost-random'' position within the whole image.
  * If '2', every agent is initialized in a ''almost-random'' position in a circle on the center of the window.
  * If '3', agents belonging to the same species are initialized in a cirlce, each placed in a dedicated location.


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

## Examples
### Single species

https://github.com/user-attachments/assets/6ffd84c8-e846-453c-93a3-d6c8e920a4d4

### Two species - random initialization

https://github.com/user-attachments/assets/83aad268-88b4-4147-bf73-610b1c9e0725



