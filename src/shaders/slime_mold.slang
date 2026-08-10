#define THREADS_1D 64
#define THREADS_2D 16

#define N_PARAMS_PER_AGENT 4

// Number of species and their colors
#define SPECIES_COUNT 4
#define RIGID_WALLS 1

/*
INIT_MODE = 1 --> All species are initialized at random positions
INIT_MODE = 2 --> All species are initialized in a circle at the center of the windows
INIT_MODE = 3 --> Every specie is initialized in a circle in the (up to 4) corners of the window
*/
#define INIT_MODE 3

// List of colors: 10 species maximum are allowed
static const float3 species_colors[10] = {
    float3(0.00, 1.0, 0.00),
    float3(1.00, 0.00, 0.00),
    float3(0.00, 0.0, 1.0),
    float3(0.80, 0.20, 1.00),
    float3(0.10, 1.00, 0.00),
    float3(0.20, 0.40, 1.00),
    float3(1.00, 0.50, 0.00),
    float3(0.00, 1.00, 1.00),
    float3(1.00, 1.00, 1.00),
    float3(0.50, 0.50, 0.50),};

// Simulation parameters (from sliders)
struct SimulationParameters
{
    uint agent_count;
    float agent_speed;
    float sensor_angle;
    float sensor_distance;
    float turn_speed;
    float random_turn_amount;
    float deposit_amount;
    float food_strength;
    float repellent_strength;
    float decay_rate;
    float diffuse_rate;
    float species_repellent;
};

// Read-only values provided by the main
uniform SimulationParameters g_params;
uniform uint g_frame;
uniform float g_delta_time;

// Texture with 'SPECIES_COUNT' layers (one per species)
RWTexture2DArray<float> g_trail_a;
RWTexture2DArray<float> g_trail_b;

RWStructuredBuffer<float> g_agent_data;
RWTexture2D<float> g_food;
RWTexture2D<float> g_repellent;

// Walls (Binary mask with only two values: 0.0 or 1.0)
RWTexture2D<float> g_walls;

uniform float2 g_brush_start_position;
uniform float2 g_brush_position;
uniform float g_brush_radius;
uniform uint g_brush_mode;

[format("rgba16f")]
RWTexture2D<float4> g_output;


uint hash(uint state)
{
    state ^= 2747636419u;
    state *= 2654435769u;
    state ^= state >> 16;
    state *= 2654435769u;
    state ^= state >> 16;
    state *= 2654435769u;
    return state;
}

float random_float_01(inout uint seed)
{
    seed = hash(seed);
    return float(seed) / 4294967295.0;
}

bool is_even_frame()
{
    return (g_frame % 2u) == 0u;
}

bool is_wall(int2 pixel, uint2 size)
{
    // This method takes the position of a pixel an returns True is there is a wall in such a position
    pixel = clamp(pixel, int2(0, 0), int2(size) - int2(1, 1));
    return g_walls[pixel] > 0.5;
}

[shader("compute")]
[numthreads(THREADS_2D, THREADS_2D, 1)]
void paint_field(uint3 dispatch_thread_id: SV_DispatchThreadID)
{
    uint2 size;
    g_food.GetDimensions(size.x, size.y);
    uint2 pixel = dispatch_thread_id.xy;

    if (any(pixel >= size))
        return;


    float2 pixel_center = float2(pixel) + 0.5;

    // Vector identifying the painting segment
    float2 brush_segment = g_brush_position - g_brush_start_position;
    float segment_length_squared = dot(brush_segment, brush_segment);

    float segment_position = 0.0;
    if (segment_length_squared > 0.0) // If the mouse has moved
    {
        float projection = dot(pixel_center - g_brush_start_position, brush_segment);
        segment_position = projection / segment_length_squared;
        segment_position = saturate(segment_position);
    }
    else
        segment_position = 0.0;
    

 
    float2 closest_brush_point = g_brush_start_position + brush_segment * segment_position;

    float distance_from_brush = distance(pixel_center, closest_brush_point);

    if (distance_from_brush >= g_brush_radius)
        return;

    // Smoothing the brush
    float falloff = saturate(1.0 - distance_from_brush / g_brush_radius);

    // Adding attraction
    if (g_brush_mode == 0u)
    {
        g_food[pixel] = max(g_food[pixel], falloff);
        g_repellent[pixel] = 0.0;
    }
    // Adding repulsion
    else
    {
        g_repellent[pixel] = max(g_repellent[pixel], falloff);
        g_food[pixel] = 0.0;
    }
}

void diffuse_decay_body(
    RWTexture2DArray<float> source,
    RWTexture2DArray<float> destination,
    int2 pixel,
    uint2 size,
    uint species_index
)
{
    // Iteration over the 3x3 window around the pixel
    float sum = 0.0;
    for (int dx = -1; dx <= 1; ++dx)
    {
        for (int dy = -1; dy <= 1; ++dy)
        {
            // Clamp prevents sample_position from being outside the frame
            int2 sample_position = clamp(pixel + int2(dx, dy), int2(0, 0), int2(size) - int2(1, 1));
            sum += source[uint3(sample_position, species_index)];
        }
    }

    // Average value within the 3x3 window analyzed
    float blurred = sum / 9.0;


    float diffused = lerp(source[uint3(pixel, species_index)], blurred, g_params.diffuse_rate);
    destination[uint3(pixel, species_index)] = diffused * g_params.decay_rate;

    
    //destination[uint3(pixel, species_index)] = max(0.0, diffused - g_params.decay_rate * g_delta_time);
}

[shader("compute")]
[numthreads(THREADS_2D, THREADS_2D, 1)]
void diffuse_decay(uint3 dispatch_thread_id: SV_DispatchThreadID)
{
    // Get the current size of the frame
    uint2 size;
    uint layers;
    g_trail_a.GetDimensions(size.x, size.y, layers);
    int2 pixel = int2(dispatch_thread_id.xy);

    // If the pixel is outside the frame, ignore it
    if (any(dispatch_thread_id.xy >= size))
        return;

    // Run diffuse+decay for every species layer of the trail array
    for (uint s = 0; s < SPECIES_COUNT; ++s)
    {
        if (is_even_frame())
            diffuse_decay_body(g_trail_a, g_trail_b, pixel, size, s); // Read from A, write to B
        else
            diffuse_decay_body(g_trail_b, g_trail_a, pixel, size, s); // Read from B, write to A
    }
}

float sense_at(RWTexture2DArray<float> trail,float2 position,uint2 size,uint species_index)
{
    int sensor_size = 3;
    int2 center = clamp(int2(position), int2(0, 0), int2(size) - int2(1, 1));
    float food = g_food[center];
    float repellent = g_repellent[center];
    g_food[center] = max(0.0, food - 0.01);
    g_repellent[center] = max(0.0, repellent - 0.01);

    float species_sum = 0.0;
    for (int dx = -sensor_size; dx <= sensor_size; ++dx)
    {
        for (int dy = -sensor_size; dy <= sensor_size; ++dy)
        {
            int2 sample_position = clamp(center + int2(dx, dy), int2(0, 0), int2(size) - int2(1, 1));
            for (uint s = 0; s < SPECIES_COUNT; ++s)
            {
                // Weigth: positive = attraction, negative = repulsion
                float weight = 0;
                if (s == species_index)
                    weight = 1.0;
                else
                    weight = -g_params.species_repellent;

                
                float contribution = trail[uint3(sample_position, s)];
                species_sum += weight * contribution;
            }
        }
    }

    float sample_count = float((2 * sensor_size + 1) * (2 * sensor_size + 1));
    return (species_sum / sample_count) + food * g_params.food_strength - repellent * g_params.repellent_strength;

    //return species_sum + food * g_params.food_strength - repellent * g_params.repellent_strength;
}

float sense(RWTexture2DArray<float> trail,
            float2 position,
            float angle,
            float sensor_angle_offset,
            uint2 size,
            uint species_index)
{
    float sensor_direction = angle + sensor_angle_offset;
    float2 direction = float2(cos(sensor_direction), sin(sensor_direction));
    return sense_at(trail, position + direction * g_params.sensor_distance, size, species_index);
}

void update_agent(RWTexture2DArray<float> trail, uint id, uint2 size, float species)
{
    // Fetch the agent information
    float x = g_agent_data[id * N_PARAMS_PER_AGENT + 0];
    float y = g_agent_data[id * N_PARAMS_PER_AGENT + 1];
    float angle = g_agent_data[id * N_PARAMS_PER_AGENT + 2];

    uint initial_seed = id * 217519u;
    
    
    if (g_frame == 0u)
    {
        // If an agent is initialized in the same position of a wall, simply try a new initialization with a different seed
        int spawn_is_wall = 1;
        while (spawn_is_wall > 0)
        {
            float2 center = float2(size) * 0.5; // Coordinates of the frame center
            initial_seed += spawn_is_wall;
            uint initial_seed_2 = initial_seed + 1u;
        // Every agent is initialized in a random position, regardless of the species
        #if INIT_MODE == 1
            x = random_float_01(initial_seed) * float(size.x);
            y = random_float_01(initial_seed) * float(size.y);
            angle = random_float_01(initial_seed) * 6.2831853;
        // Every agent is initialized in a circle in the center of the frame, regardless of the specie it belongs to
        #elif INIT_MODE == 2
            float spawn_angle = random_float_01(initial_seed) * 6.2831853;
            float spawn_radius = sqrt(random_float_01(initial_seed_2)) * 350;
            float2 offset = float2(cos(spawn_angle), sin(spawn_angle)) * spawn_radius;
            x = center.x + offset.x;
            y = center.y + offset.y;
        // Every agent (up to 4th) is initialized in a circle in one of the 4 quarters of the frame
        #else
            float spawn_angle = random_float_01(initial_seed) * 6.2831853;
            float spawn_radius = sqrt(random_float_01(initial_seed_2)) * 100;
            float2 offset = float2(cos(spawn_angle), sin(spawn_angle)) * spawn_radius;
            float2 quarter = float2(size) * 0.25;
            float2 blob_center;

            switch (int(species))
            {
            case 0:
                blob_center.x = center.x + quarter.x;
                blob_center.y = center.y + quarter.y;
                break;
            case 1:
                blob_center.x = center.x - quarter.x;
                blob_center.y = center.y - quarter.y;
                break;
            case 2:
                blob_center.x = center.x + quarter.x;
                blob_center.y = center.y - quarter.y;
                break;
            case 3:
                blob_center.x = center.x - quarter.x;
                blob_center.y = center.y + quarter.y;
                break;
            default:
                blob_center.x = center.x;
                blob_center.y = center.y;
                break;
            }
            x = blob_center.x + offset.x;
            y = blob_center.y + offset.y;
        #endif

            // Check if the "randomly" computed position coincides with the position of a wall
            if (is_wall(int2(int(x), int(y)), size))
                spawn_is_wall++;
            else
                spawn_is_wall = 0;

            // Initial angle
            float2 initial_direction = normalize(center - float2(x, y));
            angle = atan2(initial_direction.y, initial_direction.x);
        }
    }

    uint species_index = uint(species);

    // Seed is computed using the number of frame to obtain always a different value
    uint seed = id * 9781u + g_frame * 6151u + 1u; 
    float2 position = float2(x, y);

    // Sensing: agent looks in the three directions
    float forward_value = sense(trail, position, angle, 0.0, size, species_index);
    float left_value = sense(trail, position, angle, g_params.sensor_angle, size, species_index);
    float right_value = sense(trail, position, angle, -g_params.sensor_angle, size, species_index);

    // Computation of the new direction
    float steering = 0.0;
    if (forward_value > left_value && forward_value > right_value) // Forward is maximum
        steering = 0.0;
    else if (left_value > right_value) // Left is maximum
        steering = 1.0;
    else // Right is greater or equal to left
        steering = -1.0;

    // Random real-number between -1.0 and 1.0
    float random_turn = (random_float_01(seed) * 2.0 - 1.0) * g_params.random_turn_amount;

    // Final direction = sensing contribution + random contribution
    angle += (steering * g_params.turn_speed + random_turn) * g_delta_time;
    float2 direction = float2(cos(angle), sin(angle));

    
    float2 step = direction * g_params.agent_speed * g_delta_time;
    float2 old_position = position;
    float2 new_position = position + step;

    // Testing the collitions for the two axis
    // Test if, maintaining the y-level constant, the collision have occurred anyway
    bool blocked_x = is_wall(int2(int(new_position.x), int(old_position.y)), size);
    // Test if, maintaining the x-level constant, the collision have occurred anyway
    bool blocked_y = is_wall(int2(int(old_position.x), int(new_position.y)), size);

    // If an agent hits a wall, invert the direction component that caused the collision
    if (blocked_x)
    {
        direction.x = -direction.x;
        position.x = position.x;
    }
    else
        position.x = new_position.x;

    if (blocked_y)
    {
        direction.y = -direction.y;
        position.y = old_position.y;
    }
    else
        position.y = new_position.y;

    if (blocked_x || blocked_y)
        angle = atan2(direction.y, direction.x);
    

#if RIGID_WALLS == 1
    if (position.x < 0.0)
    {
        position.x = 0.0;
        direction.x = -direction.x;
    }
    else if (position.x >= float(size.x))
    {
        position.x = float(size.x) - 1.0;
        direction.x = -direction.x;
    }

    if (position.y < 0.0)
    {
        position.y = 0.0;
        direction.y = -direction.y;
    }
    else if (position.y >= float(size.y))
    {
        position.y = float(size.y) - 1.0;
        direction.y = -direction.y;
    }

    angle = atan2(direction.y, direction.x);
#else
    // Wrap around
    position.x = fmod(position.x + float(size.x), float(size.x));
    position.y = fmod(position.y + float(size.y), float(size.y));
#endif

    // Write the new information in the agent buffer
    g_agent_data[id * N_PARAMS_PER_AGENT + 0] = position.x;
    g_agent_data[id * N_PARAMS_PER_AGENT + 1] = position.y;
    g_agent_data[id * N_PARAMS_PER_AGENT + 2] = angle;
    g_agent_data[id * N_PARAMS_PER_AGENT + 3] = species;

    // Position is discretized into pixels
    uint3 pixel = uint3(uint(position.x), uint(position.y), species_index);
    
    trail[pixel] = trail[pixel] + g_params.deposit_amount;
    //trail[pixel] = min(1.0, trail[pixel] + g_params.deposit_amount);
}

[shader("compute")]
[numthreads(THREADS_1D, 1, 1)]
void update_agents(uint3 dispatch_thread_id: SV_DispatchThreadID)
{
    uint id = dispatch_thread_id.x;
    if (id >= g_params.agent_count)
        return;

    uint2 size;
    uint layers;
    float species;

    if (g_frame == 0u) // Only the first frame
    {
        // Assign species uniformly
        uint species_seed = id * 256123u;
        species = floor(random_float_01(species_seed) * float(SPECIES_COUNT));
    }
    else
        species = g_agent_data[id * N_PARAMS_PER_AGENT + 3]; // Fetch species from agent vector
    

    g_trail_a.GetDimensions(size.x, size.y, layers);

    // Update the correct buffer depending on the parity of the frame number
    if (is_even_frame())
        update_agent(g_trail_b, id, size, species);
    else
        update_agent(g_trail_a, id, size, species);
}

[shader("compute")]
[numthreads(THREADS_2D, THREADS_2D, 1)]
void render_image(uint3 dispatch_thread_id: SV_DispatchThreadID)
{
    // Get the current size of the frame
    uint2 size;
    g_output.GetDimensions(size.x, size.y);
    uint2 pixel = dispatch_thread_id.xy;

    // If the pixel is outside the frame, ignore it
    if (any(pixel >= size))
        return;

    // Background color 
    float3 color = float3(0.0, 0.0, 0.00);

    // Accumulate each species trail contribution, weighted by its own color
    for (uint s = 0; s < SPECIES_COUNT; ++s)
    {
        float intensity;

        if (is_even_frame())
            intensity = g_trail_a[uint3(pixel, s)];
        else
            intensity = g_trail_b[uint3(pixel, s)];

        color += saturate(intensity) * species_colors[s];
    }

    // Accumulate food and repellent
    color += g_food[pixel] * float3(0.75, 0.65, 0.05);
    color += g_repellent[pixel] * float3(0.85, 0.05, 0.35);

    // Visualize walls in grey
    color += g_walls[pixel] * float3(0.3, 0.3, 0.3);

    // Clipping RGB values to 1.0 and setting opacity to 100%
    g_output[pixel] = float4(saturate(color), 1.0);
}
