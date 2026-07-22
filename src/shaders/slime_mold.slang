// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#define THREADS_1D 64
#define THREADS_2D 16

struct SimulationParameters
{
    uint agent_count;
    float agent_speed;
    float sensor_angle;
    float sensor_distance;
    float turn_speed;
    float random_turn_amount;
    float deposit_amount;
    float decay_rate;
    float diffuse_rate;
};

uniform SimulationParameters g_params;
uniform uint g_frame;
uniform float g_delta_time;

RWStructuredBuffer<float> g_agent_data;
RWTexture2D<float> g_trail_a;
RWTexture2D<float> g_trail_b;

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

void diffuse_decay_body(
    RWTexture2D<float> source,
    RWTexture2D<float> destination,
    int2 pixel,
    uint2 size
)
{
    float sum = 0.0;
    for (int dx = -1; dx <= 1; ++dx)
    {
        for (int dy = -1; dy <= 1; ++dy)
        {
            int2 sample_position = clamp(
                pixel + int2(dx, dy),
                int2(0, 0),
                int2(size) - int2(1, 1)
            );
            sum += source[sample_position];
        }
    }

    float blurred = sum / 9.0;
    float diffused = lerp(source[pixel], blurred, g_params.diffuse_rate);
    destination[pixel] = diffused * g_params.decay_rate;
}

[shader("compute")]
[numthreads(THREADS_2D, THREADS_2D, 1)]
void diffuse_decay(uint3 dispatch_thread_id: SV_DispatchThreadID)
{
    uint2 size;
    g_trail_a.GetDimensions(size.x, size.y);
    int2 pixel = int2(dispatch_thread_id.xy);
    if (any(dispatch_thread_id.xy >= size))
        return;

    if (is_even_frame())
        diffuse_decay_body(g_trail_a, g_trail_b, pixel, size);
    else
        diffuse_decay_body(g_trail_b, g_trail_a, pixel, size);
}

float sense_at(RWTexture2D<float> trail, float2 position, uint2 size)
{
    int2 pixel = clamp(int2(position), int2(0, 0), int2(size) - int2(1, 1));
    return trail[pixel];
}

float sense(
    RWTexture2D<float> trail,
    float2 position,
    float angle,
    float sensor_angle_offset,
    uint2 size
)
{
    float sensor_direction = angle + sensor_angle_offset;
    float2 direction = float2(cos(sensor_direction), sin(sensor_direction));
    return sense_at(trail, position + direction * g_params.sensor_distance, size);
}

void update_agent(RWTexture2D<float> trail, uint id, uint2 size)
{
    float x = g_agent_data[id * 3 + 0];
    float y = g_agent_data[id * 3 + 1];
    float angle = g_agent_data[id * 3 + 2];

    if (g_frame == 0u)
    {
        uint initial_seed = id * 9781u + 1u;
        x = random_float_01(initial_seed) * float(size.x);
        y = random_float_01(initial_seed) * float(size.y);
        angle = random_float_01(initial_seed) * 6.2831853;
    }

    uint seed = id * 9781u + g_frame * 6151u + 1u;
    float2 position = float2(x, y);
    float forward_value = sense(trail, position, angle, 0.0, size);
    float left_value = sense(trail, position, angle, g_params.sensor_angle, size);
    float right_value = sense(trail, position, angle, -g_params.sensor_angle, size);

    float steering = 0.0;
    if (forward_value > left_value && forward_value > right_value)
        steering = 0.0;
    else if (left_value > right_value)
        steering = 1.0;
    else if (right_value > left_value)
        steering = -1.0;
    else
        steering = random_float_01(seed) > 0.5 ? 1.0 : -1.0;

    float random_turn = (random_float_01(seed) * 2.0 - 1.0) * g_params.random_turn_amount;
    angle += (steering * g_params.turn_speed + random_turn) * g_delta_time;

    float2 direction = float2(cos(angle), sin(angle));
    position += direction * g_params.agent_speed * g_delta_time;
    position.x = fmod(position.x + float(size.x), float(size.x));
    position.y = fmod(position.y + float(size.y), float(size.y));

    g_agent_data[id * 3 + 0] = position.x;
    g_agent_data[id * 3 + 1] = position.y;
    g_agent_data[id * 3 + 2] = angle;

    // Multiple agents can deposit into the same texel. These writes are
    // deliberately non-atomic for this lightweight visual demo.
    int2 pixel = int2(position);
    trail[pixel] = trail[pixel] + g_params.deposit_amount;
}

[shader("compute")]
[numthreads(THREADS_1D, 1, 1)]
void update_agents(uint3 dispatch_thread_id: SV_DispatchThreadID)
{
    uint id = dispatch_thread_id.x;
    if (id >= g_params.agent_count)
        return;

    uint2 size;
    g_trail_a.GetDimensions(size.x, size.y);
    if (is_even_frame())
        update_agent(g_trail_b, id, size);
    else
        update_agent(g_trail_a, id, size);
}

[shader("compute")]
[numthreads(THREADS_2D, THREADS_2D, 1)]
void render_image(uint3 dispatch_thread_id: SV_DispatchThreadID)
{
    uint2 size;
    g_output.GetDimensions(size.x, size.y);
    uint2 pixel = dispatch_thread_id.xy;
    if (any(pixel >= size))
        return;

    float intensity = is_even_frame() ? g_trail_a[pixel] : g_trail_b[pixel];
    float3 color = lerp(
        float3(0.0, 0.0, 0.02),
        float3(0.1, 0.9, 0.6),
        saturate(intensity)
    );
    g_output[pixel] = float4(color, 1.0);
}
