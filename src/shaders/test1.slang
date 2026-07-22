import playground;
import rendering;


#define NUM_AGENTS      4096
#define THREADS_1D      64
#define THREADS_2D      16

// -------------------- PARAMETRI REGOLABILI --------------------
// Gli slider appaiono nella UI del playground, comodissimi per il tuning.

[playground::SLIDER(60.0, 10.0, 200.0)]
uniform float agentSpeed;

[playground::SLIDER(0.4, 0.05, 1.2)]
uniform float sensorAngle;

[playground::SLIDER(12.0, 2.0, 40.0)]
uniform float sensorDist;

[playground::SLIDER(4.0, 0.5, 10.0)]
uniform float turnSpeed;

[playground::SLIDER(0.3, 0.0, 1.0)]
uniform float randomTurnAmount;

[playground::SLIDER(5.0, 0.5, 20.0)]
uniform float depositAmount;

[playground::SLIDER(0.995, 0.90, 0.999)]
uniform float decayRate;

[playground::SLIDER(0.6, 0.0, 1.0)]
uniform float diffuseRate;

[playground::TIME]
uniform float time;

[playground::FRAME_ID]
uniform float frame;


// -------------------- STATO PERSISTENTE --------------------

// Ogni agente occupa 3 float consecutivi: x, y, angle.
[playground::ZEROS(NUM_AGENTS * 3)]
RWStructuredBuffer<float> agentData;

// Due texture per il ping-pong: una è "quella di prima", l'altra
// "quella nuova". Si scambiano ruolo in base alla parità del frame.
// NOTA: WebGPU non supporta texture storage "read_write" con più di
// un componente (float4/rgba32float), quindi usiamo un solo canale
// (float) per l'intensità della trail: ci basta comunque, non ci
// serve un colore RGB separato per canale in questa fase.
[playground::BLACK_SCREEN(1.0, 1.0)]
RWTexture2D<float> trailA;

[playground::BLACK_SCREEN(1.0, 1.0)]
RWTexture2D<float> trailB;


// -------------------- UTILITY: RNG --------------------

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

float randomFloat01(inout uint seed)
{
    seed = hash(seed);
    return float(seed) / 4294967295.0;
}


// -------------------- HELPER: quale texture è "read" e quale "write" --------------------
// Su frame pari: trailA = read, trailB = write. Su frame dispari: invertite.

bool isEvenFrame()
{
    return (uint(frame) % 2u) == 0u;
}


// -------------------- PASS 1: DIFFUSE + DECAY --------------------
// Un thread per texel. Legge dalla texture "read", scrive su quella "write".

void diffuseDecayBody(RWTexture2D<float> src, RWTexture2D<float> dst, int2 pixel, uint2 size)
{
    float sum = 0.0;
    for (int dx = -1; dx <= 1; dx++)
    {
        for (int dy = -1; dy <= 1; dy++)
        {
            int2 samplePos = clamp(pixel + int2(dx, dy), int2(0, 0), int2(size) - int2(1, 1));
            sum += src[samplePos];
        }
    }
    float blurred = sum / 9.0;
    float original = src[pixel];
    float diffused = lerp(original, blurred, diffuseRate);
    dst[pixel] = diffused * decayRate;
}

[shader("compute")]
[numthreads(THREADS_2D, THREADS_2D, 1)]
[playground::CALL::SIZE_OF("trailA")]
void diffuseDecay(uint2 dispatchThreadID: SV_DispatchThreadID)
{
    uint2 size;
    trailA.GetDimensions(size.x, size.y);
    int2 pixel = int2(dispatchThreadID);
    if (pixel.x >= int(size.x) || pixel.y >= int(size.y))
        return;

    if (isEvenFrame())
        diffuseDecayBody(trailA, trailB, pixel, size);
    else
        diffuseDecayBody(trailB, trailA, pixel, size);
}


// -------------------- PASS 2: AGENT UPDATE --------------------
// Un thread per agente. Sensori sulla texture "write" (già diffusa
// in questo stesso frame), poi deposita sulla stessa texture "write".

float senseAt(RWTexture2D<float> tex, float2 pos, uint2 size)
{
    int2 p = clamp(int2(pos), int2(0, 0), int2(size) - int2(1, 1));
    return tex[p];
}

float sense(RWTexture2D<float> tex, float2 pos, float angle, float sensorAngleOffset, uint2 size)
{
    float a = angle + sensorAngleOffset;
    float2 dir = float2(cos(a), sin(a));
    return senseAt(tex, pos + dir * sensorDist, size);
}

void agentStep(RWTexture2D<float> tex, uint id, uint2 size)
{
    float x = agentData[id * 3 + 0];
    float y = agentData[id * 3 + 1];
    float angle = agentData[id * 3 + 2];

    // Inizializzazione al primo frame: posizione casuale, angolo casuale.
    if (uint(frame) == 0u)
    {
        uint seed0 = id * 9781u + 1u;
        x = randomFloat01(seed0) * float(size.x);
        y = randomFloat01(seed0) * float(size.y);
        angle = randomFloat01(seed0) * 6.2831853;
    }

    uint seed = id * 9781u + uint(frame) * 6151u + 1u;

    float2 pos = float2(x, y);
    float forwardVal = sense(tex, pos, angle, 0.0, size);
    float leftVal     = sense(tex, pos, angle, +sensorAngle, size);
    float rightVal    = sense(tex, pos, angle, -sensorAngle, size);

    float steer = 0.0;
    if (forwardVal > leftVal && forwardVal > rightVal)
    {
        steer = 0.0;
    }
    else if (leftVal > rightVal)
    {
        steer = 1.0;
    }
    else if (rightVal > leftVal)
    {
        steer = -1.0;
    }
    else
    {
        steer = randomFloat01(seed) > 0.5 ? 1.0 : -1.0;
    }

    float randomTurn = (randomFloat01(seed) * 2.0 - 1.0) * randomTurnAmount;
    angle += (steer * turnSpeed + randomTurn) * 0.016; // ~1 frame a 60fps

    float2 dir = float2(cos(angle), sin(angle));
    pos += dir * agentSpeed * 0.016;

    // Wrap-around ai bordi
    pos.x = fmod(pos.x + float(size.x), float(size.x));
    pos.y = fmod(pos.y + float(size.y), float(size.y));

    agentData[id * 3 + 0] = pos.x;
    agentData[id * 3 + 1] = pos.y;
    agentData[id * 3 + 2] = angle;

    // Deposito (nota: scritture concorrenti sullo stesso texel da
    // agenti diversi possono correre l'una sull'altra; per una demo
    // l'artefatto è trascurabile, ma è un punto migliorabile con
    // buffer atomici se serve rigore).
    int2 pixel = int2(pos);
    tex[pixel] = tex[pixel] + depositAmount;
}

[shader("compute")]
[numthreads(THREADS_1D, 1, 1)]
[playground::CALL(NUM_AGENTS / THREADS_1D, 1, 1)]
void updateAgents(uint dispatchThreadID: SV_DispatchThreadID)
{
    uint id = dispatchThreadID;
    if (id >= NUM_AGENTS)
        return;

    uint2 size;
    trailA.GetDimensions(size.x, size.y);

    if (isEvenFrame())
        agentStep(trailB, id, size); // trailB è la "write" di questo frame
    else
        agentStep(trailA, id, size);
}


// -------------------- PASS 3: RENDER --------------------

[shader("compute")]
[numthreads(THREADS_2D, THREADS_2D, 1)]
[playground::CALL::SIZE_OF("outputTexture")]
void imageMain(uint2 dispatchThreadID: SV_DispatchThreadID)
{
    drawPixel(dispatchThreadID, (int2 screenSize) => {
        uint2 size;
        trailA.GetDimensions(size.x, size.y);

        int2 pixel = int2(dispatchThreadID);
        float intensity = isEvenFrame() ? trailB[pixel] : trailA[pixel];

        float3 col = lerp(float3(0.0, 0.0, 0.02), float3(0.1, 0.9, 0.6), saturate(intensity));
        return float4(col, 1.0);
    });
}
