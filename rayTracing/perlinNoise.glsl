uint wangHash(uint seed)
{
    seed = (seed ^ 61u) ^ (seed >> 16u);
    seed *= 9u;
    seed = seed ^ (seed >> 4u);
    seed *= 0x27d4eb2du;
    seed = seed ^ (seed >> 15u);
    return seed;
}

uint cellHash(ivec3 cell)
{
    uint seed = uint(cell.x) * 73856093u ^ uint(cell.y) * 19349663u ^ uint(cell.z) * 83492791u;

    return wangHash(seed);
}

float blockNoise(vec3 p)
{
    ivec3 cell = ivec3(floor(p));
    uint seed = cellHash(cell);

    return randomFloat(seed); 
}

float fade(float t)
{
    return t * t * (3.0 - 2.0 * t);
}

float lerp(float a, float b, float t)
{
    return a + t * (b - a);
}

float valueNoise(vec3 p)
{
    ivec3 cell = ivec3(floor(p));
    vec3 localPos = p - vec3(cell); // fractional position within the cell

    // Fade (smoothstep-like)
    vec3 fadeXYZ = vec3(fade(localPos.x), fade(localPos.y), fade(localPos.z));

    // Hash the 8 corners
    uint seed000 = cellHash(cell + ivec3(0, 0, 0));
    float c000 = randomFloat(seed000);

    uint seed100 = cellHash(cell + ivec3(1, 0, 0));
    float c100 = randomFloat(seed100);

    uint seed010 = cellHash(cell + ivec3(0, 1, 0));
    float c010 = randomFloat(seed010);

    uint seed110 = cellHash(cell + ivec3(1, 1, 0));
    float c110 = randomFloat(seed110);

    uint seed001 = cellHash(cell + ivec3(0, 0, 1));
    float c001 = randomFloat(seed001);

    uint seed101 = cellHash(cell + ivec3(1, 0, 1));
    float c101 = randomFloat(seed101);

    uint seed011 = cellHash(cell + ivec3(0, 1, 1));
    float c011 = randomFloat(seed011);

    uint seed111 = cellHash(cell + ivec3(1, 1, 1));
    float c111 = randomFloat(seed111);

    // Interpolate along x
    float x00 = lerp(c000, c100, fadeXYZ.x);
    float x10 = lerp(c010, c110, fadeXYZ.x);
    float x01 = lerp(c001, c101, fadeXYZ.x);
    float x11 = lerp(c011, c111, fadeXYZ.x);

    // Interpolate along y
    float y0 = lerp(x00, x10, fadeXYZ.y);
    float y1 = lerp(x01, x11, fadeXYZ.y);

    // Interpolate along z
    return lerp(y0, y1, fadeXYZ.z);
}

float fbm(vec3 p)
{
    float total = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;
    float persistence = 0.5; // how quickly amplitude decreases
    int octaves = 5;

    for (int i = 0; i < octaves; ++i)
    {
        total += valueNoise(p * frequency) * amplitude;
        frequency *= 2.0;  // Increase frequency each octave
        amplitude *= persistence; // Decrease amplitude each octave
    }

    return total;
}

float turbulence(vec3 p)
{
    float total = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;
    float persistence = 0.5;
    int octaves = 5;

    for (int i = 0; i < octaves; ++i)
    {
        total += abs(valueNoise(p * frequency) - 0.5) * 2.0 * amplitude;
        frequency *= 2.0;
        amplitude *= persistence;
    }

    return total;
}

float marbleTexture(vec3 p, float frequency, float turbulenceAmplitude)
{
    return sin(frequency * p.x + turbulenceAmplitude * turbulence(p));
}