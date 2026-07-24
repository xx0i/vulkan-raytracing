#extension GL_EXT_control_flow_attributes : require

const float PI = 3.14159265359;

uint randomSeed(uint val0, uint val1)
{
    uint v0 = val0, v1 = val1, s0 = 0;

    [[unroll]] 
    for (uint n = 0; n < 16; n++)
    {
        s0 += 0x9e3779b9u;
        v0 += ((v1 << 4) + 0xa341316cu) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4u);
        v1 += ((v0 << 4) + 0xad90777du) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761eu);
    }

    return v0;
}

uint randomInt(inout uint seed)
{
    // PCG Hash: Far superior quality to simple LCG, avoids zero-state locks
    seed = seed * 747796405u + 2891336453u;
    uint word = ((seed >> ((seed >> 28u) + 4u)) ^ seed) * 277803737u;
    return (word >> 22u) ^ word;
}

float randomFloat(inout uint seed)
{
    return float(randomInt(seed)) * (1.0 / 4294967296.0); // Exact [0.0, 1.0) range
}

// Direct Unit Vector on Sphere (No loops!)
vec3 randomUnitVector(inout uint seed)
{
    float z = randomFloat(seed) * 2.0 - 1.0;
    float a = randomFloat(seed) * 2.0 * PI;
    float r = sqrt(max(0.0, 1.0 - z * z));
    return vec3(r * cos(a), r * sin(a), z);
}

// Cosine-Weighted Hemisphere Sample (Ideal for Lambertian Diffuse)
vec3 sampleHemisphereCosine(vec3 normal, inout uint seed)
{
    vec3 u = randomUnitVector(seed);
    return normalize(normal + u);
}

vec3 randomInUnitSphere(inout uint seed)
{
    float u1 = randomFloat(seed);
    float u2 = randomFloat(seed);
    float u3 = randomFloat(seed);

    float r = pow(u1, 1.0 / 3.0);
    float theta = u2 * 2.0 * 3.14159265359;
    float phi = acos(2.0 * u3 - 1.0);

    return vec3(
        r * sin(phi) * cos(theta),
        r * sin(phi) * sin(theta),
        r * cos(phi)
    );
}