#ifndef PBR_GLSL
#define PBR_GLSL

// ----------------------------------------------------------------------------
// GGX Distribution
// ----------------------------------------------------------------------------
float D_GGX(float NdotH, float roughness)
{
    float a = roughness * roughness;
    float a2 = a * a;

    float denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

// ----------------------------------------------------------------------------
// Smith Geometry
// ----------------------------------------------------------------------------
float G_SchlickGGX(float NdotV, float roughness)
{
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;

    return NdotV / (NdotV * (1.0 - k) + k);
}

float G_Smith(float NdotV, float NdotL, float roughness)
{
    return
        G_SchlickGGX(NdotV, roughness) *
        G_SchlickGGX(NdotL, roughness);
}

// ----------------------------------------------------------------------------
// Fresnel
// ----------------------------------------------------------------------------
vec3 F_Schlick(float cosTheta, vec3 F0)
{
    return F0 +
           (1.0 - F0) *
           pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// ----------------------------------------------------------------------------
// Cosine Hemisphere Sampling
// ----------------------------------------------------------------------------
vec3 SampleCosineHemisphere(vec2 u, vec3 N)
{
    float phi = 2.0 * PI * u.x;

    float cosTheta = sqrt(1.0 - u.y);
    float sinTheta = sqrt(u.y);

    vec3 local =
    vec3(
        cos(phi) * sinTheta,
        sin(phi) * sinTheta,
        cosTheta
    );

    vec3 up = abs(N.z) < 0.999
        ? vec3(0,0,1)
        : vec3(1,0,0);

    vec3 T = normalize(cross(up, N));
    vec3 B = cross(N, T);

    return normalize(
        T * local.x +
        B * local.y +
        N * local.z
    );
}

// ----------------------------------------------------------------------------
// GGX Sampling
// ----------------------------------------------------------------------------
vec3 SampleGGX(vec2 u, vec3 N, float roughness)
{
    float a = roughness * roughness;

    float phi = 2.0 * PI * u.x;

    float cosTheta =
        sqrt((1.0 - u.y) /
             (1.0 + (a * a - 1.0) * u.y));

    float sinTheta =
        sqrt(max(0.0, 1.0 - cosTheta * cosTheta));

    vec3 H =
    vec3(
        cos(phi) * sinTheta,
        sin(phi) * sinTheta,
        cosTheta
    );

    vec3 up =
        abs(N.z) < 0.999
        ? vec3(0,0,1)
        : vec3(1,0,0);

    vec3 T = normalize(cross(up, N));
    vec3 B = cross(N, T);

    return normalize(
        T * H.x +
        B * H.y +
        N * H.z
    );
}

// ----------------------------------------------------------------------------

struct PBRMaterial
{
    vec3 albedo;
    float roughness;
    float metallic;
};

// ----------------------------------------------------------------------------

vec3 EvaluatePBR(
    PBRMaterial mat,
    vec3 N,
    vec3 V,
    inout uint seed,
    out vec3 scatterDir,
    out float pdf)
{
    float roughness = clamp(mat.roughness, 0.05, 1.0);

    vec2 u =
    vec2(
        randomFloat(seed),
        randomFloat(seed)
    );

    // ---------------------------------------------------------
    // Metallic
    // ---------------------------------------------------------

    if(mat.metallic > 0.5)
    {
        vec3 H = SampleGGX(u, N, roughness);

        scatterDir = reflect(-V, H);

        float NdotL = max(dot(N, scatterDir), 0.0);

        if(NdotL <= 0.0)
        {
            pdf = 0.0;
            return vec3(0.0);
        }

        float NdotV = max(dot(N, V), 0.0001);
        float NdotH = max(dot(N, H), 0.0001);
        float VdotH = max(dot(V, H), 0.0001);

        vec3 F0 = mat.albedo;

        float D = D_GGX(NdotH, roughness);
        float G = G_Smith(NdotV, NdotL, roughness);

        vec3 F = F_Schlick(VdotH, F0);

        pdf = (D * NdotH) /
              max(4.0 * VdotH, 0.0001);

        return
            (D * G * F) /
            max(4.0 * NdotV * NdotL, 0.0001);
    }

    // ---------------------------------------------------------
    // Diffuse
    // ---------------------------------------------------------

    scatterDir = normalize(N + randomInUnitSphere(seed));

    float NdotL = max(dot(N, scatterDir), 0.0);

    pdf = max(NdotL / PI, 0.0001);

    return mat.albedo / PI;
}

#endif