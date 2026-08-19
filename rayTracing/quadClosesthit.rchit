#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : require

#include "random.glsl"
#include "pbr.glsl"

const uint lambertian   = 0;
const uint metal        = 1;
const uint dielectric   = 2;
const uint isotropic    = 3;
const uint diffuseLight = 4;

const int   NEE_GRID = 2;          // NEE_GRID x NEE_GRID stratified samples (try 1 or 2 first — much cheaper per-sample now)
const float FLIP_BIAS = 0.05;

struct material
{
    vec4 albedo;
    float fuzz;
    float refractionIndex;
    uint matType;
    uint padding;
    vec4 emission;
    vec4 padding2;
};

struct quad 
{
    vec3 origin;
    float pad0;
    vec3 edgeU;
    float pad1;
    vec3 edgeV;
    float pad2;
};

struct aabbObject
{
    uint type;
    uint geoIndex;
    uint matIndex;
    uint _pad0;
};

layout(set = 0, binding = 0) uniform accelerationStructureEXT topLevelAS;
layout(std430, set = 0, binding = 7) buffer materialBuffer { material m[]; } materials;
layout(std430, set = 0, binding = 8) buffer quadBuffer { quad q[]; } quads;
layout(std430, set = 0, binding = 10) buffer aabbObjectsBuffer { aabbObject aabbObj[]; } aabbObjs;

struct rayPayload 
{
    vec3 hitColor;
    vec3 directLight;
    vec3 rayOrigin;
    vec3 rayDir;
    bool hit;
    bool isEmissive;
    vec3 primaryNormal;
    vec3 primaryAlbedo;
    float hitDistance;
    uint bounce;
};

layout(location = 0) rayPayloadInEXT rayPayload payload;
layout(location = 1) rayPayloadEXT bool shadowed;

layout(push_constant) uniform PushConstants 
{
    uint frameIndex;
    uint missColour;
} pc;

bool findLight(out quad lightQuad, out material lightMat)
{
    for (uint i = 0; i < 6; ++i)
    {
        material m = materials.m[i];
        if (m.matType == diffuseLight)
        {
            lightQuad = quads.q[aabbObjs.aabbObj[i].geoIndex];
            lightMat  = m;
            return true;
        }
    }
    return false;
}

// ==================================================
// SPHERICAL RECTANGLE (SOLID-ANGLE) SAMPLING
// Urena, Fajardo, King — "An Area-Preserving Parametrization
// for Spherical Rectangles" (2013). Standard technique for
// low-variance area-light sampling.
// ==================================================
struct SphQuad
{
    vec3 o, x, y, z;
    float z0, z0sq;
    float x0, y0, y0sq;
    float x1, y1, y1sq;
    float b0, b1, b0sq, k;
    float S; // solid angle subtended by the quad, as seen from o
};

SphQuad sphQuadInit(vec3 s, vec3 ex, vec3 ey, vec3 o)
{
    SphQuad sq;
    float exl = length(ex);
    float eyl = length(ey);
    sq.x = ex / exl;
    sq.y = ey / eyl;
    sq.z = cross(sq.x, sq.y);

    vec3 d = s - o;
    sq.z0 = dot(d, sq.z);
    if (sq.z0 > 0.0)
    {
        sq.z  = -sq.z;
        sq.z0 = -sq.z0;
    }
    sq.z0sq = sq.z0 * sq.z0;

    sq.x0 = dot(d, sq.x);
    sq.y0 = dot(d, sq.y);
    sq.x1 = sq.x0 + exl;
    sq.y1 = sq.y0 + eyl;
    sq.y0sq = sq.y0 * sq.y0;
    sq.y1sq = sq.y1 * sq.y1;

    vec3 v00 = vec3(sq.x0, sq.y0, sq.z0);
    vec3 v01 = vec3(sq.x0, sq.y1, sq.z0);
    vec3 v10 = vec3(sq.x1, sq.y0, sq.z0);
    vec3 v11 = vec3(sq.x1, sq.y1, sq.z0);

    vec3 n0 = normalize(cross(v00, v10));
    vec3 n1 = normalize(cross(v10, v11));
    vec3 n2 = normalize(cross(v11, v01));
    vec3 n3 = normalize(cross(v01, v00));

    float g0 = acos(clamp(-dot(n0, n1), -1.0, 1.0));
    float g1 = acos(clamp(-dot(n1, n2), -1.0, 1.0));
    float g2 = acos(clamp(-dot(n2, n3), -1.0, 1.0));
    float g3 = acos(clamp(-dot(n3, n0), -1.0, 1.0));

    sq.b0 = n0.z;
    sq.b1 = n2.z;
    sq.b0sq = sq.b0 * sq.b0;
    sq.k = 2.0 * PI - g2 - g3;

    sq.S = g0 + g1 - sq.k; // solid angle
    sq.o = o;
    return sq;
}

// uv in [0,1]^2 -> world-space point on the quad, area-preserving in solid angle
vec3 sphQuadSample(SphQuad sq, vec2 uv)
{
    float au = uv.x * sq.S + sq.k;
    float fu = (cos(au) * sq.b0 - sq.b1) / sin(au);
    float cu = (fu >= 0.0 ? 1.0 : -1.0) / sqrt(fu * fu + sq.b0sq);
    cu = clamp(cu, -1.0, 1.0);

    float xu = -(cu * sq.z0) / sqrt(max(1.0 - cu * cu, 1e-7));
    xu = clamp(xu, sq.x0, sq.x1);

    float d2 = xu * xu + sq.z0sq;
    float h0 = sq.y0 / sqrt(d2 + sq.y0sq);
    float h1 = sq.y1 / sqrt(d2 + sq.y1sq);
    float hv = h0 + uv.y * (h1 - h0);
    float hv2 = hv * hv;
    float yv = (hv2 < 1.0 - 1e-6) ? (hv * sqrt(d2) / sqrt(max(1.0 - hv2, 1e-7))) : sq.y1;

    return sq.o + xu * sq.x + yv * sq.y + sq.z0 * sq.z;
}

// One NEE sample using solid-angle sampling — replaces the old area-sampled version.
vec3 sampleLightSolidAngle(vec3 hitPos, vec3 N, vec3 albedo, quad lightQuad, material lightMat, vec2 uv)
{
    SphQuad sq = sphQuadInit(lightQuad.origin, lightQuad.edgeU, lightQuad.edgeV, hitPos);

    if (sq.S <= 1e-6)
        return vec3(0.0); // light behind or edge-on to the surface plane

    vec3 lightPos = sphQuadSample(sq, uv);

    vec3 toLight      = lightPos - hitPos;
    float distToLight = length(toLight);
    vec3 L             = toLight / distToLight;

    float cosSurface = dot(N, L);
    if (cosSurface <= 0.0)
        return vec3(0.0);

    float bias = max(0.0005, 0.001 * distToLight);

    shadowed = true;
    traceRayEXT(topLevelAS,
        gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT | gl_RayFlagsSkipClosestHitShaderEXT,
        0xFF, 0, 0, 1,
        hitPos + bias * N, bias,
        L, distToLight - bias * 2.0,
        1);

    if (shadowed)
        return vec3(0.0);

    vec3 lightBrdf = max(albedo, vec3(0.0)) / PI;

    // pdf (solid angle measure) = 1 / S, so contribution = Le * brdf * cosSurface * S.
    // Note: no distance^2 or cosLight term here — they're absorbed into S itself,
    // which is exactly what removes the near-field variance blow-up.
    return (lightMat.emission.rgb * lightMat.emission.a) * lightBrdf * cosSurface * sq.S;
}

void main()
{
    payload.hit = true;
    payload.directLight = vec3(0.0);

    // Fetch TLAS instance info
    uint instanceID = gl_InstanceCustomIndexEXT;
    aabbObject obj  = aabbObjs.aabbObj[instanceID]; 

    uint matIdx = obj.matIndex;
    if (instanceID < 6) 
    {
        matIdx = instanceID; 
    }

    material mat = materials.m[matIdx];    
    quad q       = quads.q[obj.geoIndex];

    vec3 hitPos = gl_WorldRayOriginEXT + gl_HitTEXT * gl_WorldRayDirectionEXT;

    vec3 geoNormal = normalize(cross(q.edgeU, q.edgeV));
    vec3 N = (dot(geoNormal, gl_WorldRayDirectionEXT) < -FLIP_BIAS) ? geoNormal : -geoNormal;

    // ==================================================
    // G-BUFFER CAPTURE (BOUNCE 0 ONLY)
    // ==================================================
    if (payload.bounce == 0)
    {
        payload.primaryNormal = N;
        payload.primaryAlbedo = (mat.matType == diffuseLight) ? vec3(1.0) : max(mat.albedo.rgb, vec3(0.01));
        payload.hitDistance   = gl_HitTEXT; 
    }

    // 1. Emissive Light Source Handling
    if (mat.matType == diffuseLight)
    {
        payload.isEmissive = true;
        payload.hitColor = (payload.bounce == 0) ? (mat.emission.rgb * mat.emission.a) : vec3(0.0);
        return;
    }

    // ==================================================
    // 2. SEED PSEUDO-RANDOM GENERATOR
    // ==================================================
    uint pixelID = gl_LaunchIDEXT.x + gl_LaunchIDEXT.y * gl_LaunchSizeEXT.x;
    uint seed = initPRNG(pixelID, pc.frameIndex + payload.bounce * 17);

    // 3. Dielectric (Glass) Handling
    if (mat.matType == dielectric)
    {
        float refractionRatio = (dot(gl_WorldRayDirectionEXT, N) < 0.0) ? (1.0 / mat.refractionIndex) : mat.refractionIndex;
        vec3 unitDir = normalize(gl_WorldRayDirectionEXT);
        
        float cosTheta = min(dot(-unitDir, N), 1.0);
        float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));

        bool cannotRefract = refractionRatio * sinTheta > 1.0;
        vec3 direction;

        float r0 = (1.0 - refractionRatio) / (1.0 + refractionRatio);
        r0 = r0 * r0;
        float reflectance = r0 + (1.0 - r0) * pow(1.0 - cosTheta, 5.0);

        if (cannotRefract || reflectance > randomFloat(seed))
        {
            direction = reflect(unitDir, N);
        }
        else
        {
            direction = refract(unitDir, N, refractionRatio);
        }

        payload.hitColor   = vec3(1.0);
        payload.rayOrigin  = hitPos + 0.001 * ((dot(direction, N) < 0.0) ? -N : N);
        payload.rayDir     = direction;
        payload.isEmissive = false;
        return;
    }

    // ==================================================
    // 4. NEXT EVENT ESTIMATION (solid-angle sampled, stratified)
    // ==================================================
    quad lightQuad; material lightMat;
    if (findLight(lightQuad, lightMat))
    {
        vec3 directLightAccum = vec3(0.0);

        for (int gx = 0; gx < NEE_GRID; ++gx)
        {
            for (int gy = 0; gy < NEE_GRID; ++gy)
            {
                vec2 jitter = vec2(randomFloat(seed), randomFloat(seed));
                vec2 stratifiedU = (vec2(float(gx), float(gy)) + jitter) / float(NEE_GRID);
                directLightAccum += sampleLightSolidAngle(hitPos, N, mat.albedo.rgb, lightQuad, lightMat, stratifiedU);
            }
        }

        payload.directLight = directLightAccum / float(NEE_GRID * NEE_GRID);
    }

    // ==================================================
    // 5. Lambertian / PBR Scattering (indirect bounce)
    // ==================================================
    vec3 V = normalize(-gl_WorldRayDirectionEXT);

    PBRMaterial pbrMat;
    pbrMat.albedo = mat.albedo.rgb;
   
    if (mat.matType == lambertian)
    {
        pbrMat.roughness = 1.0;
        pbrMat.metallic = 0.0;
    }
    else
    {
        pbrMat.roughness = max(mat.fuzz, 0.05);
        pbrMat.metallic = 1.0;
    }

    vec3 scatterDir;
    float pdf;
    
    vec3 brdf = EvaluatePBR(pbrMat, N, V, seed, scatterDir, pdf);

    if (pdf < 1e-5 || dot(scatterDir, N) <= 0.0)
    {
        scatterDir = normalize(N + randomInUnitSphere(seed));
        pdf = max(dot(N, scatterDir) / 3.14159265, 1e-4);
        brdf = mat.albedo.rgb / 3.14159265;
    }

    payload.hitColor   = (brdf * max(dot(N, scatterDir), 0.0)) / pdf;
    payload.rayOrigin  = hitPos + 0.0001 * N;
    payload.rayDir     = scatterDir;
    payload.isEmissive = false;
}