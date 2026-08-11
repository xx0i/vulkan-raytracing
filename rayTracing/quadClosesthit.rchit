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
    vec3 rayOrigin;
    vec3 rayDir;
    bool hit;
    bool isEmissive;
    vec3 primaryNormal;
    vec3 primaryAlbedo;
    float hitDistance;
};

layout(location = 0) rayPayloadInEXT rayPayload payload;

layout(push_constant) uniform PushConstants 
{
    uint frameIndex;
    uint missColour;
} pc;

void main()
{
    payload.hit = true;

    // Fetch TLAS instance info
    uint instanceID = gl_InstanceCustomIndexEXT;
    aabbObject obj  = aabbObjs.aabbObj[instanceID]; 

    // Directly map the first 6 room quads to their base materials
    uint matIdx = obj.matIndex;
    if (instanceID < 6) 
    {
        matIdx = instanceID; 
    }

    material mat = materials.m[matIdx];    
    quad q       = quads.q[obj.geoIndex];

    vec3 hitPos = gl_WorldRayOriginEXT + gl_HitTEXT * gl_WorldRayDirectionEXT;

    // Geometric normal calculation
    vec3 geoNormal = normalize(cross(q.edgeU, q.edgeV));
    
    // Ensure normal points OPPOSITE to incoming ray
    vec3 N = (dot(geoNormal, gl_WorldRayDirectionEXT) < 0.0) ? geoNormal : -geoNormal;

    // ==================================================
    // G-BUFFER CAPTURE
    // ==================================================
    payload.primaryNormal = N;
    payload.primaryAlbedo = mat.albedo.rgb;
    payload.hitDistance = gl_RayTmaxEXT; 

    // 1. Emissive Light Source Handling
    if (mat.matType == diffuseLight)
    {
        payload.hitColor = mat.emission.rgb * mat.emission.a;
        payload.isEmissive = true;
        return;
    }

    // ==================================================
    // 2. SEED PSEUDO-RANDOM GENERATOR
    // ==================================================
    // Stable pixel ID + frame count (No floating-point distance jitter)
    uint pixelID = gl_LaunchIDEXT.x + gl_LaunchIDEXT.y * gl_LaunchSizeEXT.x;
    uint seed = initPRNG(pixelID, pc.frameIndex);

    // 3. Dielectric (Glass) Handling
    if (mat.matType == dielectric)
    {
        float refractionRatio = (dot(gl_WorldRayDirectionEXT, N) < 0.0) ? (1.0 / mat.refractionIndex) : mat.refractionIndex;
        vec3 unitDir = normalize(gl_WorldRayDirectionEXT);
        
        float cosTheta = min(dot(-unitDir, N), 1.0);
        float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));

        bool cannotRefract = refractionRatio * sinTheta > 1.0;
        vec3 direction;

        // Schlick's approximation for reflectance
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

        payload.hitColor   = vec3(1.0); // Pure glass lets all light through
        payload.rayOrigin  = hitPos + 0.001 * ((dot(direction, N) < 0.0) ? -N : N);
        payload.rayDir     = direction;
        payload.isEmissive = false;
        return;
    }

    // 4. Lambertian / PBR Scattering
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
    
    // Evaluate PBR BRDF & next ray direction (mutates 'seed' inout)
    vec3 brdf = EvaluatePBR(pbrMat, N, V, seed, scatterDir, pdf);

    // Fallback in case BRDF / PDF sampling fails
    if (pdf < 1e-5 || dot(scatterDir, N) <= 0.0)
    {
        scatterDir = normalize(N + randomInUnitSphere(seed));
        pdf = max(dot(N, scatterDir) / 3.14159265, 1e-4);
        brdf = mat.albedo.rgb / 3.14159265;
    }

    payload.hitColor   = (brdf * max(dot(N, scatterDir), 0.0)) / pdf;
    payload.rayOrigin  = hitPos + 0.001 * N; // Bias origin along normal
    payload.rayDir     = scatterDir;
    payload.isEmissive = false;
}