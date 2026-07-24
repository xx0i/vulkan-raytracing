#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : require

#include "random.glsl"
#include "perlinNoise.glsl"
#include "pbr.glsl"

struct sphere 
{
    vec3 center;
    float radius;
    vec4 colour;
    uint normalColouring;
    uint textured;
    uint checkered;
    uint perlinNoise;
};

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
    float padding2;
};

struct aabbObject
{
    uint type;
    uint geoIndex;
    uint matIndex;
    uint _pad0;
};

layout(set = 0, binding = 0) uniform accelerationStructureEXT topLevelAS;
layout(set = 0, binding = 3) uniform sampler2D textureSampler;
layout(set = 0, binding = 6) buffer sphereBuffer { sphere s[]; } spheres;
layout(set = 0, binding = 7) buffer materialBuffer { material m[]; } materials;
layout(std430, set = 0, binding = 10) buffer aabbObjectsBuffer { aabbObject aabbObj[]; } aabbObjs;

struct rayPayload 
{
    vec3 hitColor;
    vec3 rayOrigin;
    vec3 rayDir;
    bool hit;
    bool isEmissive;
};

layout(location = 0) rayPayloadInEXT rayPayload payload;

struct attributes
{
    vec2 uv;
};

hitAttributeEXT attributes attribs;

layout(push_constant) uniform PushConstants 
{
    uint frameIndex;
    uint missColour;
} pc;

float schlick(float cosine, float refractIndex)
{
    float r0 = (1.0 - refractIndex) / (1.0 + refractIndex);
    r0 = r0 * r0;

    return r0 + (1.0 - r0) * pow(1.0 - cosine, 5.0);
}

void main()
{
    payload.hit = true;

    aabbObject obj = aabbObjs.aabbObj[gl_InstanceCustomIndexEXT];
    material mat   = materials.m[obj.matIndex];
    sphere sph     = spheres.s[obj.geoIndex];

    vec3 hitPos = gl_WorldRayOriginEXT + gl_HitTEXT * gl_WorldRayDirectionEXT;
    vec3 normal = normalize(hitPos - sph.center);

    // 1. Emissive Material
    if (mat.matType == diffuseLight)
    {
        float strength = mat.emission.a > 0.0 ? mat.emission.a : 15.0;

        payload.hitColor   = mat.albedo.rgb * strength;
        payload.isEmissive = true;
        return;
    }

    // 2. Procedural Sphere Overrides
    if (sph.normalColouring == 1)
    {
        payload.hitColor   = 0.5 * (normal + vec3(1.0));
        payload.isEmissive = true;
        return;
    }

    if (sph.textured == 1)
    {
        mat.albedo.rgb = texture(textureSampler, attribs.uv).rgb;
    }

    if (sph.checkered == 1)
    {
        float scale = 5.0;
        float sines = sin(scale * hitPos.x) *
                      sin(scale * hitPos.y) *
                      sin(scale * hitPos.z);

        payload.hitColor   = (sines < 0.0) ? vec3(0.0) : vec3(1.0);
        payload.isEmissive = true;
        return;
    }

    if (sph.perlinNoise == 1)
    {
        float marble = marbleTexture(5.0 * hitPos, 3.0, 5.0);
        marble = marble * 0.5 + 0.5;

        mat.albedo.rgb = mix(vec3(1.0), vec3(0.0), marble);
    }

    // Face normal towards ray
    normal = (dot(normal, gl_WorldRayDirectionEXT) > 0.0) ? -normal : normal;


    uint seed = randomSeed(
        gl_LaunchIDEXT.x + pc.frameIndex * 73856093,
        gl_LaunchIDEXT.y + pc.frameIndex * 19349663
    );

    vec3 V = normalize(-gl_WorldRayDirectionEXT);


    // 3. Dielectric Materials
    if (mat.matType == dielectric)
    {
        vec3 unitDir = normalize(gl_WorldRayDirectionEXT);

        float refractionRatio =
            (dot(unitDir, normal) < 0.0) ?
            (1.0 / mat.refractionIndex) :
            mat.refractionIndex;


        float cosTheta = min(dot(-unitDir, normal), 1.0);

        float sinTheta =
            sqrt(max(0.0, 1.0 - cosTheta * cosTheta));


        bool cannotRefract =
            refractionRatio * sinTheta > 1.0;


        float reflectProb =
            schlick(cosTheta, mat.refractionIndex);


        vec3 scatterDir;

        if (cannotRefract || randomFloat(seed) < reflectProb)
        {
            scatterDir = reflect(unitDir, normal);
        }
        else
        {
            scatterDir = refract(unitDir, normal, refractionRatio);
        }


        payload.hitColor   = vec3(1.0);
        payload.rayOrigin  = hitPos + 0.001 * normal;
        payload.rayDir     = scatterDir;
        payload.isEmissive = false;

        return;
    }


    // 4. Lambertian & Metallic Materials
    if (mat.matType == lambertian || mat.matType == metal)
    {
        PBRMaterial pbrMat;

        pbrMat.albedo = mat.albedo.rgb;

        if (mat.matType == lambertian)
        {
            pbrMat.roughness = 1.0;
            pbrMat.metallic = 0.0;
        }
        else
        {
            pbrMat.roughness = clamp(mat.fuzz, 0.05, 1.0);
            pbrMat.metallic = 1.0;
        }


        vec3 scatterDir;
        float pdf;

        vec3 brdf = EvaluatePBR(
            pbrMat,
            normal,
            V,
            seed,
            scatterDir,
            pdf
        );


        if (pdf <= 0.0001)
        {
            payload.hitColor   = vec3(0.0);
            payload.isEmissive = true;
            return;
        }


        payload.hitColor   = (brdf * max(dot(normal, scatterDir), 0.0)) / pdf;
        payload.rayOrigin  = hitPos + 0.001 * normal;
        payload.rayDir     = scatterDir;
        payload.isEmissive = false;

        return;
    }
}