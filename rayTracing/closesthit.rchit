#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : require

#include "random.glsl"
#include "pbr.glsl"


struct vertex
{
    vec3 pos;
    float _pad0;

    vec3 colour;
    float _pad1;

    vec2 texCoord;
    vec2 _pad2;
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
    vec3 padding3;
};


layout(set = 0, binding = 0)
uniform accelerationStructureEXT topLevelAS;


layout(std430, set = 0, binding = 4)
readonly buffer VertexBuffer
{
    vertex vertices[];
};


layout(std430, set = 0, binding = 5)
readonly buffer IndexBuffer
{
    uint indices[];
};


layout(std430, set = 0, binding = 7)
buffer MaterialBuffer
{
    material m[];
}
materials;



struct rayPayload
{
    vec3 hitColor;

    vec3 rayOrigin;
    vec3 rayDir;

    bool hit;
    bool isEmissive;
};


layout(location = 0)
rayPayloadInEXT rayPayload payload;



layout(push_constant) uniform PushConstants
{
    uint frameIndex;
}
pc;



float schlick(float cosine, float refIdx)
{
    float r0 =
        (1.0 - refIdx) /
        (1.0 + refIdx);

    r0 *= r0;

    return r0 +
        (1.0 - r0) *
        pow(1.0 - cosine, 5.0);
}



void main()
{
    payload.hit = true;


    // --------------------------------------------------
    // Triangle fetch
    // --------------------------------------------------

    uint tri = gl_PrimitiveID;


    vertex v0 =
        vertices[indices[tri * 3 + 0]];

    vertex v1 =
        vertices[indices[tri * 3 + 1]];

    vertex v2 =
        vertices[indices[tri * 3 + 2]];



    vec3 hitPos =
        gl_WorldRayOriginEXT +
        gl_HitTEXT *
        gl_WorldRayDirectionEXT;



    vec3 normal =
        normalize(
            cross(
                v1.pos - v0.pos,
                v2.pos - v0.pos
            )
        );


    // Make normal face ray
    if(dot(normal, gl_WorldRayDirectionEXT) > 0.0)
    {
        normal = -normal;
    }



    // --------------------------------------------------
    // Material
    // --------------------------------------------------

    material mat =
        materials.m[tri];



    // --------------------------------------------------
    // Emission
    // --------------------------------------------------

    if(mat.matType == diffuseLight)
    {
        payload.hitColor =
            mat.emission.rgb *
            max(mat.emission.a, 1.0);


        payload.isEmissive = true;
        return;
    }



    // --------------------------------------------------
    // RNG
    // --------------------------------------------------

    uint seed =
        randomSeed(
            gl_LaunchIDEXT.x +
            pc.frameIndex * 73856093,

            gl_LaunchIDEXT.y +
            pc.frameIndex * 19349663
        );



    // --------------------------------------------------
    // Dielectric
    // --------------------------------------------------

    if(mat.matType == dielectric)
    {
        vec3 rayDir =
            normalize(gl_WorldRayDirectionEXT);


        float refractionRatio =
            (dot(rayDir, normal) < 0.0)
            ? 1.0 / mat.refractionIndex
            : mat.refractionIndex;


        float cosTheta =
            min(dot(-rayDir, normal), 1.0);


        float sinTheta =
            sqrt(
                max(
                    0.0,
                    1.0 -
                    cosTheta * cosTheta
                )
            );


        bool cannotRefract =
            refractionRatio * sinTheta > 1.0;


        float reflectProb =
            schlick(
                cosTheta,
                mat.refractionIndex
            );


        vec3 direction;


        if(cannotRefract ||
           randomFloat(seed) < reflectProb)
        {
            direction =
                reflect(rayDir, normal);
        }
        else
        {
            direction =
                refract(
                    rayDir,
                    normal,
                    refractionRatio
                );
        }


        payload.hitColor = vec3(1.0);

        payload.rayOrigin =
            hitPos + normal * 0.001;

        payload.rayDir =
            direction;

        payload.isEmissive = false;

        return;
    }



    // --------------------------------------------------
    // PBR scattering
    // --------------------------------------------------

    vec3 V =
        normalize(-gl_WorldRayDirectionEXT);



    PBRMaterial pbrMat;

    pbrMat.albedo =
        mat.albedo.rgb;


    if(mat.matType == lambertian)
    {
        pbrMat.roughness = 1.0;
        pbrMat.metallic = 0.0;
    }
    else
    {
        pbrMat.roughness =
            clamp(
                mat.fuzz,
                0.05,
                1.0
            );

        pbrMat.metallic = 1.0;
    }



    vec3 scatterDir;
    float pdf;


    vec3 brdf =
        EvaluatePBR(
            pbrMat,
            normal,
            V,
            seed,
            scatterDir,
            pdf
        );


    if(pdf < 1e-5 ||
       dot(scatterDir, normal) <= 0.0)
    {
        payload.hitColor = vec3(0.0);
        payload.isEmissive = true;
        return;
    }



    payload.hitColor =
        (brdf *
        max(dot(normal, scatterDir), 0.0))
        / pdf;


    payload.rayOrigin =
        hitPos + normal * 0.001;


    payload.rayDir =
        scatterDir;


    payload.isEmissive = false;
}