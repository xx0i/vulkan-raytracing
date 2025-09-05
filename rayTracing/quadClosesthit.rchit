#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_debug_printf : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

#include "random.glsl"
#include "perlinNoise.glsl"

struct quad
{
    vec3 origin;
    float pad0; 
    vec3 edgeU;
    float pad1; 
    vec3 edgeV;
    float pad2; 
};

const uint lambertian = 0;
const uint metal = 1;
const uint dielectric = 2;
const uint isotropic = 3;
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
layout(set = 0, binding = 1, rgba8) readonly uniform image2D outputImage;
layout(set = 0, binding = 3) uniform sampler2D textureSampler;
layout(set = 0, binding = 7) buffer materialBuffer {material m[];}materials;
layout(set = 0, binding = 8) buffer quadBuffer {quad q[];}quads;
layout(std430, set = 0, binding = 10) buffer aabbObjectsBuffer {aabbObject aabbObj[];}aabbObjs;

struct rayPayload 
{
    vec3 colour;
    vec3 rayDir;
    int depth;
};

layout(location = 0) rayPayloadInEXT rayPayload payload;
layout(location = 1) rayPayloadEXT rayPayload newPayload;

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

vec3 toSRGB(vec3 linearColor)
{
    return pow(linearColor, vec3(1.0 / 2.2));
}

vec3 cosineSampleHemisphere(float u1, float u2)
{
    float r = sqrt(u1);
    float theta = 2.0 * 3.1415926 * u2;

    float x = r * cos(theta);
    float y = r * sin(theta);
    float z = sqrt(1.0 - u1); // Bias toward normal direction

    return vec3(x, y, z); // This is in tangent space
}

mat3 buildOrthonormalBasis(vec3 n)
{
    vec3 t = abs(n.z) < 0.999 ? normalize(cross(n, vec3(0.0, 0.0, 1.0))) : normalize(cross(n, vec3(1.0, 0.0, 0.0)));
    vec3 b = cross(n, t);
    return mat3(t, b, n); // Matrix transforms from tangent to world space
}

float schlick(float cosine, float refractIndex)
{
    float r0 = (1.0 - refractIndex) / (1.0 + refractIndex);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * pow(1.0 - cosine, 5.0);
}

void main()
{
    aabbObject obj = aabbObjs.aabbObj[gl_InstanceCustomIndexEXT];
    material mat = materials.m[obj.matIndex];    

    vec3 hitPos = gl_WorldRayOriginEXT + gl_HitTEXT * gl_WorldRayDirectionEXT;
    vec3 normal;

    quad q = quads.q[obj.geoIndex];
    normal = normalize(cross(q.edgeU.xyz, q.edgeV.xyz));

    if(mat.matType == lambertian)
    {
	const int MAX_DEPTH = 5;
	if (payload.depth >= MAX_DEPTH)
	{
    	    payload.colour = vec3(0.0);
    	    return;
	}
	
	uint seed = randomSeed(gl_LaunchIDEXT.x + pc.frameIndex * 73856093, gl_LaunchIDEXT.y + pc.frameIndex * 19349663);
	//vec3 scatterDirection = normalize(normal + randomInUnitSphere(seed));
	float u1 = randomFloat(seed);
	float u2 = randomFloat(seed);
	vec3 dir1 = cosineSampleHemisphere(u1, u2);
	vec3 dir2 = randomInUnitSphere(seed);
	vec3 scatterDirection = normalize(mix(dir1, dir2, 0.5));
	
	if(length(scatterDirection) < 1e-3)
	{
	    scatterDirection = normal;
	}

	newPayload.rayDir = scatterDirection;
	newPayload.depth = payload.depth + 1;
	newPayload.colour = vec3(0.0);

	traceRayEXT(topLevelAS, gl_RayFlagsOpaqueEXT, 0xFF, 0, 0, 0, hitPos + 0.001 * normal, 0.001, scatterDirection, 10000.0, 1);

vec3 attenuation = mat.albedo.rgb;

// If the material is fully black, don’t kill the path
if (length(attenuation) < 0.001) {
    payload.colour = newPayload.colour;  // pass light through
} else {
    payload.colour = attenuation * newPayload.colour;
}
        return;
    }
    else if(mat.matType == metal)
    {
	const int MAX_DEPTH = 5;
	if (payload.depth >= MAX_DEPTH)
	{
    	    payload.colour = vec3(0.0);
    	    return;
	}

	vec3 reflected = reflect(normalize(payload.rayDir), normal);
	uint seed = randomSeed(gl_LaunchIDEXT.x + pc.frameIndex * 73856093, gl_LaunchIDEXT.y + pc.frameIndex * 19349663);
	vec3 fuzzOffset = mat.fuzz * randomInUnitSphere(seed);
	vec3 scatterDirection = normalize(reflected + fuzzOffset);
	
	if(dot(scatterDirection, normal) > 0.0)
	{
	    newPayload.rayDir = scatterDirection;
	    newPayload.depth = payload.depth + 1;
	    newPayload.colour = vec3(0.0);
	    
	    traceRayEXT(topLevelAS, gl_RayFlagsOpaqueEXT, 0xFF, 0, 0, 0, hitPos + 0.001 * normal, 0.001, scatterDirection, 10000.0, 1);
	    
	    payload.colour = mat.albedo.rgb * newPayload.colour;

	}
	else
	{
	    payload.colour = vec3(0.0);
	}
	return;
    }
  else if (mat.matType == dielectric)
  {
    const int MAX_DEPTH = 5;
    if (payload.depth >= MAX_DEPTH)
    {
        payload.colour = vec3(0.0);
        return;
    }

    vec3 unitDir = normalize(payload.rayDir);
    float refIdx = mat.refractionIndex;
    vec3 outwardNormal;
    float niOverNt;
    float cosine;

    if (dot(unitDir, normal) < 0.0) {
        // Ray is outside the surface
        outwardNormal = normal;
        niOverNt = 1.0 / refIdx;
        cosine = -dot(unitDir, normal);
    } else {
        // Ray is inside the surface
        outwardNormal = -normal;
        niOverNt = refIdx;
        cosine = dot(unitDir, normal);
    }

    vec3 refracted = refract(unitDir, outwardNormal, niOverNt);

    // Use Schlick approximation for reflect probability
    float reflectProb = (length(refracted) > 0.0) ? schlick(cosine, refIdx) : 1.0;

    uint seed = randomSeed(gl_LaunchIDEXT.x + pc.frameIndex * 73856093, gl_LaunchIDEXT.y + pc.frameIndex * 19349663);
    float randVal = randomFloat(seed);

    vec3 scatterDir;
    if (randVal < reflectProb)
        scatterDir = reflect(unitDir, normal);
    else
        scatterDir = refracted;

    newPayload.rayDir = scatterDir;
    newPayload.depth = payload.depth + 1;
    newPayload.colour = vec3(0.0);

    traceRayEXT(
        topLevelAS, 
        gl_RayFlagsOpaqueEXT, 0xFF, 0, 0, 0,
        hitPos + 0.001 * scatterDir, 0.001, scatterDir, 10000.0,
        1
    );

    // For dielectrics, use the traced color without tinting
    payload.colour = newPayload.colour;
    return;
   }
   if (mat.matType == diffuseLight)
   {
    	if (dot(normal, -payload.rayDir) > 0.0)
	{
       	     payload.colour = mat.albedo.rgb * mat.emission.a; 
	}    
	else
        {
	    payload.colour = vec3(0.0);

	}
        return;
    }
}