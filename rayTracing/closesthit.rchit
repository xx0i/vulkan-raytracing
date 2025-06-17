#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : require

#include "random.glsl"

struct vertex 
{
    vec3 pos;
    float _pad0;
    vec3 colour;
    float _pad1;
    vec2 texCoord;
    vec2 _pad2;
};

struct sphere 
{
    vec3 center;
    float radius;
    vec4 colour;
    uint normalColouring;
    uint padding[3];
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
};

layout(set = 0, binding = 0) uniform accelerationStructureEXT topLevelAS;
layout(set = 0, binding = 1, rgba8) readonly uniform image2D outputImage;
layout(set = 0, binding = 3) uniform sampler2D textureSampler;
layout(set = 0, binding = 4) readonly buffer VertexBuffer {vertex vertices[];};
layout(set = 0, binding = 5) readonly buffer IndexBuffer {uint indices[];};
layout(set = 0, binding = 6) buffer sphereBuffer {sphere s[];}spheres;
layout(set = 0, binding = 7) buffer materialBuffer {material m[];}materials;

struct rayPayload 
{
    vec3 colour;
    vec3 rayDir;
    int depth;
};

layout(location = 0) rayPayloadInEXT rayPayload payload;
layout(location = 1) rayPayloadEXT rayPayload newPayload;

hitAttributeEXT vec2 attribs;

layout(push_constant) uniform PushConstants 
{
    uint frameIndex;
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

void main()
{

    uint index0 = indices[gl_PrimitiveID * 3 + 0];
    uint index1 = indices[gl_PrimitiveID * 3 + 1];
    uint index2 = indices[gl_PrimitiveID * 3 + 2];

    vec3 colour0 = vertices[index0].colour;
    vec3 colour1 = vertices[index1].colour;
    vec3 colour2 = vertices[index2].colour;

    vec2 texCoord0 = vertices[index0].texCoord;
    vec2 texCoord1 = vertices[index1].texCoord;
    vec2 texCoord2 = vertices[index2].texCoord;

    float u = attribs.x;
    float v = attribs.y;
    float w = 1.0 - u - v;

    vec2 texCoord = w * texCoord0 + u * texCoord1 + v * texCoord2;

    vec3 texColour = texture(textureSampler, texCoord).rgb;
    vec3 vertexColour = w * colour0 + u * colour1 + v * colour2;

    sphere sph = spheres.s[gl_PrimitiveID];
    material mat = materials.m[gl_PrimitiveID];    

    vec3 hitPos = gl_WorldRayOriginEXT + gl_HitTEXT * gl_WorldRayDirectionEXT;
    vec3 normal = normalize(hitPos - sph.center);

    if (sph.normalColouring == 1)
    {	
	vec3 normalColour = 0.5 * (normal + vec3(1.0));
	payload.colour = normalColour;
 	return;
    }
    
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

	payload.colour = mat.albedo.rgb * newPayload.colour;
        return;
    }

    else
    {
	//payload.colour = toSRGB(texColour);
	//payload.colour = toSRGB(vertexColour);
	payload.colour = sph.colour.rgb;
	return;
    }
}