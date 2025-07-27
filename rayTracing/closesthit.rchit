#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_GOOGLE_include_directive : require

#include "random.glsl"
#include "perlinNoise.glsl"

struct vertex 
{
    vec3 pos;
    float _pad0;
    vec3 colour;
    float _pad1;
    vec2 texCoord;
    vec2 _pad2;
};

const uint sphereShape = 0;
const uint quadShape = 1;

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
};

layout(set = 0, binding = 0) uniform accelerationStructureEXT topLevelAS;
layout(set = 0, binding = 1, rgba8) readonly uniform image2D outputImage;
layout(set = 0, binding = 3) uniform sampler2D textureSampler;
layout(set = 0, binding = 4) readonly buffer VertexBuffer {vertex vertices[];};
layout(set = 0, binding = 5) readonly buffer IndexBuffer {uint indices[];};
layout(set = 0, binding = 6) buffer sphereBuffer {sphere s[];}spheres;
layout(set = 0, binding = 7) buffer materialBuffer {material m[];}materials;
layout(set = 0, binding = 8) buffer quadBuffer {quad q[];}quads;
layout(set = 0, binding = 9) buffer geoTypeBuffer {uint gt[];}geoTypes;

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

    uint index0 = indices[gl_PrimitiveID * 3 + 0];
    uint index1 = indices[gl_PrimitiveID * 3 + 1];
    uint index2 = indices[gl_PrimitiveID * 3 + 2];

    vec3 colour0 = vertices[index0].colour;
    vec3 colour1 = vertices[index1].colour;
    vec3 colour2 = vertices[index2].colour;

    vec2 texCoord0 = vertices[index0].texCoord;
    vec2 texCoord1 = vertices[index1].texCoord;
    vec2 texCoord2 = vertices[index2].texCoord;

    float u = attribs.uv.x;
    float v = attribs.uv.y;
    float w = 1.0 - u - v;

    vec2 texCoord = w * texCoord0 + u * texCoord1 + v * texCoord2;

    vec3 texColour = texture(textureSampler, texCoord).rgb;
    vec3 vertexColour = w * colour0 + u * colour1 + v * colour2;

    sphere sph = spheres.s[gl_PrimitiveID];
    uint geoType = geoTypes.gt[gl_PrimitiveID];
    material mat = materials.m[gl_PrimitiveID];    

    vec3 hitPos = gl_WorldRayOriginEXT + gl_HitTEXT * gl_WorldRayDirectionEXT;
    vec3 normal;

    if(geoType == sphereShape)
    {
	sphere sph = spheres.s[gl_PrimitiveID];
	normal = normalize(hitPos - sph.center);

	if (sph.normalColouring == 1)
	{	
	    vec3 normalColour = 0.5 * (normal + vec3(1.0));
	    payload.colour = normalColour;
 	    return;
    	}

	if (sph.textured == 1)
	{
            vec2 uv = attribs.uv;
	    vec3 texColour = texture(textureSampler, uv).rgb;
            payload.colour = texColour;
            return;
	}
    
	if(sph.checkered == 1)
	{
            float scale = 5.0; // Adjust as needed for checker size
            float sines = sin(scale * hitPos.x) * sin(scale * hitPos.y) * sin(scale * hitPos.z);

            vec3 colourA = vec3(0.0, 0.0, 0.0); // Dark squares
            vec3 colourB = vec3(1.0, 1.0, 1.0); // Light squares

            if (sines < 0.0)
		payload.colour = colourA;
            else
           	payload.colour = colourB;

            return;
	} 
    
	if(sph.perlinNoise == 1)
	{
	float scale = 5.0;
	    float frequency = 3.0;
	    float turbulenceAmplitude = 5.0;

	    float marble = marbleTexture(scale * hitPos, frequency, turbulenceAmplitude);

	    marble = marble * 0.5 + 0.5;

	    vec3 colorA = vec3(1.0, 1.0, 1.0);
	    vec3 colorB = vec3(0.0, 0.0, 0.0);

	    vec3 finalColor = mix(colorA, colorB, marble);

	    payload.colour = finalColor;
	    return;
	}
    }
    if(geoType == quadShape)
    {
        quad q = quads.q[gl_PrimitiveID];
	normal = normalize(cross(q.edgeU.xyz, q.edgeV.xyz));
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
    else
    {
	//payload.colour = toSRGB(texColour);
	//payload.colour = toSRGB(vertexColour);
	payload.colour = sph.colour.rgb;
	return;
    }
}