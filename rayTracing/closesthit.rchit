#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable

struct vertex 
{
    vec3 pos;
float _pad0;
    vec3 colour;
float _pad1;
    vec2 texCoord;
vec2 _pad2;
};

layout(set = 0, binding = 0) uniform accelerationStructureEXT topLevelAS;
layout(set = 0, binding = 1, rgba8) readonly uniform image2D outputImage;
layout(set = 0, binding = 3) readonly buffer VertexBuffer {vertex vertices[];};
layout(set = 0, binding = 4) readonly buffer IndexBuffer {uint indices[];};

layout(location = 0) rayPayloadInEXT vec3 payload;

hitAttributeEXT vec2 attribs;

void main()
{
	uint index0 = indices[gl_PrimitiveID * 3 + 0];
	uint index1 = indices[gl_PrimitiveID * 3 + 1];
	uint index2 = indices[gl_PrimitiveID * 3 + 2];

	vec3 color0 = vertices[index0].colour;
	vec3 color1 = vertices[index1].colour;
	vec3 color2 = vertices[index2].colour;

	float u = attribs.x;
	float v = attribs.y;
	float w = 1.0 - u - v;

	payload = w * color0 + u * color1 + v * color2;
}