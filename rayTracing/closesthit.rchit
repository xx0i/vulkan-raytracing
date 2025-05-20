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
layout(set = 0, binding = 3) uniform sampler2D textureSampler;
layout(set = 0, binding = 4) readonly buffer VertexBuffer {vertex vertices[];};
layout(set = 0, binding = 5) readonly buffer IndexBuffer {uint indices[];};

layout(location = 0) rayPayloadInEXT vec3 payload;

hitAttributeEXT vec2 attribs;

vec3 toSRGB(vec3 linearColor)
{
    return pow(linearColor, vec3(1.0 / 2.2));
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

    payload = toSRGB(texColour);
}