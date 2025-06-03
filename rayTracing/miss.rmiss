#version 460
#extension GL_EXT_ray_tracing : require

struct rayPayload 
{
    vec3 colour;
    vec3 rayDir;
};

layout(location = 0) rayPayloadInEXT rayPayload payload;

void main()
{
	float t = 0.5 * (payload.rayDir.z + 1.0);
	vec3 white = vec3(1.0);
	vec3 blue = vec3(0.5, 0.7, 1.0); 
	payload.colour = mix(white, blue, t);

}