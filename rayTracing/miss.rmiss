#version 460
#extension GL_EXT_ray_tracing : require

struct rayPayload 
{
    vec3 colour;
    vec3 rayDir;
    int depth;
};

layout(push_constant) uniform PushConstants 
{
    uint frameIndex;
    uint missColour;
} pc;


layout(location = 0) rayPayloadInEXT rayPayload payload;

void main()
{
	if(pc.missColour == 1)
	{
	    float t = 0.5 * (payload.rayDir.z + 1.0);
	    vec3 white = vec3(1.0);
	    vec3 blue = vec3(0.5, 0.7, 1.0); 
	    payload.colour = mix(white, blue, t);
	}
	if(pc.missColour == 0)
	{
	    payload.colour = vec3(0.0);
	}
}