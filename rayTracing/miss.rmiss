#version 460
#extension GL_EXT_ray_tracing : require

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
    payload.hit = false;    
    payload.isEmissive = true; 

    // Far plane depth for rays that miss all geometry
    payload.hitDistance = 10000.0;

    if (pc.missColour == 1)
    {
        // Gradient sky (white at horizon, blue looking up)
        // Adjust to .z if your world coordinate system is Z-Up!
        float t = 0.5 * (normalize(gl_WorldRayDirectionEXT).y + 1.0); 
        vec3 white = vec3(1.0);
        vec3 blue  = vec3(0.5, 0.7, 1.0); 
        
        payload.hitColor = mix(white, blue, t);
    }
    else
    {
        // Pure black sky
        payload.hitColor = vec3(0.0);
    }
}