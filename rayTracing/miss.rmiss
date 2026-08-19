#version 460
#extension GL_EXT_ray_tracing : require
struct rayPayload 
{
    vec3 hitColor;
    vec3 directLight;    // NEW — must match closest-hit/raygen struct exactly
    vec3 rayOrigin;
    vec3 rayDir;
    bool hit;
    bool isEmissive;
    vec3 primaryNormal;
    vec3 primaryAlbedo;
    float hitDistance;
    uint bounce;          // NEW — see note below
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
    payload.directLight = vec3(0.0);   // NEW — avoid leaving it undefined on a miss
    payload.hitDistance = 10000.0;
    if (pc.missColour == 1)
    {
        float t = 0.5 * (normalize(gl_WorldRayDirectionEXT).y + 1.0); 
        vec3 white = vec3(1.0);
        vec3 blue  = vec3(0.5, 0.7, 1.0); 
        
        payload.hitColor = mix(white, blue, t);
    }
    else
    {
        payload.hitColor = vec3(0.0);
    }
}