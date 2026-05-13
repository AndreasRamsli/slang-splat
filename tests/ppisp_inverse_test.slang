import "utility/ppisp_tonemap.slang";

uniform StructuredBuffer<float4> g_InputColors;
uniform StructuredBuffer<float2> g_SensorCoords;
uniform RWStructuredBuffer<float4> g_OutputForward;
uniform RWStructuredBuffer<float4> g_OutputRecovered;
uniform uint g_Count;
uniform PPISPTonemap g_PPISP;

[shader("compute")]
[numthreads(64, 1, 1)]
void csRoundTripPPISP(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint index = dispatchThreadID.x;
    if (index >= g_Count) return;

    float3 radiance = g_InputColors[index].xyz;
    float2 sensorCoord = g_SensorCoords[index];
    float3 forward = ppisp_apply_tonemap(radiance, sensorCoord, g_PPISP);
    float3 recovered = ppisp_inverse_tonemap(forward, sensorCoord, g_PPISP);
    g_OutputForward[index] = float4(forward, 1.0);
    g_OutputRecovered[index] = float4(recovered, 1.0);
}