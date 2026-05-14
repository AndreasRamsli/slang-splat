import "utility/math/constants.slang";
import "utility/math/numeric.slang";
import "utility/ppisp_tonemap.slang";
import "utility/random/random.slang";

static const uint PHOTOMETRIC_THREADS = 64u;
static const uint PHOTOMETRIC_TARGET_TILE_DIM = 16u;
static const uint PPISP_PARAM_COUNT = 36u;
static const float SANITIZE_HUGE_VALUE = 1e8;

uniform StructuredBuffer<float> g_Params;
uniform RWStructuredBuffer<float> g_Grads;
uniform RWStructuredBuffer<float> g_LossSum;

uniform RWStructuredBuffer<float4> g_ObservationMeanSamples;
uniform RWStructuredBuffer<float2> g_ObservationSensorCoords;
uniform StructuredBuffer<uint> g_TrackObservationRanges;
uniform StructuredBuffer<uint> g_TrackPairRanges;
uniform StructuredBuffer<uint> g_ObservationFrameIndices;

uniform uint g_FrameCount;
uniform uint g_TrackCount;
uniform uint g_SampledPairCount;
uniform uint g_TotalPairCount;
uniform uint g_SamplingSeed;
uniform uint g_NeighborhoodSize;

uniform Texture2D<float4> g_PairDatasetSourceFrame;
uniform StructuredBuffer<float2> g_ObservationXY;
uniform StructuredBuffer<uint> g_DatasetFrameObservationRanges;
uniform StructuredBuffer<uint> g_DatasetFrameObservationIndices;
uniform uint g_DatasetFrameIndex;
uniform uint g_DatasetFrameWidth;
uniform uint g_DatasetFrameHeight;

uniform Texture2D<float4> g_PhotometricSourceTarget;
uniform RWTexture2D<float4> g_PhotometricTarget;
uniform uint g_PhotometricSourceWidth;
uniform uint g_PhotometricSourceHeight;
uniform uint g_PhotometricTargetWidth;
uniform uint g_PhotometricTargetHeight;
uniform PPISPTonemap g_PhotometricTargetPPISPParams;

float load_param(uint frameIndex, uint paramId)
{
    return g_Params[paramId * g_FrameCount + frameIndex];
}

PPISPTonemap load_ppisp(uint frameIndex)
{
    PPISPTonemap params;
    params.exposureEv = load_param(frameIndex, 0u);
    params.vignetteCenterX = float3(load_param(frameIndex, 1u), load_param(frameIndex, 2u), load_param(frameIndex, 3u));
    params.vignetteCenterY = float3(load_param(frameIndex, 4u), load_param(frameIndex, 5u), load_param(frameIndex, 6u));
    params.vignetteCoeffR2 = float3(load_param(frameIndex, 7u), load_param(frameIndex, 8u), load_param(frameIndex, 9u));
    params.vignetteCoeffR4 = float3(load_param(frameIndex, 10u), load_param(frameIndex, 11u), load_param(frameIndex, 12u));
    params.vignetteCoeffR6 = float3(load_param(frameIndex, 13u), load_param(frameIndex, 14u), load_param(frameIndex, 15u));
    params.chromaOffsetR = float2(load_param(frameIndex, 16u), load_param(frameIndex, 17u));
    params.chromaOffsetG = float2(load_param(frameIndex, 18u), load_param(frameIndex, 19u));
    params.chromaOffsetB = float2(load_param(frameIndex, 20u), load_param(frameIndex, 21u));
    params.chromaOffsetW = float2(load_param(frameIndex, 22u), load_param(frameIndex, 23u));
    params.crfTau = float3(load_param(frameIndex, 24u), load_param(frameIndex, 25u), load_param(frameIndex, 26u));
    params.crfEta = float3(load_param(frameIndex, 27u), load_param(frameIndex, 28u), load_param(frameIndex, 29u));
    params.crfXi = float3(load_param(frameIndex, 30u), load_param(frameIndex, 31u), load_param(frameIndex, 32u));
    params.crfGamma = float3(load_param(frameIndex, 33u), load_param(frameIndex, 34u), load_param(frameIndex, 35u));
    return params;
}

PPISPTonemap.Differential zero_ppisp_differential()
{
    PPISPTonemap.Differential diff;
    diff.exposureEv = 0.0;
    diff.vignetteCenterX = 0.0;
    diff.vignetteCenterY = 0.0;
    diff.vignetteCoeffR2 = 0.0;
    diff.vignetteCoeffR4 = 0.0;
    diff.vignetteCoeffR6 = 0.0;
    diff.chromaOffsetR = 0.0;
    diff.chromaOffsetG = 0.0;
    diff.chromaOffsetB = 0.0;
    diff.chromaOffsetW = 0.0;
    diff.crfTau = 0.0;
    diff.crfEta = 0.0;
    diff.crfXi = 0.0;
    diff.crfGamma = 0.0;
    return diff;
}

void clear_local_grads(inout float grads[PPISP_PARAM_COUNT])
{
    [unroll] for (uint paramId = 0u; paramId < PPISP_PARAM_COUNT; paramId++) grads[paramId] = 0.0;
}

void add_local_ppisp_grad(inout float grads[PPISP_PARAM_COUNT], PPISPTonemap.Differential diff)
{
    grads[0u] += diff.exposureEv;
    grads[1u] += diff.vignetteCenterX.x;
    grads[2u] += diff.vignetteCenterX.y;
    grads[3u] += diff.vignetteCenterX.z;
    grads[4u] += diff.vignetteCenterY.x;
    grads[5u] += diff.vignetteCenterY.y;
    grads[6u] += diff.vignetteCenterY.z;
    grads[7u] += diff.vignetteCoeffR2.x;
    grads[8u] += diff.vignetteCoeffR2.y;
    grads[9u] += diff.vignetteCoeffR2.z;
    grads[10u] += diff.vignetteCoeffR4.x;
    grads[11u] += diff.vignetteCoeffR4.y;
    grads[12u] += diff.vignetteCoeffR4.z;
    grads[13u] += diff.vignetteCoeffR6.x;
    grads[14u] += diff.vignetteCoeffR6.y;
    grads[15u] += diff.vignetteCoeffR6.z;
    grads[16u] += diff.chromaOffsetR.x;
    grads[17u] += diff.chromaOffsetR.y;
    grads[18u] += diff.chromaOffsetG.x;
    grads[19u] += diff.chromaOffsetG.y;
    grads[20u] += diff.chromaOffsetB.x;
    grads[21u] += diff.chromaOffsetB.y;
    grads[22u] += diff.chromaOffsetW.x;
    grads[23u] += diff.chromaOffsetW.y;
    grads[24u] += diff.crfTau.x;
    grads[25u] += diff.crfTau.y;
    grads[26u] += diff.crfTau.z;
    grads[27u] += diff.crfEta.x;
    grads[28u] += diff.crfEta.y;
    grads[29u] += diff.crfEta.z;
    grads[30u] += diff.crfXi.x;
    grads[31u] += diff.crfXi.y;
    grads[32u] += diff.crfXi.z;
    grads[33u] += diff.crfGamma.x;
    grads[34u] += diff.crfGamma.y;
    grads[35u] += diff.crfGamma.z;
}

float4 load_observation_mean_sample(uint observationIndex)
{
    return sanitize_float4(g_ObservationMeanSamples[observationIndex], SANITIZE_HUGE_VALUE);
}

float2 load_observation_sensor_coord(uint observationIndex)
{
    return g_ObservationSensorCoords[observationIndex];
}

uint sampled_pair_index(uint sampleIndex)
{
    if (g_TotalPairCount == 0u) return 0u;
    return hash_u32(g_SamplingSeed ^ ((sampleIndex + 1u) * 0x9e3779b9u)) % g_TotalPairCount;
}

uint find_track_index(uint pairIndex)
{
    uint left = 0u;
    uint right = g_TrackCount;
    while (left + 1u < right)
    {
        uint mid = left + ((right - left) >> 1u);
        if (g_TrackPairRanges[mid] <= pairIndex) {
            left = mid;
        } else {
            right = mid;
        }
    }
    return left;
}

uint2 decode_observation_pair_index(uint pairIndex, uint observationCount)
{
    uint left = 0u;
    uint remaining = observationCount > 0u ? observationCount - 1u : 0u;
    uint rank = pairIndex;
    while (remaining > 0u && rank >= remaining)
    {
        rank -= remaining;
        left += 1u;
        remaining -= 1u;
    }
    return uint2(left, left + 1u + rank);
}

void atomic_add_ppisp_grad(uint frameIndex, float grads[PPISP_PARAM_COUNT])
{
    [unroll] for (uint paramId = 0u; paramId < PPISP_PARAM_COUNT; paramId++)
    {
        if (grads[paramId] != 0.0) InterlockedAdd(g_Grads[paramId * g_FrameCount + frameIndex], grads[paramId]);
    }
}

[shader("compute")]
[numthreads(PHOTOMETRIC_THREADS, 1, 1)]
void csBuildPhotometricObservationDataset(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint start = g_DatasetFrameObservationRanges[g_DatasetFrameIndex];
    uint stop = g_DatasetFrameObservationRanges[g_DatasetFrameIndex + 1u];
    uint entryCount = stop - start;
    uint threadIndex = dispatchThreadID.x;
    if (threadIndex >= entryCount) return;

    uint observationIndex = g_DatasetFrameObservationIndices[start + threadIndex];
    float2 center = g_ObservationXY[observationIndex];
    uint neighborhoodSize = max(g_NeighborhoodSize, 1u);
    uint neighborhoodSampleCount = neighborhoodSize * neighborhoodSize;
    uint radius = neighborhoodSize / 2u;
    int2 basePixel = int2(floor(center));
    uint width = max(g_DatasetFrameWidth, 1u);
    uint height = max(g_DatasetFrameHeight, 1u);
    float4 meanSample = 0.0;
    float2 meanSensorCoord = 0.0;

    for (uint sampleY = 0u; sampleY < neighborhoodSize; sampleY++)
    {
        for (uint sampleX = 0u; sampleX < neighborhoodSize; sampleX++)
        {
            int dx = int(sampleX) - int(radius);
            int dy = int(sampleY) - int(radius);
            uint px = uint(clamp(basePixel.x + dx, 0, int(width) - 1));
            uint py = uint(clamp(basePixel.y + dy, 0, int(height) - 1));
            uint2 pixel = uint2(px, py);
            float4 sample = sanitize_float4(g_PairDatasetSourceFrame[pixel], SANITIZE_HUGE_VALUE);
            float2 sensorCoord = float2(
                (float(px) + 0.5) / max(float(width), 1.0),
                (float(py) + 0.5) / max(float(height), 1.0)
            );
            meanSample += sample;
            meanSensorCoord += sensorCoord;
        }
    }

    float invSampleCount = 1.0 / max(float(neighborhoodSampleCount), 1.0);
    g_ObservationMeanSamples[observationIndex] = meanSample * invSampleCount;
    g_ObservationSensorCoords[observationIndex] = meanSensorCoord * invSampleCount;
}

[Differentiable]
float3 photometric_apply_tonemap(no_diff float3 radiance, no_diff float2 sensorCoord, PPISPTonemap params)
{
    return ppisp_apply_tonemap(radiance, sensorCoord, params);
}

[Differentiable]
float3 photometric_inverse_tonemap(no_diff float3 tonemappedColor, no_diff float2 sensorCoord, PPISPTonemap params)
{
    return ppisp_inverse_tonemap(tonemappedColor, sensorCoord, params);
}

[shader("compute")]
[numthreads(PHOTOMETRIC_TARGET_TILE_DIM, PHOTOMETRIC_TARGET_TILE_DIM, 1)]
void csApplyPhotometricTargetTonemap(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint px = dispatchThreadID.x;
    uint py = dispatchThreadID.y;
    if (px >= g_PhotometricTargetWidth || py >= g_PhotometricTargetHeight) return;

    float4 sample = sanitize_float4(g_PhotometricSourceTarget[uint2(px, py)], SANITIZE_HUGE_VALUE);
    float2 sensorCoord = float2(
        (float(px) + 0.5) / max(float(g_PhotometricSourceWidth), 1.0),
        (float(py) + 0.5) / max(float(g_PhotometricSourceHeight), 1.0)
    );
    sample.xyz = photometric_inverse_tonemap(max(sample.xyz, 0.0), sensorCoord, g_PhotometricTargetPPISPParams);
    g_PhotometricTarget[uint2(px, py)] = sample;
}

[shader("compute")]
[numthreads(PHOTOMETRIC_THREADS, 1, 1)]
void csPhotometricPairLossBackward(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint directedPairIndex = dispatchThreadID.x;
    uint totalDirectedPairCount = g_SampledPairCount * 2u;
    if (directedPairIndex >= totalDirectedPairCount || g_TotalPairCount == 0u || g_TrackCount == 0u) return;

    uint sampleIndex = directedPairIndex >> 1u;
    bool currentIsB = (directedPairIndex & 1u) != 0u;
    uint globalPairIndex = sampled_pair_index(sampleIndex);
    uint trackIndex = find_track_index(globalPairIndex);
    uint localPairIndex = globalPairIndex - g_TrackPairRanges[trackIndex];
    uint observationStart = g_TrackObservationRanges[trackIndex];
    uint observationStop = g_TrackObservationRanges[trackIndex + 1u];
    uint observationCount = observationStop - observationStart;
    if (observationCount < 2u) return;

    uint2 decodedPair = decode_observation_pair_index(localPairIndex, observationCount);
    uint observationIndexA = observationStart + decodedPair.x;
    uint observationIndexB = observationStart + decodedPair.y;
    uint currentObservationIndex = currentIsB ? observationIndexB : observationIndexA;
    uint otherObservationIndex = currentIsB ? observationIndexA : observationIndexB;
    uint currentFrameIndex = g_ObservationFrameIndices[currentObservationIndex];
    uint otherFrameIndex = g_ObservationFrameIndices[otherObservationIndex];
    if (currentFrameIndex >= g_FrameCount || otherFrameIndex >= g_FrameCount) return;

    float lossScale = 1.0 / max(float(g_SampledPairCount), 1.0);
    PPISPTonemap currentParams = load_ppisp(currentFrameIndex);
    PPISPTonemap otherParams = load_ppisp(otherFrameIndex);
    float4 currentSample = load_observation_mean_sample(currentObservationIndex);
    float4 otherSample = load_observation_mean_sample(otherObservationIndex);
    float2 currentSensorCoord = load_observation_sensor_coord(currentObservationIndex);
    float2 otherSensorCoord = load_observation_sensor_coord(otherObservationIndex);
    float3 currentMeanTonemapped = max(currentSample.xyz, 0.0);
    float3 otherMeanLinear = photometric_inverse_tonemap(max(otherSample.xyz, 0.0), otherSensorCoord, otherParams);
    float3 reconstructedCurrentMean = photometric_apply_tonemap(otherMeanLinear, currentSensorCoord, currentParams);
    float3 meanDiff = currentMeanTonemapped - reconstructedCurrentMean;
    float pairLoss = dot(abs(meanDiff), float3(1.0 / 3.0));
    float3 dReconstructedMean = -sign(meanDiff) * (lossScale / 3.0);
    DifferentialPair<PPISPTonemap> dpCurrent = diffPair(currentParams, zero_ppisp_differential());
    bwd_diff(photometric_apply_tonemap)(otherMeanLinear, currentSensorCoord, dpCurrent, dReconstructedMean);

    float localGrads[PPISP_PARAM_COUNT];
    clear_local_grads(localGrads);
    add_local_ppisp_grad(localGrads, dpCurrent.d);
    atomic_add_ppisp_grad(currentFrameIndex, localGrads);
    if (pairLoss != 0.0) InterlockedAdd(g_LossSum[0], pairLoss * lossScale);
}