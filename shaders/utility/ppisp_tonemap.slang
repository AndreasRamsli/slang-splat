import "utility/math/constants.slang";

static const float2 PPISP_WHITE_RG = float2(1.0 / 3.0, 1.0 / 3.0);
static const float PPISP_SOFT_CLAMP_EPS = 1e-4;
static const float PPISP_INVERSE_ATTENUATION_FLOOR = 1e-4;

struct PPISPTonemap : IDifferentiable
{
    float exposureEv;
    float3 vignetteCenterX;
    float3 vignetteCenterY;
    float3 vignetteCoeffR2;
    float3 vignetteCoeffR4;
    float3 vignetteCoeffR6;
    float2 chromaOffsetR;
    float2 chromaOffsetG;
    float2 chromaOffsetB;
    float2 chromaOffsetW;
    float3 crfTau;
    float3 crfEta;
    float3 crfXi;
    float3 crfGamma;

    [Differentiable]
    float3 apply(float3 radiance, no_diff float2 sensorCoord)
    {
        float3 color = ppisp_apply_exposure(radiance, exposureEv);
        color = ppisp_apply_vignetting(color, sensorCoord, this);
        color = ppisp_apply_color_correction(color, this);
        return ppisp_apply_crf(color, this);
    }

    [Differentiable]
    float3 inverse(float3 color, no_diff float2 sensorCoord)
    {
        return ppisp_inverse_tonemap(color, sensorCoord, this);
    }
};

[Differentiable]
float ppisp_soft_floor(float x, no_diff float floorValue)
{
    float delta = x - floorValue;
    return floorValue + 0.5 * (delta + sqrt(delta * delta + floorValue * floorValue));
}

[Differentiable]
float ppisp_soft_unit_clamp(float x, no_diff float eps)
{
    float lower = ppisp_soft_floor(x, eps);
    return 1.0 - ppisp_soft_floor(1.0 - lower, eps);
}

[Differentiable]
float3 ppisp_apply_exposure(float3 radiance, float exposureEv)
{
    return radiance * exp2(exposureEv);
}

[Differentiable]
float3 ppisp_inverse_exposure(float3 color, float exposureEv)
{
    return color * exp2(-exposureEv);
}

[Differentiable]
float3 ppisp_vignette_attenuation(no_diff float2 sensorCoord, PPISPTonemap params)
{
    float3 dx = sensorCoord.x - params.vignetteCenterX;
    float3 dy = sensorCoord.y - params.vignetteCenterY;
    float3 r2 = dx * dx + dy * dy;
    return saturate(1.0 + params.vignetteCoeffR2 * r2 + params.vignetteCoeffR4 * r2 * r2 + params.vignetteCoeffR6 * r2 * r2 * r2);
}

[Differentiable]
float3 ppisp_apply_vignetting(float3 color, no_diff float2 sensorCoord, PPISPTonemap params)
{
    return color * ppisp_vignette_attenuation(sensorCoord, params);
}

[Differentiable]
float3 ppisp_inverse_vignetting(float3 color, no_diff float2 sensorCoord, PPISPTonemap params)
{
    float3 attenuation = ppisp_vignette_attenuation(sensorCoord, params);
    float3 attenuationSafe = float3(
        ppisp_soft_floor(attenuation.x, PPISP_INVERSE_ATTENUATION_FLOOR),
        ppisp_soft_floor(attenuation.y, PPISP_INVERSE_ATTENUATION_FLOOR),
        ppisp_soft_floor(attenuation.z, PPISP_INVERSE_ATTENUATION_FLOOR)
    );
    return color / attenuationSafe;
}

[Differentiable]
float3 ppisp_lift_chroma(float2 rg)
{
    return float3(rg, 1.0);
}

[Differentiable]
float3 ppisp_homography_nullspace(float3 tR, float3 tG, float3 tB, float3 tW)
{
    float3 c0 = cross(tW, tR);
    float3 c1 = cross(tW, tG);
    float3 c2 = cross(tW, tB);
    float3 m0 = float3(c0.x, c1.x, c2.x);
    float3 m1 = float3(c0.y, c1.y, c2.y);
    float3 m2 = float3(c0.z, c1.z, c2.z);
    float3 k = cross(m0, m1);
    if (dot(k, k) <= TINY_VALUE) k = cross(m0, m2);
    if (dot(k, k) <= TINY_VALUE) k = cross(m1, m2);
    return k;
}

[Differentiable]
float3x3 ppisp_color_correction_matrix(PPISPTonemap params)
{
    float3 tR = ppisp_lift_chroma(float2(1.0, 0.0) + params.chromaOffsetR);
    float3 tG = ppisp_lift_chroma(float2(0.0, 1.0) + params.chromaOffsetG);
    float3 tB = ppisp_lift_chroma(params.chromaOffsetB);
    float3 tW = ppisp_lift_chroma(PPISP_WHITE_RG + params.chromaOffsetW);
    float3 k = ppisp_homography_nullspace(tR, tG, tB, tW);
    if (dot(k, k) <= TINY_VALUE) return float3x3(0.0);

    float3 sR = tR * k.x;
    float3 sG = tG * k.y;
    float3 sB = tB * k.z;
    if (abs(sB.z) <= TINY_VALUE) return float3x3(0.0);

    float invScale = 1.0 / sB.z;
    float3 h0 = (sR - sB) * invScale;
    float3 h1 = (sG - sB) * invScale;
    float3 h2 = sB * invScale;
    return float3x3(h0, h1, h2);
}

[Differentiable]
float ppisp_matrix_sq_norm(float3x3 m)
{
    return dot(m[0], m[0]) + dot(m[1], m[1]) + dot(m[2], m[2]);
}

[Differentiable]
float3x3 ppisp_inverse3x3_or_zero(float3x3 m)
{
    float det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
              - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
              + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if (abs(det) <= TINY_VALUE) return float3x3(0.0);
    float invDet = 1.0 / det;
    return float3x3(
        float3(
            m[1][1] * m[2][2] - m[1][2] * m[2][1],
            m[0][2] * m[2][1] - m[0][1] * m[2][2],
            m[0][1] * m[1][2] - m[0][2] * m[1][1]
        ) * invDet,
        float3(
            m[1][2] * m[2][0] - m[1][0] * m[2][2],
            m[0][0] * m[2][2] - m[0][2] * m[2][0],
            m[0][2] * m[1][0] - m[0][0] * m[1][2]
        ) * invDet,
        float3(
            m[1][0] * m[2][1] - m[1][1] * m[2][0],
            m[0][1] * m[2][0] - m[0][0] * m[2][1],
            m[0][0] * m[1][1] - m[0][1] * m[1][0]
        ) * invDet
    );
}

[Differentiable]
float3 ppisp_apply_color_correction(float3 color, PPISPTonemap params)
{
    float intensity = max(color.x + color.y + color.z, 0.0);
    float3 xRGI = float3(color.x, color.y, intensity);
    float3x3 h = ppisp_color_correction_matrix(params);
    if (ppisp_matrix_sq_norm(h) <= TINY_VALUE) return color;

    float3 correctedRGI = h[0] * xRGI.x + h[1] * xRGI.y + h[2] * xRGI.z;
    correctedRGI *= intensity / max(correctedRGI.z + TINY_VALUE, SMALL_VALUE);
    return float3(correctedRGI.x, correctedRGI.y, correctedRGI.z - correctedRGI.x - correctedRGI.y);
}

[Differentiable]
float3 ppisp_inverse_color_correction(float3 color, PPISPTonemap params)
{
    float intensity = max(color.x + color.y + color.z, 0.0);
    if (intensity <= TINY_VALUE) return 0.0;

    float3x3 h = ppisp_color_correction_matrix(params);
    if (ppisp_matrix_sq_norm(h) <= TINY_VALUE) return color;

    float3x3 inverseH = ppisp_inverse3x3_or_zero(h);
    if (ppisp_matrix_sq_norm(inverseH) <= TINY_VALUE) return color;

    float3 correctedRGI = float3(color.x, color.y, intensity);
    float3 sourceRGI = inverseH[0] * correctedRGI.x + inverseH[1] * correctedRGI.y + inverseH[2] * correctedRGI.z;
    if (sourceRGI.z <= TINY_VALUE) return color;
    sourceRGI *= intensity / sourceRGI.z;
    return float3(sourceRGI.x, sourceRGI.y, intensity - sourceRGI.x - sourceRGI.y);
}

[Differentiable]
float ppisp_positive_pow(float x, float exponent)
{
    return x <= 0.0 ? 0.0 : pow(max(x, TINY_VALUE), exponent);
}

[Differentiable]
float ppisp_apply_crf_channel(float x, float tau, float eta, float xi, float gamma)
{
    float x01 = saturate(x);
    float tauSafe = max(tau, SMALL_VALUE);
    float etaSafe = max(eta, SMALL_VALUE);
    float xiSafe = clamp(xi, SMALL_VALUE, 1.0 - SMALL_VALUE);
    float gammaSafe = max(gamma, SMALL_VALUE);
    float a = (etaSafe * xiSafe) / max(tauSafe * (1.0 - xiSafe) + etaSafe * xiSafe, SMALL_VALUE);
    float b = 1.0 - a;
    float lower = a * ppisp_positive_pow(x01 / xiSafe, tauSafe);
    float upper = 1.0 - b * ppisp_positive_pow((1.0 - x01) / (1.0 - xiSafe), etaSafe);
    return ppisp_positive_pow(saturate(x01 <= xiSafe ? lower : upper), gammaSafe);
}

[Differentiable]
float ppisp_inverse_crf_channel(float y, float tau, float eta, float xi, float gamma)
{
    float y01 = ppisp_soft_unit_clamp(y, PPISP_SOFT_CLAMP_EPS);
    float tauSafe = ppisp_soft_floor(tau, SMALL_VALUE);
    float etaSafe = ppisp_soft_floor(eta, SMALL_VALUE);
    float xiSafe = ppisp_soft_unit_clamp(xi, SMALL_VALUE);
    float gammaSafe = ppisp_soft_floor(gamma, SMALL_VALUE);
    float denom = ppisp_soft_floor(tauSafe * (1.0 - xiSafe) + etaSafe * xiSafe, SMALL_VALUE);
    float a = (etaSafe * xiSafe) / denom;
    float aSafe = ppisp_soft_floor(a, SMALL_VALUE);
    float bSafe = ppisp_soft_floor(1.0 - a, SMALL_VALUE);
    float breakpoint = ppisp_positive_pow(aSafe, gammaSafe);
    float z = ppisp_positive_pow(y01, 1.0 / gammaSafe);
    float x01 = y01 <= breakpoint
        ? xiSafe * ppisp_positive_pow(z / aSafe, 1.0 / tauSafe)
        : 1.0 - (1.0 - xiSafe) * ppisp_positive_pow((1.0 - z) / bSafe, 1.0 / etaSafe);
    return ppisp_soft_unit_clamp(x01, PPISP_SOFT_CLAMP_EPS);
}

[Differentiable]
float3 ppisp_apply_crf(float3 color, PPISPTonemap params)
{
    return float3(
        ppisp_apply_crf_channel(color.x, params.crfTau.x, params.crfEta.x, params.crfXi.x, params.crfGamma.x),
        ppisp_apply_crf_channel(color.y, params.crfTau.y, params.crfEta.y, params.crfXi.y, params.crfGamma.y),
        ppisp_apply_crf_channel(color.z, params.crfTau.z, params.crfEta.z, params.crfXi.z, params.crfGamma.z)
    );
}

[Differentiable]
float3 ppisp_inverse_crf(float3 color, PPISPTonemap params)
{
    return float3(
        ppisp_inverse_crf_channel(color.x, params.crfTau.x, params.crfEta.x, params.crfXi.x, params.crfGamma.x),
        ppisp_inverse_crf_channel(color.y, params.crfTau.y, params.crfEta.y, params.crfXi.y, params.crfGamma.y),
        ppisp_inverse_crf_channel(color.z, params.crfTau.z, params.crfEta.z, params.crfXi.z, params.crfGamma.z)
    );
}

[Differentiable]
float3 ppisp_apply_tonemap(float3 radiance, no_diff float2 sensorCoord, PPISPTonemap params)
{
    return params.apply(radiance, sensorCoord);
}

[Differentiable]
float3 ppisp_inverse_tonemap(float3 color, no_diff float2 sensorCoord, PPISPTonemap params)
{
    float3 radiance = ppisp_inverse_crf(color, params);
    radiance = ppisp_inverse_color_correction(radiance, params);
    radiance = ppisp_inverse_vignetting(radiance, sensorCoord, params);
    return ppisp_inverse_exposure(radiance, params.exposureEv);
}
