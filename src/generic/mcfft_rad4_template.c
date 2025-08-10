/**
 * MIT License
 * 
 * Copyright (c) 2025 Georgii Zagoruiko
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "utils.h"

static void MC_FUNC_TEMPLATE(MC_FFT_DIRECTION, rad4_mono_depth1, g) (float * restrict re, float * restrict im, uint32_t fftLength) {
    for (uint32_t stepIdx = 0; stepIdx < fftLength; stepIdx += 4u) {
            float t0_re = *re + re[2u];
            float t0_im = *im + im[2u];
            float t1_re = *re - re[2u];
            float t1_im = *im - im[2u];
            float t2_re = re[1u] + re[3u];
            float t2_im = im[1u] + im[3u];
            float t3_re = im[1u] - im[3u]; // conj
            float t3_im = re[3u] - re[1u]; // conj

            *re++ = t0_re + t2_re;
            *im++ = t0_im + t2_im;
#if MC_INVERSE_FFT
            *re++ = t1_re - t3_re;
            *im++ = t1_im - t3_im;
#else
            *re++ = t1_re + t3_re;
            *im++ = t1_im + t3_im;
#endif
            *re++ = t0_re - t2_re;
            *im++ = t0_im - t2_im;
#if MC_INVERSE_FFT
            *re++ = t1_re + t3_re;
            *im++ = t1_im + t3_im;
#else
            *re++ = t1_re - t3_re;
            *im++ = t1_im - t3_im;
#endif
    }
}

static void MC_FUNC_TEMPLATE(MC_FFT_DIRECTION, dif_rad4_mono_depth2, g) (float * restrict re, float * restrict im, const float * restrict twiddle, uint32_t fftLength) {
    float * restrict aRe = re;
    float * restrict aIm = im;
    float * restrict bRe = &re[4u];
    float * restrict bIm = &im[4u];
    float * restrict cRe = &re[8u];
    float * restrict cIm = &im[8u];
    float * restrict dRe = &re[12u];
    float * restrict dIm = &im[12u];
    for (uint32_t stepIdx = 0; stepIdx < fftLength; stepIdx += 16u) {
        const float *twdB_re = twiddle;
        const float *twdB_im = &twiddle[4u];
        const float *twdC_re = &twiddle[8u];
        const float *twdC_im = &twiddle[12u];
        const float *twdD_re = &twiddle[16u];
        const float *twdD_im = &twiddle[20u];
        for (uint32_t i = 0; i < 4u; ++i) {
            float t0_re = *aRe + *cRe;
            float t0_im = *aIm + *cIm;
            float t1_re = *aRe - *cRe;
            float t1_im = *aIm - *cIm;
            float t2_re = *bRe + *dRe;
            float t2_im = *bIm + *dIm;
            float t3_re = *bIm - *dIm; // conj
            float t3_im = *dRe - *bRe; // conj

            *aRe++ = t0_re + t2_re;
            *aIm++ = t0_im + t2_im;
            float sumC_re = t0_re - t2_re;
            float sumC_im = t0_im - t2_im;
#if MC_INVERSE_FFT
            float sumB_re = t1_re - t3_re;
            float sumB_im = t1_im - t3_im;
            float sumD_re = t1_re + t3_re;
            float sumD_im = t1_im + t3_im;

            *bRe++ = twdB_re[i] * sumB_re + twdB_im[i] * sumB_im;
            *bIm++ = twdB_re[i] * sumB_im - twdB_im[i] * sumB_re;
            *cRe++ = twdC_re[i] * sumC_re + twdC_im[i] * sumC_im;
            *cIm++ = twdC_re[i] * sumC_im - twdC_im[i] * sumC_re;
            *dRe++ = twdD_re[i] * sumD_re + twdD_im[i] * sumD_im;
            *dIm++ = twdD_re[i] * sumD_im - twdD_im[i] * sumD_re;
#else
            float sumB_re = t1_re + t3_re;
            float sumB_im = t1_im + t3_im;
            float sumD_re = t1_re - t3_re;
            float sumD_im = t1_im - t3_im;

            *bRe++ = twdB_re[i] * sumB_re - twdB_im[i] * sumB_im;
            *bIm++ = twdB_re[i] * sumB_im + twdB_im[i] * sumB_re;
            *cRe++ = twdC_re[i] * sumC_re - twdC_im[i] * sumC_im;
            *cIm++ = twdC_re[i] * sumC_im + twdC_im[i] * sumC_re;
            *dRe++ = twdD_re[i] * sumD_re - twdD_im[i] * sumD_im;
            *dIm++ = twdD_re[i] * sumD_im + twdD_im[i] * sumD_re;
#endif
        }
        aRe += 12u;
        aIm += 12u;
        bRe += 12u;
        bIm += 12u;
        cRe += 12u;
        cIm += 12u;
        dRe += 12u;
        dIm += 12u;
    }
}

static void MC_FUNC_TEMPLATE(MC_FFT_DIRECTION, dif_rad4_mono_depth2_odd, g) (float * restrict re, float * restrict im, const float * restrict twiddle, uint32_t fftLength) {
    float * restrict aRe = re;
    float * restrict aIm = im;
    float * restrict bRe = &re[2u];
    float * restrict bIm = &im[2u];
    float * restrict cRe = &re[4u];
    float * restrict cIm = &im[4u];
    float * restrict dRe = &re[6u];
    float * restrict dIm = &im[6u];

    for (uint32_t stepIdx = 0; stepIdx < fftLength; stepIdx += 8u) {
        float t0_re = aRe[0] + cRe[0];
        float t0_im = aIm[0] + cIm[0];
        float t1_re = aRe[0] - cRe[0];
        float t1_im = aIm[0] - cIm[0];
        float t2_re = bRe[0] + dRe[0];
        float t2_im = bIm[0] + dIm[0];
        float t3_re = bIm[0] - dIm[0]; // conj
        float t3_im = dRe[0] - bRe[0]; // conj

        aRe[0] = t0_re + t2_re;
        aIm[0] = t0_im + t2_im;
        cRe[0] = t0_re - t2_re;
        cIm[0] = t0_im - t2_im;
#if MC_INVERSE_FFT
        bRe[0] = t1_re - t3_re;
        bIm[0] = t1_im - t3_im;
        dRe[0] = t1_re + t3_re;
        dIm[0] = t1_im + t3_im;
#else
        bRe[0] = t1_re + t3_re;
        bIm[0] = t1_im + t3_im;
        dRe[0] = t1_re - t3_re;
        dIm[0] = t1_im - t3_im;
#endif
        t0_re = aRe[1u] + cRe[1u];
        t0_im = aIm[1u] + cIm[1u];
        t1_re = aRe[1u] - cRe[1u];
        t1_im = aIm[1u] - cIm[1u];
        t2_re = bRe[1u] + dRe[1u];
        t2_im = bIm[1u] + dIm[1u];
        t3_re = bIm[1u] - dIm[1u]; // conj
        t3_im = dRe[1u] - bRe[1u]; // conj

        aRe[1u] = t0_re + t2_re;
        aIm[1u] = t0_im + t2_im;
        float sumC_re = t0_re - t2_re;
        float sumC_im = t0_im - t2_im;
#if MC_INVERSE_FFT
        float sumB_re = t1_re - t3_re;
        float sumB_im = t1_im - t3_im;
        float sumD_re = t1_re + t3_re;
        float sumD_im = t1_im + t3_im;

        bRe[1u] = sumB_re*twiddle[0] + sumB_im*twiddle[1u];
        bIm[1u] = sumB_im*twiddle[0] - sumB_re*twiddle[1u];
        cRe[1u] = sumC_re*twiddle[2u] + sumC_im*twiddle[3u];
        cIm[1u] = sumC_im*twiddle[2u] - sumC_re*twiddle[3u];
        dRe[1u] = sumD_re*twiddle[4u] + sumD_im*twiddle[5u];
        dIm[1u] = sumD_im*twiddle[4u] - sumD_re*twiddle[5u];
#else
        float sumB_re = t1_re + t3_re;
        float sumB_im = t1_im + t3_im;
        float sumD_re = t1_re - t3_re;
        float sumD_im = t1_im - t3_im;

        bRe[1u] = sumB_re*twiddle[0] - sumB_im*twiddle[1u];
        bIm[1u] = sumB_im*twiddle[0] + sumB_re*twiddle[1u];
        cRe[1u] = sumC_re*twiddle[2u] - sumC_im*twiddle[3u];
        cIm[1u] = sumC_im*twiddle[2u] + sumC_re*twiddle[3u];
        dRe[1u] = sumD_re*twiddle[4u] - sumD_im*twiddle[5u];
        dIm[1u] = sumD_im*twiddle[4u] + sumD_re*twiddle[5u];
#endif
        aRe += 8u;
        aIm += 8u;
        bRe += 8u;
        bIm += 8u;
        cRe += 8u;
        cIm += 8u;
        dRe += 8u;
        dIm += 8u;
    }
}

static void MC_FUNC_TEMPLATE(MC_FFT_DIRECTION, dif_rad4_mono_loop, g) (float * restrict re, float * restrict im, const float * restrict twiddle, uint32_t fftLength, uint32_t step) {
    const uint32_t qStep = step>>2u;
    float * restrict aRe = re;
    float * restrict aIm = im;
    float * restrict bRe = &re[qStep];
    float * restrict bIm = &im[qStep];
    float * restrict cRe = &re[2u*qStep];
    float * restrict cIm = &im[2u*qStep];
    float * restrict dRe = &re[3u*qStep];
    float * restrict dIm = &im[3u*qStep];
    for (uint32_t stepIdx = 0; stepIdx < fftLength; stepIdx += step) {
        const float *twdB_re = twiddle;
        const float *twdB_im = &twiddle[8u];
        const float *twdC_re = &twiddle[16u];
        const float *twdC_im = &twiddle[24u];
        const float *twdD_re = &twiddle[32u];
        const float *twdD_im = &twiddle[40u];
        for (uint32_t i = 0; i < qStep; i += 8u) {
            for (uint32_t j = 0; j < 8u; ++j) {
                float t0_re = *aRe + *cRe;
                float t0_im = *aIm + *cIm;
                float t1_re = *aRe - *cRe;
                float t1_im = *aIm - *cIm;
                float t2_re = *bRe + *dRe;
                float t2_im = *bIm + *dIm;
                float t3_re = *bIm - *dIm; // conj
                float t3_im = *dRe - *bRe; // conj

                *aRe++ = t0_re + t2_re;
                *aIm++ = t0_im + t2_im;
                float sumC_re = t0_re - t2_re;
                float sumC_im = t0_im - t2_im;
#if MC_INVERSE_FFT
                float sumB_re = t1_re - t3_re;
                float sumB_im = t1_im - t3_im;
                float sumD_re = t1_re + t3_re;
                float sumD_im = t1_im + t3_im;

                *bRe++ = sumB_re * twdB_re[j] + sumB_im * twdB_im[j];
                *bIm++ = sumB_im * twdB_re[j] - sumB_re * twdB_im[j];
                *cRe++ = sumC_re * twdC_re[j] + sumC_im * twdC_im[j];
                *cIm++ = sumC_im * twdC_re[j] - sumC_re * twdC_im[j];
                *dRe++ = sumD_re * twdD_re[j] + sumD_im * twdD_im[j];
                *dIm++ = sumD_im * twdD_re[j] - sumD_re * twdD_im[j];
#else
                float sumB_re = t1_re + t3_re;
                float sumB_im = t1_im + t3_im;
                float sumD_re = t1_re - t3_re;
                float sumD_im = t1_im - t3_im;

                *bRe++ = sumB_re * twdB_re[j] - sumB_im * twdB_im[j];
                *bIm++ = sumB_im * twdB_re[j] + sumB_re * twdB_im[j];
                *cRe++ = sumC_re * twdC_re[j] - sumC_im * twdC_im[j];
                *cIm++ = sumC_im * twdC_re[j] + sumC_re * twdC_im[j];
                *dRe++ = sumD_re * twdD_re[j] - sumD_im * twdD_im[j];
                *dIm++ = sumD_im * twdD_re[j] + sumD_re * twdD_im[j];
#endif
            }
            twdB_re += 48u;
            twdB_im += 48u;
            twdC_re += 48u;
            twdC_im += 48u;
            twdD_re += 48u;
            twdD_im += 48u;
        }
        aRe += 3u*qStep;
        aIm += 3u*qStep;
        bRe += 3u*qStep;
        bIm += 3u*qStep;
        cRe += 3u*qStep;
        cIm += 3u*qStep;
        dRe += 3u*qStep;
        dIm += 3u*qStep;
    }
}

static void MC_FUNC_TEMPLATE(MC_FFT_DIRECTION, dit_rad4_mono_depth2, g) (float * restrict re, float * restrict im, const float * restrict twiddle, uint32_t fftLength) {
    float * restrict aRe = re;
    float * restrict aIm = im;
    float * restrict bRe = &re[4u];
    float * restrict bIm = &im[4u];
    float * restrict cRe = &re[8u];
    float * restrict cIm = &im[8u];
    float * restrict dRe = &re[12u];
    float * restrict dIm = &im[12u];
    for (uint32_t stepIdx = 0; stepIdx < fftLength; stepIdx += 16u) {
        const float *twdB_re = twiddle;
        const float *twdB_im = &twiddle[4u];
        const float *twdC_re = &twiddle[8u];
        const float *twdC_im = &twiddle[12u];
        const float *twdD_re = &twiddle[16u];
        const float *twdD_im = &twiddle[20u];
        for (uint32_t i = 0; i < 4u; ++i) {
#if MC_INVERSE_FFT
            float acc0_re = *bRe * twdB_re[i] + *bIm * twdB_im[i];
            float acc0_im = *bIm * twdB_re[i] - *bRe * twdB_im[i];
            float acc1_re = *cRe * twdC_re[i] + *cIm * twdC_im[i];
            float acc1_im = *cIm * twdC_re[i] - *cRe * twdC_im[i];
            float acc2_re = *dRe * twdD_re[i] + *dIm * twdD_im[i];
            float acc2_im = *dIm * twdD_re[i] - *dRe * twdD_im[i];
#else
            float acc0_re = *bRe * twdB_re[i] - *bIm * twdB_im[i];
            float acc0_im = *bIm * twdB_re[i] + *bRe * twdB_im[i];
            float acc1_re = *cRe * twdC_re[i] - *cIm * twdC_im[i];
            float acc1_im = *cIm * twdC_re[i] + *cRe * twdC_im[i];
            float acc2_re = *dRe * twdD_re[i] - *dIm * twdD_im[i];
            float acc2_im = *dIm * twdD_re[i] + *dRe * twdD_im[i];
#endif
            float t0_re = *aRe + acc1_re;
            float t0_im = *aIm + acc1_im;
            float t1_re = *aRe - acc1_re;
            float t1_im = *aIm - acc1_im;
            float t2_re = acc0_re + acc2_re;
            float t2_im = acc0_im + acc2_im;
            float t3_re = acc0_im - acc2_im; // conj
            float t3_im = acc2_re - acc0_re; // conj

            *aRe++ = t0_re + t2_re;
            *aIm++ = t0_im + t2_im;
#if MC_INVERSE_FFT
            *bRe++ = t1_re - t3_re;
            *bIm++ = t1_im - t3_im;
#else
            *bRe++ = t1_re + t3_re;
            *bIm++ = t1_im + t3_im;
#endif
            *cRe++ = t0_re - t2_re;
            *cIm++ = t0_im - t2_im;
#if MC_INVERSE_FFT
            *dRe++ = t1_re + t3_re;
            *dIm++ = t1_im + t3_im;
#else
            *dRe++ = t1_re - t3_re;
            *dIm++ = t1_im - t3_im;
#endif
        }
        aRe += 12u;
        aIm += 12u;
        bRe += 12u;
        bIm += 12u;
        cRe += 12u;
        cIm += 12u;
        dRe += 12u;
        dIm += 12u;
    }
}

static void MC_FUNC_TEMPLATE(MC_FFT_DIRECTION, dit_rad4_mono_depth2_odd, g) (float * restrict re, float * restrict im, const float * restrict twiddle, uint32_t fftLength) {
    float * restrict aRe = re;
    float * restrict aIm = im;
    float * restrict bRe = &re[2u];
    float * restrict bIm = &im[2u];
    float * restrict cRe = &re[4u];
    float * restrict cIm = &im[4u];
    float * restrict dRe = &re[6u];
    float * restrict dIm = &im[6u];

    for (uint32_t stepIdx = 0; stepIdx < fftLength; stepIdx += 8u) {
        float t0_re = aRe[0] + cRe[0];
        float t0_im = aIm[0] + cIm[0];
        float t1_re = aRe[0] - cRe[0];
        float t1_im = aIm[0] - cIm[0];
        float t2_re = bRe[0] + dRe[0];
        float t2_im = bIm[0] + dIm[0];
        float t3_re = bIm[0] - dIm[0]; // conj
        float t3_im = dRe[0] - bRe[0]; // conj

        aRe[0] = t0_re + t2_re;
        aIm[0] = t0_im + t2_im;
#if MC_INVERSE_FFT
        bRe[0] = t1_re - t3_re;
        bIm[0] = t1_im - t3_im;
#else
        bRe[0] = t1_re + t3_re;
        bIm[0] = t1_im + t3_im;
#endif
        cRe[0] = t0_re - t2_re;
        cIm[0] = t0_im - t2_im;
#if MC_INVERSE_FFT
        dRe[0] = t1_re + t3_re;
        dIm[0] = t1_im + t3_im;
#else
        dRe[0] = t1_re - t3_re;
        dIm[0] = t1_im - t3_im;
#endif

#if MC_INVERSE_FFT
        float acc0_re = bRe[1u] * twiddle[0] + bIm[1u] * twiddle[1u];
        float acc0_im = bIm[1u] * twiddle[0] - bRe[1u] * twiddle[1u];
        float acc1_re = cRe[1u] * twiddle[2u] + cIm[1u] * twiddle[3u];
        float acc1_im = cIm[1u] * twiddle[2u] - cRe[1u] * twiddle[3u];
        float acc2_re = dRe[1u] * twiddle[4u] + dIm[1u] * twiddle[5u];
        float acc2_im = dIm[1u] * twiddle[4u] - dRe[1u] * twiddle[5u];
#else
        float acc0_re = bRe[1u] * twiddle[0] - bIm[1u] * twiddle[1u];
        float acc0_im = bIm[1u] * twiddle[0] + bRe[1u] * twiddle[1u];
        float acc1_re = cRe[1u] * twiddle[2u] - cIm[1u] * twiddle[3u];
        float acc1_im = cIm[1u] * twiddle[2u] + cRe[1u] * twiddle[3u];
        float acc2_re = dRe[1u] * twiddle[4u] - dIm[1u] * twiddle[5u];
        float acc2_im = dIm[1u] * twiddle[4u] + dRe[1u] * twiddle[5u];
#endif
        t0_re = aRe[1u] + acc1_re;
        t0_im = aIm[1u] + acc1_im;
        t1_re = aRe[1u] - acc1_re;
        t1_im = aIm[1u] - acc1_im;
        t2_re = acc0_re + acc2_re;
        t2_im = acc0_im + acc2_im;
        t3_re = acc0_im - acc2_im; // conj
        t3_im = acc2_re - acc0_re; // conj

        aRe[1u] = t0_re + t2_re;
        aIm[1u] = t0_im + t2_im;
#if MC_INVERSE_FFT
        bRe[1u] = t1_re - t3_re;
        bIm[1u] = t1_im - t3_im;
#else
        bRe[1u] = t1_re + t3_re;
        bIm[1u] = t1_im + t3_im;
#endif
        cRe[1u] = t0_re - t2_re;
        cIm[1u] = t0_im - t2_im;
#if MC_INVERSE_FFT
        dRe[1u] = t1_re + t3_re;
        dIm[1u] = t1_im + t3_im;
#else
        dRe[1u] = t1_re - t3_re;
        dIm[1u] = t1_im - t3_im;
#endif
        aRe += 8u;
        aIm += 8u;
        bRe += 8u;
        bIm += 8u;
        cRe += 8u;
        cIm += 8u;
        dRe += 8u;
        dIm += 8u;
    }
}

static void MC_FUNC_TEMPLATE(MC_FFT_DIRECTION, dit_rad4_mono_loop, g) (float * restrict re, float * restrict im, const float * restrict twiddle, uint32_t fftLength, uint32_t step) {
    const uint32_t qStep = step>>2u;
    float * restrict aRe = re;
    float * restrict aIm = im;
    float * restrict bRe = &re[qStep];
    float * restrict bIm = &im[qStep];
    float * restrict cRe = &re[2u*qStep];
    float * restrict cIm = &im[2u*qStep];
    float * restrict dRe = &re[3u*qStep];
    float * restrict dIm = &im[3u*qStep];
    for (uint32_t stepIdx = 0; stepIdx < fftLength; stepIdx += step) {
        const float *twdB_re = twiddle;
        const float *twdB_im = &twiddle[8u];
        const float *twdC_re = &twiddle[16u];
        const float *twdC_im = &twiddle[24u];
        const float *twdD_re = &twiddle[32u];
        const float *twdD_im = &twiddle[40u];
        for (uint32_t i = 0; i < qStep; i += 8u) {
            for (uint32_t j = 0; j < 8u; ++j) {
#if MC_INVERSE_FFT
                float acc0_re = *bRe * twdB_re[j] + *bIm * twdB_im[j];
                float acc0_im = *bIm * twdB_re[j] - *bRe * twdB_im[j];
                float acc1_re = *cRe * twdC_re[j] + *cIm * twdC_im[j];
                float acc1_im = *cIm * twdC_re[j] - *cRe * twdC_im[j];
                float acc2_re = *dRe * twdD_re[j] + *dIm * twdD_im[j];
                float acc2_im = *dIm * twdD_re[j] - *dRe * twdD_im[j];
#else
                float acc0_re = *bRe * twdB_re[j] - *bIm * twdB_im[j];
                float acc0_im = *bIm * twdB_re[j] + *bRe * twdB_im[j];
                float acc1_re = *cRe * twdC_re[j] - *cIm * twdC_im[j];
                float acc1_im = *cIm * twdC_re[j] + *cRe * twdC_im[j];
                float acc2_re = *dRe * twdD_re[j] - *dIm * twdD_im[j];
                float acc2_im = *dIm * twdD_re[j] + *dRe * twdD_im[j];
#endif
                float t0_re = *aRe + acc1_re;
                float t0_im = *aIm + acc1_im;
                float t1_re = *aRe - acc1_re;
                float t1_im = *aIm - acc1_im;
                float t2_re = acc0_re + acc2_re;
                float t2_im = acc0_im + acc2_im;
                float t3_re = acc0_im - acc2_im; // conj
                float t3_im = acc2_re - acc0_re; // conj

                *aRe++ = t0_re + t2_re;
                *aIm++ = t0_im + t2_im;
#if MC_INVERSE_FFT
                *bRe++ = t1_re - t3_re;
                *bIm++ = t1_im - t3_im;
#else
                *bRe++ = t1_re + t3_re;
                *bIm++ = t1_im + t3_im;
#endif
                *cRe++ = t0_re - t2_re;
                *cIm++ = t0_im - t2_im;
#if MC_INVERSE_FFT
                *dRe++ = t1_re + t3_re;
                *dIm++ = t1_im + t3_im;
#else
                *dRe++ = t1_re - t3_re;
                *dIm++ = t1_im - t3_im;
#endif
            }
            twdB_re += 48u;
            twdB_im += 48u;
            twdC_re += 48u;
            twdC_im += 48u;
            twdD_re += 48u;
            twdD_im += 48u;
        }
        aRe += 3u*qStep;
        aIm += 3u*qStep;
        bRe += 3u*qStep;
        bIm += 3u*qStep;
        cRe += 3u*qStep;
        cIm += 3u*qStep;
        dRe += 3u*qStep;
        dIm += 3u*qStep;
    }
}



