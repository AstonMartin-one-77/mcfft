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

#include "mcfft_neon.h"
#include "generic/mcfft_generic.h"
#define MC_FFT_DIRECTION fft
#define MC_INVERSE_FFT (0)
#include "mcfft_rad4_template_neon.c"
#undef MC_FFT_DIRECTION
#undef MC_INVERSE_FFT

#define MC_FFT_DIRECTION ifft
#define MC_INVERSE_FFT (1u)
#include "mcfft_rad4_template_neon.c"

void mc_shuffle_mono_neon(float * restrict re, float * restrict im, float * restrict buffer, 
                         const uint16_t * restrict digitRev,  uint32_t length) {
    mc_shuffle_mono_g(re, im, buffer, digitRev, length);
}

static void st_rad2_mono_depth1_neon(float *re, float *im, uint32_t fftLength) {
    for (uint32_t stepIdx = 0; stepIdx < fftLength; stepIdx += 8u) {
        float32x4x2_t re_v = vld2q_f32(re);
        float32x4x2_t im_v = vld2q_f32(im);
        
        float32x4_t sbRe_v = vsubq_f32(re_v.val[0], re_v.val[1u]);
        float32x4_t sbIm_v = vsubq_f32(im_v.val[0], im_v.val[1u]);
        re_v.val[0] = vaddq_f32(re_v.val[0], re_v.val[1u]);
        re_v.val[1u] = sbRe_v;
        im_v.val[0] = vaddq_f32(im_v.val[0], im_v.val[1u]);
        im_v.val[1u] = sbIm_v;
        vst2q_f32(re, re_v);
        vst2q_f32(im, im_v);
        re += 8u;
        im += 8u;
    }
}

void mc_fft_dif_mono_core_neon(float * restrict re, float * restrict im, const float * restrict twiddle, uint32_t pow2) {
    const uint32_t fftLength = 1u<<(pow2);
    uint32_t step = fftLength;
    do {
        st_fft_dif_rad4_mono_loop_neon(re, im, twiddle, fftLength, step);
        twiddle += MC_TWIDDLE_STAGE_SIZE(step);
        step >>= 2u;
    } while (step > 16u);
    if (pow2 % 2u) {
        st_fft_dif_rad4_mono_depth2_odd_neon(re, im, twiddle, fftLength);
        st_rad2_mono_depth1_neon(re, im, fftLength);
    } else {
        st_fft_dif_rad4_mono_depth2_neon(re, im, twiddle, fftLength);
        st_fft_rad4_mono_depth1_neon(re, im, fftLength);
    }
}

void mc_ifft_dif_mono_core_neon(float * restrict re, float * restrict im, const float * restrict twiddle, uint32_t pow2) {
    const uint32_t fftLength = 1u<<(pow2);
    uint32_t step = fftLength;
    do {
        st_ifft_dif_rad4_mono_loop_neon(re, im, twiddle, fftLength, step);
        twiddle += MC_TWIDDLE_STAGE_SIZE(step);
        step >>= 2u;
    } while (step > 16u);
    if (pow2 % 2u) {
        st_ifft_dif_rad4_mono_depth2_odd_neon(re, im, twiddle, fftLength);
        st_rad2_mono_depth1_neon(re, im, fftLength);
    } else {
        st_ifft_dif_rad4_mono_depth2_neon(re, im, twiddle, fftLength);
        st_ifft_rad4_mono_depth1_neon(re, im, fftLength);
    }
}

void mc_fft_dit_mono_core_neon(float * restrict re, float * restrict im, const float * restrict twiddle, uint32_t pow2) {
    const uint32_t fftLength = 1u<<(pow2);
    uint32_t step = 0;
    if (pow2 % 2u) {
        /* Shift pointer of twiddle factor to the end, more twiddle factors first for better memory alignement */
        twiddle += MC_TWIDDLE_LENGTH(pow2);
        step = 8u;
        twiddle -= MC_TWIDDLE_STAGE_SIZE(step);
        st_rad2_mono_depth1_neon(re, im, fftLength);
        st_fft_dit_rad4_mono_depth2_odd_neon(re, im, twiddle, fftLength);
    } else {
        /* Shift pointer of twiddle factor to the end, more twiddle factors first for better memory alignement */
        twiddle += MC_TWIDDLE_LENGTH(pow2);
        step = 16;
        twiddle -= MC_TWIDDLE_STAGE_SIZE(step);
        st_fft_rad4_mono_depth1_neon(re, im, fftLength);
        st_fft_dit_rad4_mono_depth2_neon(re, im, twiddle, fftLength);
    }
    do {
        step <<= 2u;
        twiddle -= MC_TWIDDLE_STAGE_SIZE(step);
        st_fft_dit_rad4_mono_loop_neon(re, im, twiddle, fftLength, step);
    } while (step != fftLength);
}

void mc_ifft_dit_mono_core_neon(float * restrict re, float * restrict im, const float * restrict twiddle, uint32_t pow2) {
    const uint32_t fftLength = 1u<<(pow2);
    uint32_t step = 0;
    if (pow2 % 2u) {
        /* Shift pointer of twiddle factor to the end, more twiddle factors first for better memory alignement */
        twiddle += MC_TWIDDLE_LENGTH(pow2);
        step = 8u;
        twiddle -= MC_TWIDDLE_STAGE_SIZE(step);
        st_rad2_mono_depth1_neon(re, im, fftLength);
        st_ifft_dit_rad4_mono_depth2_odd_neon(re, im, twiddle, fftLength);
    } else {
        /* Shift pointer of twiddle factor to the end, more twiddle factors first for better memory alignement */
        twiddle += MC_TWIDDLE_LENGTH(pow2);
        step = 16;
        twiddle -= MC_TWIDDLE_STAGE_SIZE(step);
        st_ifft_rad4_mono_depth1_neon(re, im, fftLength);
        st_ifft_dit_rad4_mono_depth2_neon(re, im, twiddle, fftLength);
    }
    do {
        step <<= 2u;
        twiddle -= MC_TWIDDLE_STAGE_SIZE(step);
        st_ifft_dit_rad4_mono_loop_neon(re, im, twiddle, fftLength, step);
    } while (step != fftLength);
}


