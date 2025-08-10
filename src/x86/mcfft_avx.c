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

#include "mcfft_avx.h"
#include "generic/mcfft_generic.h"
#define MC_FFT_DIRECTION fft
#define MC_INVERSE_FFT (0)
#include "mcfft_rad4_template_avx.c"
#undef MC_FFT_DIRECTION
#undef MC_INVERSE_FFT

#define MC_FFT_DIRECTION ifft
#define MC_INVERSE_FFT (1u)
#include "mcfft_rad4_template_avx.c"

void mc_shuffle_mono_avx(float * restrict re, float * restrict im, float * restrict buffer, 
                         const uint16_t * restrict digitRev,  uint32_t length) {
    float * restrict tmp_re = buffer;
    float * restrict tmp_im = &buffer[length];
    for (uint32_t i = 0; i < length; i += 8u) {
        __m128i indices_u16 = _mm_loadu_si128((const __m128i *)&digitRev[i]);
        __m256i indices = _mm256_cvtepu16_epi32(indices_u16);
        __m256 re_vals = _mm256_i32gather_ps(re, indices, 4);
        __m256 im_vals = _mm256_i32gather_ps(im, indices, 4);
        _mm256_storeu_ps(&tmp_re[i], re_vals);
        _mm256_storeu_ps(&tmp_im[i], im_vals);
    }
    memcpy(re, tmp_re, sizeof(re[0])*length);
    memcpy(im, tmp_im, sizeof(im[0])*length);
}

static void st_rad2_mono_depth1_avx(float *re, float *im, uint32_t fftLength) {
    for (uint32_t stepIdx = 0; stepIdx < fftLength; stepIdx += 8u) {
        __m128 re0_v = _mm_loadu_ps(re);
        __m128 re1_v = _mm_loadu_ps(re+4u);
        __m128 im0_v = _mm_loadu_ps(im);
        __m128 im1_v = _mm_loadu_ps(im+4u);
        __m128 tRe_v = _mm_shuffle_ps(re0_v, re1_v, _MM_SHUFFLE(2, 0, 2, 0));
        __m128 bRe_v = _mm_shuffle_ps(re0_v, re1_v, _MM_SHUFFLE(3, 1, 3, 1));
        __m128 tIm_v = _mm_shuffle_ps(im0_v, im1_v, _MM_SHUFFLE(2, 0, 2, 0));
        __m128 bIm_v = _mm_shuffle_ps(im0_v, im1_v, _MM_SHUFFLE(3, 1, 3, 1));
        
        __m128 sbRe_v = _mm_sub_ps(tRe_v, bRe_v);
        __m128 sbIm_v = _mm_sub_ps(tIm_v, bIm_v);
        tRe_v = _mm_add_ps(tRe_v, bRe_v);
        tIm_v = _mm_add_ps(tIm_v, bIm_v);
        re0_v = _mm_unpacklo_ps(tRe_v, sbRe_v);
        re1_v = _mm_unpackhi_ps(tRe_v, sbRe_v);
        _mm_storeu_ps(re, re0_v);
        _mm_storeu_ps(re+4u, re1_v);
        im0_v = _mm_unpacklo_ps(tIm_v, sbIm_v);
        im1_v = _mm_unpackhi_ps(tIm_v, sbIm_v);
        _mm_storeu_ps(im, im0_v);
        _mm_storeu_ps(im+4u, im1_v);
        re += 8u;
        im += 8u;
    }
}

void mc_fft_dif_mono_core_avx(float * restrict re, float * restrict im, const float * restrict twiddle, uint32_t pow2) {
    const uint32_t fftLength = 1u<<(pow2);
    uint32_t step = fftLength;
    do {
        st_fft_dif_rad4_mono_loop_avx(re, im, twiddle, fftLength, step);
        twiddle += MC_TWIDDLE_STAGE_SIZE(step);
        step >>= 2u;
    } while (step > 16u);
    if (pow2 % 2u) {
        st_fft_dif_rad4_mono_depth2_odd_avx(re, im, twiddle, fftLength);
        st_rad2_mono_depth1_avx(re, im, fftLength);
    } else {
        st_fft_dif_rad4_mono_depth2_avx(re, im, twiddle, fftLength);
        st_fft_rad4_mono_depth1_avx(re, im, fftLength);
    }
}

void mc_ifft_dif_mono_core_avx(float * restrict re, float * restrict im, const float * restrict twiddle, uint32_t pow2) {
    const uint32_t fftLength = 1u<<(pow2);
    uint32_t step = fftLength;
    do {
        st_ifft_dif_rad4_mono_loop_avx(re, im, twiddle, fftLength, step);
        twiddle += MC_TWIDDLE_STAGE_SIZE(step);
        step >>= 2u;
    } while (step > 16u);
    if (pow2 % 2u) {
        st_ifft_dif_rad4_mono_depth2_odd_avx(re, im, twiddle, fftLength);
        st_rad2_mono_depth1_avx(re, im, fftLength);
    } else {
        st_ifft_dif_rad4_mono_depth2_avx(re, im, twiddle, fftLength);
        st_ifft_rad4_mono_depth1_avx(re, im, fftLength);
    }
}

void mc_fft_dit_mono_core_avx(float * restrict re, float * restrict im, const float * restrict twiddle, uint32_t pow2) {
    const uint32_t fftLength = 1u<<(pow2);
    uint32_t step = 0;
    if (pow2 % 2u) {
        /* Shift pointer of twiddle factor to the end, more twiddle factors first for better memory alignement */
        twiddle += MC_TWIDDLE_LENGTH(pow2);
        step = 8u;
        twiddle -= MC_TWIDDLE_STAGE_SIZE(step);
        st_rad2_mono_depth1_avx(re, im, fftLength);
        st_fft_dit_rad4_mono_depth2_odd_avx(re, im, twiddle, fftLength);
    } else {
        /* Shift pointer of twiddle factor to the end, more twiddle factors first for better memory alignement */
        twiddle += MC_TWIDDLE_LENGTH(pow2);
        step = 16;
        twiddle -= MC_TWIDDLE_STAGE_SIZE(step);
        st_fft_rad4_mono_depth1_avx(re, im, fftLength);
        st_fft_dit_rad4_mono_depth2_avx(re, im, twiddle, fftLength);
    }
    do {
        step <<= 2u;
        twiddle -= MC_TWIDDLE_STAGE_SIZE(step);
        st_fft_dit_rad4_mono_loop_avx(re, im, twiddle, fftLength, step);
    } while (step != fftLength);
}

void mc_ifft_dit_mono_core_avx(float * restrict re, float * restrict im, const float * restrict twiddle, uint32_t pow2) {
    const uint32_t fftLength = 1u<<(pow2);
    uint32_t step = 0;
    if (pow2 % 2u) {
        /* Shift pointer of twiddle factor to the end, more twiddle factors first for better memory alignement */
        twiddle += MC_TWIDDLE_LENGTH(pow2);
        step = 8u;
        twiddle -= MC_TWIDDLE_STAGE_SIZE(step);
        st_rad2_mono_depth1_avx(re, im, fftLength);
        st_ifft_dit_rad4_mono_depth2_odd_avx(re, im, twiddle, fftLength);
    } else {
        /* Shift pointer of twiddle factor to the end, more twiddle factors first for better memory alignement */
        twiddle += MC_TWIDDLE_LENGTH(pow2);
        step = 16;
        twiddle -= MC_TWIDDLE_STAGE_SIZE(step);
        st_ifft_rad4_mono_depth1_avx(re, im, fftLength);
        st_ifft_dit_rad4_mono_depth2_avx(re, im, twiddle, fftLength);
    }
    do {
        step <<= 2u;
        twiddle -= MC_TWIDDLE_STAGE_SIZE(step);
        st_ifft_dit_rad4_mono_loop_avx(re, im, twiddle, fftLength, step);
    } while (step != fftLength);
}


