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

#include "mcfft_generic.h"
#define MC_FFT_DIRECTION fft
#define MC_INVERSE_FFT (0)
#include "mcfft_rad4_template.c"
#undef MC_FFT_DIRECTION
#undef MC_INVERSE_FFT

#define MC_FFT_DIRECTION ifft
#define MC_INVERSE_FFT (1u)
#include "mcfft_rad4_template.c"

void mc_shuffle_mono_g(float * restrict re, float * restrict im, float * restrict buffer, 
                       const uint16_t * restrict digitRev,  uint32_t length) {
    float * restrict re_tmp = buffer;
    float * restrict im_tmp = &buffer[length];
    for (uint32_t i = 0; i < length; ++i) {
        re_tmp[i] = re[digitRev[i]];
        im_tmp[i] = im[digitRev[i]];
    }
    memcpy(re, re_tmp, length*sizeof(float));
    memcpy(im, im_tmp, length*sizeof(float));
}

void st_rad2_mono_depth1_g(float * restrict re, float * restrict im, uint32_t fftLength) {
    for (uint32_t stepIdx = 0; stepIdx < fftLength; stepIdx += 2u) {
        float accRe = re[stepIdx+1u];
        float accIm = im[stepIdx+1u];
        re[stepIdx+1u] = re[stepIdx] - accRe;
        im[stepIdx+1u] = im[stepIdx] - accIm;
        re[stepIdx] += accRe;
        im[stepIdx] += accIm;
    }
}

void mc_fft_dif_mono_core_g(float * restrict re, float * restrict im, const float * restrict twiddle, uint32_t pow2) {
    const uint32_t fftLength = 1u<<(pow2);
    uint32_t step = fftLength;
    do {
        st_fft_dif_rad4_mono_loop_g(re, im, twiddle, fftLength, step);
        twiddle += MC_TWIDDLE_STAGE_SIZE(step);
        step >>= 2u;
    } while (step > 16u);
    if (pow2 % 2u) {
        st_fft_dif_rad4_mono_depth2_odd_g(re, im, twiddle, fftLength);
        st_rad2_mono_depth1_g(re, im, fftLength);
    } else {
        st_fft_dif_rad4_mono_depth2_g(re, im, twiddle, fftLength);
        st_fft_rad4_mono_depth1_g(re, im, fftLength);
    }
}

void mc_ifft_dif_mono_core_g(float * restrict re, float * restrict im, const float * restrict twiddle, uint32_t pow2) {
    const uint32_t fftLength = 1u<<(pow2);
    uint32_t step = fftLength;
    do {
        st_ifft_dif_rad4_mono_loop_g(re, im, twiddle, fftLength, step);
        twiddle += MC_TWIDDLE_STAGE_SIZE(step);
        step >>= 2u;
    } while (step > 16u);
    if (pow2 % 2u) {
        st_ifft_dif_rad4_mono_depth2_odd_g(re, im, twiddle, fftLength);
        st_rad2_mono_depth1_g(re, im, fftLength);
    } else {
        st_ifft_dif_rad4_mono_depth2_g(re, im, twiddle, fftLength);
        st_ifft_rad4_mono_depth1_g(re, im, fftLength);
    }
}

void mc_fft_dit_mono_core_g(float * restrict re, float * restrict im, const float * restrict twiddle, uint32_t pow2) {
    const uint32_t fftLength = 1u<<(pow2);
    uint32_t step = 0;
    if (pow2 % 2u) {
        /* Shift pointer of twiddle factor to the end, more twiddle factors first for better memory alignement */
        twiddle += MC_TWIDDLE_LENGTH(pow2);
        step = 8u;
        twiddle -= MC_TWIDDLE_STAGE_SIZE(step);
        st_rad2_mono_depth1_g(re, im, fftLength);
        st_fft_dit_rad4_mono_depth2_odd_g(re, im, twiddle, fftLength);
    } else {
        /* Shift pointer of twiddle factor to the end, more twiddle factors first for better memory alignement */
        twiddle += MC_TWIDDLE_LENGTH(pow2);
        step = 16;
        twiddle -= MC_TWIDDLE_STAGE_SIZE(step);
        st_fft_rad4_mono_depth1_g(re, im, fftLength);
        st_fft_dit_rad4_mono_depth2_g(re, im, twiddle, fftLength);
    }
    do {
        step <<= 2u;
        twiddle -= MC_TWIDDLE_STAGE_SIZE(step);
        st_fft_dit_rad4_mono_loop_g(re, im, twiddle, fftLength, step);
    } while (step != fftLength);
}

void mc_ifft_dit_mono_core_g(float * restrict re, float * restrict im, const float * restrict twiddle, uint32_t pow2) {
    const uint32_t fftLength = 1u<<(pow2);
    uint32_t step = 0;
    if (pow2 % 2u) {
        /* Shift pointer of twiddle factor to the end, more twiddle factors first for better memory alignement */
        twiddle += MC_TWIDDLE_LENGTH(pow2);
        step = 8u;
        twiddle -= MC_TWIDDLE_STAGE_SIZE(step);
        st_rad2_mono_depth1_g(re, im, fftLength);
        st_ifft_dit_rad4_mono_depth2_odd_g(re, im, twiddle, fftLength);
    } else {
        /* Shift pointer of twiddle factor to the end, more twiddle factors first for better memory alignement */
        twiddle += MC_TWIDDLE_LENGTH(pow2);
        step = 16;
        twiddle -= MC_TWIDDLE_STAGE_SIZE(step);
        st_ifft_rad4_mono_depth1_g(re, im, fftLength);
        st_ifft_dit_rad4_mono_depth2_g(re, im, twiddle, fftLength);
    }
    do {
        step <<= 2u;
        twiddle -= MC_TWIDDLE_STAGE_SIZE(step);
        st_ifft_dit_rad4_mono_loop_g(re, im, twiddle, fftLength, step);
    } while (step != fftLength);
}

#include <math.h>

uint32_t mc_fft_rad4_get_twiddle_stage_g(float * restrict out, uint32_t step) {
    double phi = -6.28318530718/((double)step);
    uint32_t res = 0;
    if (8u == step) { /**< Depth 2: It is case when power of 2 is odd */
        /** Skip first values with 0 (always twiddle == 1+j0 => doesn't have any effect) */
        *out++ = (float)cos(phi);
        *out++ = (float)sin(phi);
        *out++ = (float)cos(2.0*phi);
        *out++ = (float)sin(2.0*phi);
        *out++ = (float)cos(3.0*phi);
        *out++ = (float)sin(3.0*phi);
        res = 6u;
    } else if (16u == step) { /**< Depth 2: case when power of 2 is even => power of 4 */
        /** Don't skip first values to align it for SIMD */
        for (uint32_t k = 0; k < 4u; ++k) {
            *out++ = (float)cos(phi*(double)k);
        }
        for (uint32_t k = 0; k < 4u; ++k) {
            *out++ = (float)sin(phi*(double)k);
        }
        for (uint32_t k = 0; k < 4u; ++k) {
            *out++ = (float)cos(2.0*phi*(double)k);
        }
        for (uint32_t k = 0; k < 4u; ++k) {
            *out++ = (float)sin(2.0*phi*(double)k);
        }
        for (uint32_t k = 0; k < 4u; ++k) {
            *out++ = (float)cos(3.0*phi*(double)k);
        }
        for (uint32_t k = 0; k < 4u; ++k) {
            *out++ = (float)sin(3.0*phi*(double)k);
        }
        res = 24u;
    } else {
        for (uint32_t i = 0; i < (step>>2u); i += 8u) {
            for (uint32_t k = 0; k < 8u; ++k) {
                *out++ = (float)cos(phi*(double)(i+k));
            }
            for (uint32_t k = 0; k < 8u; ++k) {
                *out++ = (float)sin(phi*(double)(i+k));
            }
            for (uint32_t k = 0; k < 8u; ++k) {
                *out++ = (float)cos(2.0*phi*(double)(i+k));
            }
            for (uint32_t k = 0; k < 8u; ++k) {
                *out++ = (float)sin(2.0*phi*(double)(i+k));
            }
            for (uint32_t k = 0; k < 8u; ++k) {
                *out++ = (float)cos(3.0*phi*(double)(i+k));
            }
            for (uint32_t k = 0; k < 8u; ++k) {
                *out++ = (float)sin(3.0*phi*(double)(i+k));
            }
        }
        res = 6u*(step>>2u);
    }

    return res;
}

