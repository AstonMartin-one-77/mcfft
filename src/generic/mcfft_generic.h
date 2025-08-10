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


#ifndef MC_FFT_GENERIC_H
#define MC_FFT_GENERIC_H

#ifdef __cplusplus
extern "C" {
#endif

#include "mcfft.h"

void mc_shuffle_mono_g(float * restrict re, float * restrict im, float * restrict buffer, 
                       const uint16_t * restrict digitRev,  uint32_t length);
void mc_fft_dif_mono_core_g(float * restrict re, float * restrict im, const float * restrict twiddle, uint32_t pow2);
void mc_ifft_dif_mono_core_g(float * restrict re, float * restrict im, const float * restrict twiddle, uint32_t pow2);
void mc_fft_dit_mono_core_g(float * restrict re, float * restrict im, const float * restrict twiddle, uint32_t pow2);
void mc_ifft_dit_mono_core_g(float * restrict re, float * restrict im, const float * restrict twiddle, uint32_t pow2);
uint32_t mc_fft_rad4_get_twiddle_stage_g(float * restrict out, uint32_t step);

#ifdef __cplusplus
}
#endif

#endif /* MC_FFT_GENERIC_H */
