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

void mc_fftr_unpack_dual_to_perm(float * restrict perm0_re, float * restrict perm0_im,
                                 float * restrict perm1_re, float * restrict perm1_im,
                                 const float * restrict dualRe, const float * restrict dualIm, 
                                 uint32_t length)
{
    MC_NULLPTR_ASSERT(perm0_re);
    MC_NULLPTR_ASSERT(perm0_im);
    MC_NULLPTR_ASSERT(perm1_re);
    MC_NULLPTR_ASSERT(perm1_im);
    MC_NULLPTR_ASSERT(dualRe);
    MC_NULLPTR_ASSERT(dualIm);
    MC_ASSERT(MC_MAX_FFT_LENGTH >= length);

    uint32_t iterations = (length>>1u);

    /* Nyquist & DC */
    *perm0_re = *dualRe;
    *perm0_im = dualRe[iterations];
    *perm1_re = *dualIm;
    *perm1_im = dualIm[iterations];

    for (uint32_t i = 1u; i < iterations; ++i) {
        const float * restrict reFwd = &dualRe[i];
        const float * restrict imFwd = &dualIm[i];
        const float * restrict reRev = &dualRe[length-i];
        const float * restrict imRev = &dualIm[length-i];

        perm0_re[i] = 0.5f*(dualRe[i] + dualRe[length-i]);
        perm0_im[i] = 0.5f*(dualIm[i] - dualIm[length-i]);
        perm1_re[i] = 0.5f*(dualIm[i] + dualIm[length-i]);
        perm1_im[i] = 0.5f*(dualRe[length-i] - dualRe[i]);
    }
}

void mc_fft_norm(float * restrict re, float * restrict im, uint32_t length)
{
    MC_NULLPTR_ASSERT(re);
    MC_NULLPTR_ASSERT(im);
    MC_ASSERT(MC_MAX_FFT_LENGTH >= length);

    float norm_coeff = 1.f / (float)length;

    for (uint32_t i = 0; i < length; ++i) {
        re[i] *= norm_coeff;
        im[i] *= norm_coeff;
    }
}

#include <math.h>

void mc_test_add_sinwave(float *output, uint32_t length, float gain, float freq, float fs) {
    float step = freq/fs;
    for (uint32_t i = 0; i < length; ++i) {
        output[i] += gain * sinf(2.f * MC_PI * (float)i * step);
    }
}

float mc_test_mean_error(const float *v0, const float *v1, uint32_t length) {
    float res = 0.f;
    for (uint32_t i = 0; i < length; ++i) {
        res += fabsf(*v0++ - *v1++);
    }
    return res / (float)length;
}

