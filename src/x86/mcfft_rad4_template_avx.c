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

#include <immintrin.h>
#include "utils.h"

static void MC_FUNC_TEMPLATE(MC_FFT_DIRECTION, rad4_mono_depth1, avx) (float * restrict re, float * restrict im, uint32_t fftLength) {

    for (uint32_t stepIdx = 0; stepIdx < fftLength; stepIdx += 16u) {
        __m128 re_v0 = _mm_loadu_ps(re);
        __m128 re_v1 = _mm_loadu_ps(re+4u);
        __m128 re_v2 = _mm_loadu_ps(re+8u);
        __m128 re_v3 = _mm_loadu_ps(re+12u);
        __m128 im_v0 = _mm_loadu_ps(im);
        __m128 im_v1 = _mm_loadu_ps(im+4u);
        __m128 im_v2 = _mm_loadu_ps(im+8u);
        __m128 im_v3 = _mm_loadu_ps(im+12u);
        _MM_TRANSPOSE4_PS(re_v0, re_v1, re_v2, re_v3);
        _MM_TRANSPOSE4_PS(im_v0, im_v1, im_v2, im_v3);

        __m128 t0_vRe = _mm_add_ps(re_v0, re_v2);
        __m128 t0_vIm = _mm_add_ps(im_v0, im_v2);
        __m128 t1_vRe = _mm_sub_ps(re_v0, re_v2);
        __m128 t1_vIm = _mm_sub_ps(im_v0, im_v2);
        __m128 t2_vRe = _mm_add_ps(re_v1, re_v3);
        __m128 t2_vIm = _mm_add_ps(im_v1, im_v3);
        __m128 t3_vRe = _mm_sub_ps(im_v1, im_v3);
        __m128 t3_vIm = _mm_sub_ps(re_v3, re_v1);

        re_v0 = _mm_add_ps(t0_vRe, t2_vRe);
#if MC_INVERSE_FFT
        re_v1 = _mm_sub_ps(t1_vRe, t3_vRe);
        re_v3 = _mm_add_ps(t1_vRe, t3_vRe);
        im_v1 = _mm_sub_ps(t1_vIm, t3_vIm);
        im_v3 = _mm_add_ps(t1_vIm, t3_vIm);
#else
        re_v1 = _mm_add_ps(t1_vRe, t3_vRe);
        re_v3 = _mm_sub_ps(t1_vRe, t3_vRe);
        im_v1 = _mm_add_ps(t1_vIm, t3_vIm);
        im_v3 = _mm_sub_ps(t1_vIm, t3_vIm);
#endif
        re_v2 = _mm_sub_ps(t0_vRe, t2_vRe);
        im_v0 = _mm_add_ps(t0_vIm, t2_vIm);
        im_v2 = _mm_sub_ps(t0_vIm, t2_vIm);
        _MM_TRANSPOSE4_PS(re_v0, re_v1, re_v2, re_v3);
        _MM_TRANSPOSE4_PS(im_v0, im_v1, im_v2, im_v3);
        _mm_storeu_ps(re, re_v0);
        _mm_storeu_ps(re+4u, re_v1);
        _mm_storeu_ps(re+8u, re_v2);
        _mm_storeu_ps(re+12u, re_v3);
        _mm_storeu_ps(im, im_v0);
        _mm_storeu_ps(im+4u, im_v1);
        _mm_storeu_ps(im+8u, im_v2);
        _mm_storeu_ps(im+12u, im_v3);
        re += 16u;
        im += 16u;
    }
}

static void MC_FUNC_TEMPLATE(MC_FFT_DIRECTION, dit_rad4_mono_depth2_odd, avx) (float * restrict re, float * restrict im, const float * restrict twiddle, uint32_t fftLength) {
    __m128 twdB_re = _mm_set_ps(twiddle[0], 1.f, twiddle[0], 1.f);
    __m128 twdB_im = _mm_set_ps(twiddle[1], 0.f, twiddle[1], 0.f);
    __m128 twdC_re = _mm_set_ps(twiddle[2], 1.f, twiddle[2], 1.f);
    __m128 twdC_im = _mm_set_ps(twiddle[3], 0.f, twiddle[3], 0.f);
    __m128 twdD_re = _mm_set_ps(twiddle[4], 1.f, twiddle[4], 1.f);
    __m128 twdD_im = _mm_set_ps(twiddle[5], 0.f, twiddle[5], 0.f);

    for (uint32_t stepIdx = 0; stepIdx < fftLength; stepIdx += 16u) {
        __m128 re_v0 = _mm_loadu_ps(re);     /**< aRe0, aRe1, bRe0, bRe1 */
        __m128 re_v1 = _mm_loadu_ps(re+4u);  /**< cRe0, cRe1, dRe0, dRe1 */
        __m128 re_v2 = _mm_loadu_ps(re+8u);  /**< aRe2, aRe3, bRe2, bRe3 */
        __m128 re_v3 = _mm_loadu_ps(re+12u); /**< cRe2, cRe3, dRe2, dRe3 */
        __m128 im_v0 = _mm_loadu_ps(im);
        __m128 im_v1 = _mm_loadu_ps(im+4u);
        __m128 im_v2 = _mm_loadu_ps(im+8u);
        __m128 im_v3 = _mm_loadu_ps(im+12u);

        __m128 aRe_v = _mm_shuffle_ps(re_v0, re_v2, _MM_SHUFFLE(1, 0, 1, 0));
        __m128 bRe_v = _mm_shuffle_ps(re_v0, re_v2, _MM_SHUFFLE(3, 2, 3, 2));
        __m128 cRe_v = _mm_shuffle_ps(re_v1, re_v3, _MM_SHUFFLE(1, 0, 1, 0));
        __m128 dRe_v = _mm_shuffle_ps(re_v1, re_v3, _MM_SHUFFLE(3, 2, 3, 2));

        __m128 aIm_v = _mm_shuffle_ps(im_v0, im_v2, _MM_SHUFFLE(1, 0, 1, 0));
        __m128 bIm_v = _mm_shuffle_ps(im_v0, im_v2, _MM_SHUFFLE(3, 2, 3, 2));
        __m128 cIm_v = _mm_shuffle_ps(im_v1, im_v3, _MM_SHUFFLE(1, 0, 1, 0));
        __m128 dIm_v = _mm_shuffle_ps(im_v1, im_v3, _MM_SHUFFLE(3, 2, 3, 2));

        __m128 acc0Re_v = _mm_mul_ps(bRe_v, twdB_re);
        __m128 acc0Im_v = _mm_mul_ps(bIm_v, twdB_re);
        __m128 acc1Re_v = _mm_mul_ps(cRe_v, twdC_re);
        __m128 acc1Im_v = _mm_mul_ps(cIm_v, twdC_re);
        __m128 acc2Re_v = _mm_mul_ps(dRe_v, twdD_re);
        __m128 acc2Im_v = _mm_mul_ps(dIm_v, twdD_re);
#if MC_INVERSE_FFT
        acc0Re_v = _mm_fmadd_ps(bIm_v, twdB_im, acc0Re_v);
        acc0Im_v = _mm_fnmadd_ps(bRe_v, twdB_im, acc0Im_v);
        acc1Re_v = _mm_fmadd_ps(cIm_v, twdC_im, acc1Re_v);
        acc1Im_v = _mm_fnmadd_ps(cRe_v, twdC_im, acc1Im_v);
        acc2Re_v = _mm_fmadd_ps(dIm_v, twdD_im, acc2Re_v);
        acc2Im_v = _mm_fnmadd_ps(dRe_v, twdD_im, acc2Im_v);
#else
        acc0Re_v = _mm_fnmadd_ps(bIm_v, twdB_im, acc0Re_v);
        acc0Im_v = _mm_fmadd_ps(bRe_v, twdB_im, acc0Im_v);
        acc1Re_v = _mm_fnmadd_ps(cIm_v, twdC_im, acc1Re_v);
        acc1Im_v = _mm_fmadd_ps(cRe_v, twdC_im, acc1Im_v);
        acc2Re_v = _mm_fnmadd_ps(dIm_v, twdD_im, acc2Re_v);
        acc2Im_v = _mm_fmadd_ps(dRe_v, twdD_im, acc2Im_v);
#endif
        re_v0 = _mm_add_ps(aRe_v, acc1Re_v);
        im_v0 = _mm_add_ps(aIm_v, acc1Im_v);
        re_v1 = _mm_sub_ps(aRe_v, acc1Re_v);
        im_v1 = _mm_sub_ps(aIm_v, acc1Im_v);
        re_v2 = _mm_add_ps(acc0Re_v, acc2Re_v);
        im_v2 = _mm_add_ps(acc0Im_v, acc2Im_v);
        re_v3 = _mm_sub_ps(acc0Im_v, acc2Im_v); // conj
        im_v3 = _mm_sub_ps(acc2Re_v, acc0Re_v); // conj

        aRe_v = _mm_add_ps(re_v0, re_v2);
        aIm_v = _mm_add_ps(im_v0, im_v2);
#if MC_INVERSE_FFT
        bRe_v = _mm_sub_ps(re_v1, re_v3);
        bIm_v = _mm_sub_ps(im_v1, im_v3);
#else
        bRe_v = _mm_add_ps(re_v1, re_v3);
        bIm_v = _mm_add_ps(im_v1, im_v3);
#endif
        cRe_v = _mm_sub_ps(re_v0, re_v2);
        cIm_v = _mm_sub_ps(im_v0, im_v2);
#if MC_INVERSE_FFT
        dRe_v = _mm_add_ps(re_v1, re_v3);
        dIm_v = _mm_add_ps(im_v1, im_v3);
#else
        dRe_v = _mm_sub_ps(re_v1, re_v3);
        dIm_v = _mm_sub_ps(im_v1, im_v3);
#endif
        re_v0 = _mm_movelh_ps(aRe_v, bRe_v);
        re_v1 = _mm_movelh_ps(cRe_v, dRe_v);
        re_v2 = _mm_movehl_ps(bRe_v, aRe_v);
        re_v3 = _mm_movehl_ps(dRe_v, cRe_v);
        im_v0 = _mm_movelh_ps(aIm_v, bIm_v);
        im_v1 = _mm_movelh_ps(cIm_v, dIm_v);
        im_v2 = _mm_movehl_ps(bIm_v, aIm_v);
        im_v3 = _mm_movehl_ps(dIm_v, cIm_v);

        _mm_storeu_ps(re, re_v0);
        _mm_storeu_ps(im, im_v0);
        _mm_storeu_ps(re+4u, re_v1);
        _mm_storeu_ps(im+4u, im_v1);
        _mm_storeu_ps(re+8u, re_v2);
        _mm_storeu_ps(im+8u, im_v2);
        _mm_storeu_ps(re+12u, re_v3);
        _mm_storeu_ps(im+12u, im_v3);
        re += 16u;
        im += 16u;
    }
}

static void MC_FUNC_TEMPLATE(MC_FFT_DIRECTION, dit_rad4_mono_depth2, avx)(float * restrict re, float * restrict im, const float * restrict twiddle, uint32_t fftLength) {
    __m128 twdB_vRe = _mm_loadu_ps(twiddle);
    __m128 twdB_vIm = _mm_loadu_ps(twiddle+4u);
    __m128 twdC_vRe = _mm_loadu_ps(twiddle+8u);
    __m128 twdC_vIm = _mm_loadu_ps(twiddle+12u);
    __m128 twdD_vRe = _mm_loadu_ps(twiddle+16u);
    __m128 twdD_vIm = _mm_loadu_ps(twiddle+20u);
    
    for (uint32_t stepIdx = 0; stepIdx < fftLength; stepIdx += 16u) {
        __m128 re_v0 = _mm_loadu_ps(re);
        __m128 re_v1 = _mm_loadu_ps(re+4u);
        __m128 re_v2 = _mm_loadu_ps(re+8u);
        __m128 re_v3 = _mm_loadu_ps(re+12u);
        __m128 im_v0 = _mm_loadu_ps(im);
        __m128 im_v1 = _mm_loadu_ps(im+4u);
        __m128 im_v2 = _mm_loadu_ps(im+8u);
        __m128 im_v3 = _mm_loadu_ps(im+12u);

        __m128 acc0_vRe = _mm_mul_ps(re_v1, twdB_vRe);
        __m128 acc0_vIm = _mm_mul_ps(im_v1, twdB_vRe);
#if MC_INVERSE_FFT
        acc0_vRe = _mm_fmadd_ps(im_v1, twdB_vIm, acc0_vRe);
        acc0_vIm = _mm_fnmadd_ps(re_v1, twdB_vIm, acc0_vIm);
#else
        acc0_vRe = _mm_fnmadd_ps(im_v1, twdB_vIm, acc0_vRe);
        acc0_vIm = _mm_fmadd_ps(re_v1, twdB_vIm, acc0_vIm);
#endif
        __m128 acc1_vRe = _mm_mul_ps(re_v2, twdC_vRe);
        __m128 acc1_vIm = _mm_mul_ps(im_v2, twdC_vRe);
#if MC_INVERSE_FFT
        acc1_vRe = _mm_fmadd_ps(im_v2, twdC_vIm, acc1_vRe);
        acc1_vIm = _mm_fnmadd_ps(re_v2, twdC_vIm, acc1_vIm);
#else
        acc1_vRe = _mm_fnmadd_ps(im_v2, twdC_vIm, acc1_vRe);
        acc1_vIm = _mm_fmadd_ps(re_v2, twdC_vIm, acc1_vIm);
#endif
        __m128 acc2_vRe = _mm_mul_ps(re_v3, twdD_vRe);
        __m128 acc2_vIm = _mm_mul_ps(im_v3, twdD_vRe);
#if MC_INVERSE_FFT
        acc2_vRe = _mm_fmadd_ps(im_v3, twdD_vIm, acc2_vRe);
        acc2_vIm = _mm_fnmadd_ps(re_v3, twdD_vIm, acc2_vIm);
#else
        acc2_vRe = _mm_fnmadd_ps(im_v3, twdD_vIm, acc2_vRe);
        acc2_vIm = _mm_fmadd_ps(re_v3, twdD_vIm, acc2_vIm);
#endif

        __m128 t0_vRe = _mm_add_ps(re_v0, acc1_vRe);
        __m128 t0_vIm = _mm_add_ps(im_v0, acc1_vIm);
        __m128 t1_vRe = _mm_sub_ps(re_v0, acc1_vRe);
        __m128 t1_vIm = _mm_sub_ps(im_v0, acc1_vIm);
        __m128 t2_vRe = _mm_add_ps(acc0_vRe, acc2_vRe);
        __m128 t2_vIm = _mm_add_ps(acc0_vIm, acc2_vIm);
        __m128 t3_vRe = _mm_sub_ps(acc0_vIm, acc2_vIm);
        __m128 t3_vIm = _mm_sub_ps(acc2_vRe, acc0_vRe);

        re_v0 = _mm_add_ps(t0_vRe, t2_vRe);
#if MC_INVERSE_FFT
        re_v1 = _mm_sub_ps(t1_vRe, t3_vRe);
        re_v3 = _mm_add_ps(t1_vRe, t3_vRe);
        im_v1 = _mm_sub_ps(t1_vIm, t3_vIm);
        im_v3 = _mm_add_ps(t1_vIm, t3_vIm);
#else
        re_v1 = _mm_add_ps(t1_vRe, t3_vRe);
        re_v3 = _mm_sub_ps(t1_vRe, t3_vRe);
        im_v1 = _mm_add_ps(t1_vIm, t3_vIm);
        im_v3 = _mm_sub_ps(t1_vIm, t3_vIm);
#endif
        re_v2 = _mm_sub_ps(t0_vRe, t2_vRe);
        im_v0 = _mm_add_ps(t0_vIm, t2_vIm);
        im_v2 = _mm_sub_ps(t0_vIm, t2_vIm);

        _mm_storeu_ps(re, re_v0);
        _mm_storeu_ps(re+4u, re_v1);
        _mm_storeu_ps(re+8u, re_v2);
        _mm_storeu_ps(re+12u, re_v3);
        _mm_storeu_ps(im, im_v0);
        _mm_storeu_ps(im+4u, im_v1);
        _mm_storeu_ps(im+8u, im_v2);
        _mm_storeu_ps(im+12u, im_v3);

        re += 16u;
        im += 16u;
    }
}

static void MC_FUNC_TEMPLATE(MC_FFT_DIRECTION, dit_rad4_mono_loop, avx)(float *re, float *im, const float * restrict twiddle, uint32_t fftLength, uint32_t step) {
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
        const float *twd = twiddle;
        for (uint32_t i = 0; i < qStep; i += 8u) {
            __m256 twdB_vRe = _mm256_loadu_ps(twd);
            __m256 twdB_vIm = _mm256_loadu_ps(twd+8u);
            __m256 twdC_vRe = _mm256_loadu_ps(twd+16u);
            __m256 twdC_vIm = _mm256_loadu_ps(twd+24u);
            __m256 twdD_vRe = _mm256_loadu_ps(twd+32u);
            __m256 twdD_vIm = _mm256_loadu_ps(twd+40u);

            __m256 re_v0 = _mm256_loadu_ps(aRe);
            __m256 re_v1 = _mm256_loadu_ps(bRe);
            __m256 re_v2 = _mm256_loadu_ps(cRe);
            __m256 re_v3 = _mm256_loadu_ps(dRe);
            __m256 im_v0 = _mm256_loadu_ps(aIm);
            __m256 im_v1 = _mm256_loadu_ps(bIm);
            __m256 im_v2 = _mm256_loadu_ps(cIm);
            __m256 im_v3 = _mm256_loadu_ps(dIm);

            __m256 acc0_vRe = _mm256_mul_ps(re_v1, twdB_vRe);
            __m256 acc0_vIm = _mm256_mul_ps(im_v1, twdB_vRe);
#if MC_INVERSE_FFT
            acc0_vRe = _mm256_fmadd_ps(im_v1, twdB_vIm, acc0_vRe);
            acc0_vIm = _mm256_fnmadd_ps(re_v1, twdB_vIm, acc0_vIm);
#else
            acc0_vRe = _mm256_fnmadd_ps(im_v1, twdB_vIm, acc0_vRe);
            acc0_vIm = _mm256_fmadd_ps(re_v1, twdB_vIm, acc0_vIm);
#endif
            __m256 acc1_vRe = _mm256_mul_ps(re_v2, twdC_vRe);
            __m256 acc1_vIm = _mm256_mul_ps(im_v2, twdC_vRe);
#if MC_INVERSE_FFT
            acc1_vRe = _mm256_fmadd_ps(im_v2, twdC_vIm, acc1_vRe);
            acc1_vIm = _mm256_fnmadd_ps(re_v2, twdC_vIm, acc1_vIm);
#else
            acc1_vRe = _mm256_fnmadd_ps(im_v2, twdC_vIm, acc1_vRe);
            acc1_vIm = _mm256_fmadd_ps(re_v2, twdC_vIm, acc1_vIm);
#endif
            __m256 acc2_vRe = _mm256_mul_ps(re_v3, twdD_vRe);
            __m256 acc2_vIm = _mm256_mul_ps(im_v3, twdD_vRe);
#if MC_INVERSE_FFT
            acc2_vRe = _mm256_fmadd_ps(im_v3, twdD_vIm, acc2_vRe);
            acc2_vIm = _mm256_fnmadd_ps(re_v3, twdD_vIm, acc2_vIm);
#else
            acc2_vRe = _mm256_fnmadd_ps(im_v3, twdD_vIm, acc2_vRe);
            acc2_vIm = _mm256_fmadd_ps(re_v3, twdD_vIm, acc2_vIm);
#endif

            __m256 t0_vRe = _mm256_add_ps(re_v0, acc1_vRe);
            __m256 t0_vIm = _mm256_add_ps(im_v0, acc1_vIm);
            __m256 t1_vRe = _mm256_sub_ps(re_v0, acc1_vRe);
            __m256 t1_vIm = _mm256_sub_ps(im_v0, acc1_vIm);
            __m256 t2_vRe = _mm256_add_ps(acc0_vRe, acc2_vRe);
            __m256 t2_vIm = _mm256_add_ps(acc0_vIm, acc2_vIm);
            __m256 t3_vRe = _mm256_sub_ps(acc0_vIm, acc2_vIm);
            __m256 t3_vIm = _mm256_sub_ps(acc2_vRe, acc0_vRe);

            re_v0 = _mm256_add_ps(t0_vRe, t2_vRe);
#if MC_INVERSE_FFT
            re_v1 = _mm256_sub_ps(t1_vRe, t3_vRe);
            re_v3 = _mm256_add_ps(t1_vRe, t3_vRe);
            im_v1 = _mm256_sub_ps(t1_vIm, t3_vIm);
            im_v3 = _mm256_add_ps(t1_vIm, t3_vIm);
#else
            re_v1 = _mm256_add_ps(t1_vRe, t3_vRe);
            re_v3 = _mm256_sub_ps(t1_vRe, t3_vRe);
            im_v1 = _mm256_add_ps(t1_vIm, t3_vIm);
            im_v3 = _mm256_sub_ps(t1_vIm, t3_vIm);
#endif
            re_v2 = _mm256_sub_ps(t0_vRe, t2_vRe);
            im_v0 = _mm256_add_ps(t0_vIm, t2_vIm);
            im_v2 = _mm256_sub_ps(t0_vIm, t2_vIm);

            _mm256_storeu_ps(aRe, re_v0);
            _mm256_storeu_ps(bRe, re_v1);
            _mm256_storeu_ps(cRe, re_v2);
            _mm256_storeu_ps(dRe, re_v3);
            _mm256_storeu_ps(aIm, im_v0);
            _mm256_storeu_ps(bIm, im_v1);
            _mm256_storeu_ps(cIm, im_v2);
            _mm256_storeu_ps(dIm, im_v3);
            
            aRe += 8u;
            bRe += 8u;
            cRe += 8u;
            dRe += 8u;
            aIm += 8u;
            bIm += 8u;
            cIm += 8u;
            dIm += 8u;
            twd += 48u;
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

static void MC_FUNC_TEMPLATE(MC_FFT_DIRECTION, dif_rad4_mono_depth2_odd, avx) (float * restrict re, float * restrict im, const float * restrict twiddle, uint32_t fftLength) {
    __m128 twdB_re = _mm_set_ps(twiddle[0], 1.f, twiddle[0], 1.f);
    __m128 twdB_im = _mm_set_ps(twiddle[1], 0.f, twiddle[1], 0.f);
    __m128 twdC_re = _mm_set_ps(twiddle[2], 1.f, twiddle[2], 1.f);
    __m128 twdC_im = _mm_set_ps(twiddle[3], 0.f, twiddle[3], 0.f);
    __m128 twdD_re = _mm_set_ps(twiddle[4], 1.f, twiddle[4], 1.f);
    __m128 twdD_im = _mm_set_ps(twiddle[5], 0.f, twiddle[5], 0.f);

    for (uint32_t stepIdx = 0; stepIdx < fftLength; stepIdx += 16u) {
        __m128 re_v0 = _mm_loadu_ps(re);     /**< aRe0, aRe1, bRe0, bRe1 */
        __m128 re_v1 = _mm_loadu_ps(re+4u);  /**< cRe0, cRe1, dRe0, dRe1 */
        __m128 re_v2 = _mm_loadu_ps(re+8u);  /**< aRe2, aRe3, bRe2, bRe3 */
        __m128 re_v3 = _mm_loadu_ps(re+12u); /**< cRe2, cRe3, dRe2, dRe3 */
        __m128 im_v0 = _mm_loadu_ps(im);
        __m128 im_v1 = _mm_loadu_ps(im+4u);
        __m128 im_v2 = _mm_loadu_ps(im+8u);
        __m128 im_v3 = _mm_loadu_ps(im+12u);

        __m128 aRe_v = _mm_shuffle_ps(re_v0, re_v2, _MM_SHUFFLE(1, 0, 1, 0));
        __m128 bRe_v = _mm_shuffle_ps(re_v0, re_v2, _MM_SHUFFLE(3, 2, 3, 2));
        __m128 cRe_v = _mm_shuffle_ps(re_v1, re_v3, _MM_SHUFFLE(1, 0, 1, 0));
        __m128 dRe_v = _mm_shuffle_ps(re_v1, re_v3, _MM_SHUFFLE(3, 2, 3, 2));

        __m128 aIm_v = _mm_shuffle_ps(im_v0, im_v2, _MM_SHUFFLE(1, 0, 1, 0));
        __m128 bIm_v = _mm_shuffle_ps(im_v0, im_v2, _MM_SHUFFLE(3, 2, 3, 2));
        __m128 cIm_v = _mm_shuffle_ps(im_v1, im_v3, _MM_SHUFFLE(1, 0, 1, 0));
        __m128 dIm_v = _mm_shuffle_ps(im_v1, im_v3, _MM_SHUFFLE(3, 2, 3, 2));

        re_v0 = _mm_add_ps(aRe_v, cRe_v);
        im_v0 = _mm_add_ps(aIm_v, cIm_v);
        re_v1 = _mm_sub_ps(aRe_v, cRe_v);
        im_v1 = _mm_sub_ps(aIm_v, cIm_v);
        re_v2 = _mm_add_ps(bRe_v, dRe_v);
        im_v2 = _mm_add_ps(bIm_v, dIm_v);
        re_v3 = _mm_sub_ps(bIm_v, dIm_v); // conj
        im_v3 = _mm_sub_ps(dRe_v, bRe_v); // conj

        aRe_v = _mm_add_ps(re_v0, re_v2);
        aIm_v = _mm_add_ps(im_v0, im_v2);
        cRe_v = _mm_sub_ps(re_v0, re_v2);
        cIm_v = _mm_sub_ps(im_v0, im_v2);
#if MC_INVERSE_FFT
        bRe_v = _mm_sub_ps(re_v1, re_v3);
        bIm_v = _mm_sub_ps(im_v1, im_v3);
        dRe_v = _mm_add_ps(re_v1, re_v3);
        dIm_v = _mm_add_ps(im_v1, im_v3);

        __m128 accRe_v = _mm_mul_ps(bRe_v, twdB_re);
        __m128 accIm_v = _mm_mul_ps(bIm_v, twdB_re);
        accRe_v = _mm_fmadd_ps(bIm_v, twdB_im, accRe_v);
        accIm_v = _mm_fnmadd_ps(bRe_v, twdB_im, accIm_v);
        bRe_v = accRe_v;
        bIm_v = accIm_v;
        accRe_v = _mm_mul_ps(cRe_v, twdC_re);
        accIm_v = _mm_mul_ps(cIm_v, twdC_re);
        accRe_v = _mm_fmadd_ps(cIm_v, twdC_im, accRe_v);
        accIm_v = _mm_fnmadd_ps(cRe_v, twdC_im, accIm_v);
        cRe_v = accRe_v;
        cIm_v = accIm_v;
        accRe_v = _mm_mul_ps(dRe_v, twdD_re);
        accIm_v = _mm_mul_ps(dIm_v, twdD_re);
        accRe_v = _mm_fmadd_ps(dIm_v, twdD_im, accRe_v);
        accIm_v = _mm_fnmadd_ps(dRe_v, twdD_im, accIm_v);
        dRe_v = accRe_v;
        dIm_v = accIm_v;
#else
        bRe_v = _mm_add_ps(re_v1, re_v3);
        bIm_v = _mm_add_ps(im_v1, im_v3);
        dRe_v = _mm_sub_ps(re_v1, re_v3);
        dIm_v = _mm_sub_ps(im_v1, im_v3);

        __m128 accRe_v = _mm_mul_ps(bRe_v, twdB_re);
        __m128 accIm_v = _mm_mul_ps(bIm_v, twdB_re);
        accRe_v = _mm_fnmadd_ps(bIm_v, twdB_im, accRe_v);
        accIm_v = _mm_fmadd_ps(bRe_v, twdB_im, accIm_v);
        bRe_v = accRe_v;
        bIm_v = accIm_v;
        accRe_v = _mm_mul_ps(cRe_v, twdC_re);
        accIm_v = _mm_mul_ps(cIm_v, twdC_re);
        accRe_v = _mm_fnmadd_ps(cIm_v, twdC_im, accRe_v);
        accIm_v = _mm_fmadd_ps(cRe_v, twdC_im, accIm_v);
        cRe_v = accRe_v;
        cIm_v = accIm_v;
        accRe_v = _mm_mul_ps(dRe_v, twdD_re);
        accIm_v = _mm_mul_ps(dIm_v, twdD_re);
        accRe_v = _mm_fnmadd_ps(dIm_v, twdD_im, accRe_v);
        accIm_v = _mm_fmadd_ps(dRe_v, twdD_im, accIm_v);
        dRe_v = accRe_v;
        dIm_v = accIm_v;
#endif
        re_v0 = _mm_movelh_ps(aRe_v, bRe_v);
        re_v1 = _mm_movelh_ps(cRe_v, dRe_v);
        re_v2 = _mm_movehl_ps(bRe_v, aRe_v);
        re_v3 = _mm_movehl_ps(dRe_v, cRe_v);
        im_v0 = _mm_movelh_ps(aIm_v, bIm_v);
        im_v1 = _mm_movelh_ps(cIm_v, dIm_v);
        im_v2 = _mm_movehl_ps(bIm_v, aIm_v);
        im_v3 = _mm_movehl_ps(dIm_v, cIm_v);

        _mm_storeu_ps(re, re_v0);
        _mm_storeu_ps(im, im_v0);
        _mm_storeu_ps(re+4u, re_v1);
        _mm_storeu_ps(im+4u, im_v1);
        _mm_storeu_ps(re+8u, re_v2);
        _mm_storeu_ps(im+8u, im_v2);
        _mm_storeu_ps(re+12u, re_v3);
        _mm_storeu_ps(im+12u, im_v3);
        re += 16u;
        im += 16u;
    }
}

static void MC_FUNC_TEMPLATE(MC_FFT_DIRECTION, dif_rad4_mono_depth2, avx)(float * restrict re, float * restrict im, const float * restrict twiddle, uint32_t fftLength) {
    __m128 twdB_vRe = _mm_loadu_ps(twiddle);
    __m128 twdB_vIm = _mm_loadu_ps(twiddle+4u);
    __m128 twdC_vRe = _mm_loadu_ps(twiddle+8u);
    __m128 twdC_vIm = _mm_loadu_ps(twiddle+12u);
    __m128 twdD_vRe = _mm_loadu_ps(twiddle+16u);
    __m128 twdD_vIm = _mm_loadu_ps(twiddle+20u);
    
    for (uint32_t stepIdx = 0; stepIdx < fftLength; stepIdx += 16u) {
        __m128 re_v0 = _mm_loadu_ps(re);
        __m128 re_v1 = _mm_loadu_ps(re+4u);
        __m128 re_v2 = _mm_loadu_ps(re+8u);
        __m128 re_v3 = _mm_loadu_ps(re+12u);
        __m128 im_v0 = _mm_loadu_ps(im);
        __m128 im_v1 = _mm_loadu_ps(im+4u);
        __m128 im_v2 = _mm_loadu_ps(im+8u);
        __m128 im_v3 = _mm_loadu_ps(im+12u);

        __m128 t0_vRe = _mm_add_ps(re_v0, re_v2);
        __m128 t0_vIm = _mm_add_ps(im_v0, im_v2);
        __m128 t1_vRe = _mm_sub_ps(re_v0, re_v2);
        __m128 t1_vIm = _mm_sub_ps(im_v0, im_v2);
        __m128 t2_vRe = _mm_add_ps(re_v1, re_v3);
        __m128 t2_vIm = _mm_add_ps(im_v1, im_v3);
        __m128 t3_vRe = _mm_sub_ps(im_v1, im_v3);
        __m128 t3_vIm = _mm_sub_ps(re_v3, re_v1);

        re_v0 = _mm_add_ps(t0_vRe, t2_vRe);
        im_v0 = _mm_add_ps(t0_vIm, t2_vIm);
        re_v2 = _mm_sub_ps(t0_vRe, t2_vRe);
        im_v2 = _mm_sub_ps(t0_vIm, t2_vIm);
#if MC_INVERSE_FFT
        re_v1 = _mm_sub_ps(t1_vRe, t3_vRe);
        im_v1 = _mm_sub_ps(t1_vIm, t3_vIm);
        re_v3 = _mm_add_ps(t1_vRe, t3_vRe);
        im_v3 = _mm_add_ps(t1_vIm, t3_vIm);

        t1_vRe = _mm_mul_ps(re_v1, twdB_vRe);
        t1_vIm = _mm_mul_ps(im_v1, twdB_vRe);
        t1_vRe = _mm_fmadd_ps(im_v1, twdB_vIm, t1_vRe);
        t1_vIm = _mm_fnmadd_ps(re_v1, twdB_vIm, t1_vIm);
        t2_vRe = _mm_mul_ps(re_v2, twdC_vRe);
        t2_vIm = _mm_mul_ps(im_v2, twdC_vRe);
        t2_vRe = _mm_fmadd_ps(im_v2, twdC_vIm, t2_vRe);
        t2_vIm = _mm_fnmadd_ps(re_v2, twdC_vIm, t2_vIm);
        t3_vRe = _mm_mul_ps(re_v3, twdD_vRe);
        t3_vIm = _mm_mul_ps(im_v3, twdD_vRe);
        t3_vRe = _mm_fmadd_ps(im_v3, twdD_vIm, t3_vRe);
        t3_vIm = _mm_fnmadd_ps(re_v3, twdD_vIm, t3_vIm);
#else
        re_v1 = _mm_add_ps(t1_vRe, t3_vRe);
        im_v1 = _mm_add_ps(t1_vIm, t3_vIm);
        re_v3 = _mm_sub_ps(t1_vRe, t3_vRe);
        im_v3 = _mm_sub_ps(t1_vIm, t3_vIm);

        t1_vRe = _mm_mul_ps(re_v1, twdB_vRe);
        t1_vIm = _mm_mul_ps(im_v1, twdB_vRe);
        t1_vRe = _mm_fnmadd_ps(im_v1, twdB_vIm, t1_vRe);
        t1_vIm = _mm_fmadd_ps(re_v1, twdB_vIm, t1_vIm);
        t2_vRe = _mm_mul_ps(re_v2, twdC_vRe);
        t2_vIm = _mm_mul_ps(im_v2, twdC_vRe);
        t2_vRe = _mm_fnmadd_ps(im_v2, twdC_vIm, t2_vRe);
        t2_vIm = _mm_fmadd_ps(re_v2, twdC_vIm, t2_vIm);
        t3_vRe = _mm_mul_ps(re_v3, twdD_vRe);
        t3_vIm = _mm_mul_ps(im_v3, twdD_vRe);
        t3_vRe = _mm_fnmadd_ps(im_v3, twdD_vIm, t3_vRe);
        t3_vIm = _mm_fmadd_ps(re_v3, twdD_vIm, t3_vIm);
#endif

        _mm_storeu_ps(re, re_v0);
        _mm_storeu_ps(re+4u, t1_vRe);
        _mm_storeu_ps(re+8u, t2_vRe);
        _mm_storeu_ps(re+12u, t3_vRe);
        _mm_storeu_ps(im, im_v0);
        _mm_storeu_ps(im+4u, t1_vIm);
        _mm_storeu_ps(im+8u, t2_vIm);
        _mm_storeu_ps(im+12u, t3_vIm);

        re += 16u;
        im += 16u;
    }
}

static void MC_FUNC_TEMPLATE(MC_FFT_DIRECTION, dif_rad4_mono_loop, avx)(float *re, float *im, const float * restrict twiddle, uint32_t fftLength, uint32_t step) {
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
        const float *twd = twiddle;
        for (uint32_t i = 0; i < qStep; i += 8u) {
            __m256 twdB_vRe = _mm256_loadu_ps(twd);
            __m256 twdB_vIm = _mm256_loadu_ps(twd+8u);
            __m256 twdC_vRe = _mm256_loadu_ps(twd+16u);
            __m256 twdC_vIm = _mm256_loadu_ps(twd+24u);
            __m256 twdD_vRe = _mm256_loadu_ps(twd+32u);
            __m256 twdD_vIm = _mm256_loadu_ps(twd+40u);

            __m256 re_v0 = _mm256_loadu_ps(aRe);
            __m256 re_v1 = _mm256_loadu_ps(bRe);
            __m256 re_v2 = _mm256_loadu_ps(cRe);
            __m256 re_v3 = _mm256_loadu_ps(dRe);
            __m256 im_v0 = _mm256_loadu_ps(aIm);
            __m256 im_v1 = _mm256_loadu_ps(bIm);
            __m256 im_v2 = _mm256_loadu_ps(cIm);
            __m256 im_v3 = _mm256_loadu_ps(dIm);

            __m256 t0_vRe = _mm256_add_ps(re_v0, re_v2);
            __m256 t0_vIm = _mm256_add_ps(im_v0, im_v2);
            __m256 t1_vRe = _mm256_sub_ps(re_v0, re_v2);
            __m256 t1_vIm = _mm256_sub_ps(im_v0, im_v2);
            __m256 t2_vRe = _mm256_add_ps(re_v1, re_v3);
            __m256 t2_vIm = _mm256_add_ps(im_v1, im_v3);
            __m256 t3_vRe = _mm256_sub_ps(im_v1, im_v3);
            __m256 t3_vIm = _mm256_sub_ps(re_v3, re_v1);

            re_v0 = _mm256_add_ps(t0_vRe, t2_vRe);
            im_v0 = _mm256_add_ps(t0_vIm, t2_vIm);
            re_v2 = _mm256_sub_ps(t0_vRe, t2_vRe);
            im_v2 = _mm256_sub_ps(t0_vIm, t2_vIm);
#if MC_INVERSE_FFT
            re_v1 = _mm256_sub_ps(t1_vRe, t3_vRe);
            im_v1 = _mm256_sub_ps(t1_vIm, t3_vIm);
            re_v3 = _mm256_add_ps(t1_vRe, t3_vRe);
            im_v3 = _mm256_add_ps(t1_vIm, t3_vIm);

            t1_vRe = _mm256_mul_ps(re_v1, twdB_vRe);
            t1_vIm = _mm256_mul_ps(im_v1, twdB_vRe);
            t1_vRe = _mm256_fmadd_ps(im_v1, twdB_vIm, t1_vRe);
            t1_vIm = _mm256_fnmadd_ps(re_v1, twdB_vIm, t1_vIm);
            t2_vRe = _mm256_mul_ps(re_v2, twdC_vRe);
            t2_vIm = _mm256_mul_ps(im_v2, twdC_vRe);
            t2_vRe = _mm256_fmadd_ps(im_v2, twdC_vIm, t2_vRe);
            t2_vIm = _mm256_fnmadd_ps(re_v2, twdC_vIm, t2_vIm);
            t3_vRe = _mm256_mul_ps(re_v3, twdD_vRe);
            t3_vIm = _mm256_mul_ps(im_v3, twdD_vRe);
            t3_vRe = _mm256_fmadd_ps(im_v3, twdD_vIm, t3_vRe);
            t3_vIm = _mm256_fnmadd_ps(re_v3, twdD_vIm, t3_vIm);
#else
            re_v1 = _mm256_add_ps(t1_vRe, t3_vRe);
            im_v1 = _mm256_add_ps(t1_vIm, t3_vIm);
            re_v3 = _mm256_sub_ps(t1_vRe, t3_vRe);
            im_v3 = _mm256_sub_ps(t1_vIm, t3_vIm);

            t1_vRe = _mm256_mul_ps(re_v1, twdB_vRe);
            t1_vIm = _mm256_mul_ps(im_v1, twdB_vRe);
            t1_vRe = _mm256_fnmadd_ps(im_v1, twdB_vIm, t1_vRe);
            t1_vIm = _mm256_fmadd_ps(re_v1, twdB_vIm, t1_vIm);
            t2_vRe = _mm256_mul_ps(re_v2, twdC_vRe);
            t2_vIm = _mm256_mul_ps(im_v2, twdC_vRe);
            t2_vRe = _mm256_fnmadd_ps(im_v2, twdC_vIm, t2_vRe);
            t2_vIm = _mm256_fmadd_ps(re_v2, twdC_vIm, t2_vIm);
            t3_vRe = _mm256_mul_ps(re_v3, twdD_vRe);
            t3_vIm = _mm256_mul_ps(im_v3, twdD_vRe);
            t3_vRe = _mm256_fnmadd_ps(im_v3, twdD_vIm, t3_vRe);
            t3_vIm = _mm256_fmadd_ps(re_v3, twdD_vIm, t3_vIm);
#endif

            _mm256_storeu_ps(aRe, re_v0);
            _mm256_storeu_ps(bRe, t1_vRe);
            _mm256_storeu_ps(cRe, t2_vRe);
            _mm256_storeu_ps(dRe, t3_vRe);
            _mm256_storeu_ps(aIm, im_v0);
            _mm256_storeu_ps(bIm, t1_vIm);
            _mm256_storeu_ps(cIm, t2_vIm);
            _mm256_storeu_ps(dIm, t3_vIm);

            aRe += 8u;
            bRe += 8u;
            cRe += 8u;
            dRe += 8u;
            aIm += 8u;
            bIm += 8u;
            cIm += 8u;
            dIm += 8u;
            twd += 48u;
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
