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

#include <arm_neon.h>
#include "utils.h"

static void MC_FUNC_TEMPLATE(MC_FFT_DIRECTION, rad4_mono_depth1, neon) (float * restrict re, float * restrict im, uint32_t fftLength) {
    for (uint32_t stepIdx = 0; stepIdx < fftLength; stepIdx += 16u) {
        float32x4x4_t re_v = vld4q_f32(re);
        float32x4x4_t im_v = vld4q_f32(im);
        float32x4_t t0_vRe = vaddq_f32(re_v.val[0], re_v.val[2u]);
        float32x4_t t0_vIm = vaddq_f32(im_v.val[0],im_v.val[2]);
        float32x4_t t1_vRe = vsubq_f32(re_v.val[0], re_v.val[2u]);
        float32x4_t t1_vIm = vsubq_f32(im_v.val[0],im_v.val[2]);
        float32x4_t t2_vRe = vaddq_f32(re_v.val[1u], re_v.val[3u]);
        float32x4_t t2_vIm = vaddq_f32(im_v.val[1u],im_v.val[3u]);
        float32x4_t t3_vRe = vsubq_f32(im_v.val[1u],im_v.val[3u]);
        float32x4_t t3_vIm = vsubq_f32(re_v.val[3u], re_v.val[1u]);

        re_v.val[0] = vaddq_f32(t0_vRe, t2_vRe);
        re_v.val[2u] = vsubq_f32(t0_vRe, t2_vRe);
        im_v.val[0] = vaddq_f32(t0_vIm, t2_vIm);
        im_v.val[2u] = vsubq_f32(t0_vIm, t2_vIm);
#if MC_INVERSE_FFT
        re_v.val[1u] = vsubq_f32(t1_vRe, t3_vRe);
        im_v.val[1u] = vsubq_f32(t1_vIm, t3_vIm);
        re_v.val[3u] = vaddq_f32(t1_vRe, t3_vRe);
        im_v.val[3u] = vaddq_f32(t1_vIm, t3_vIm);
#else
        re_v.val[1u] = vaddq_f32(t1_vRe, t3_vRe);
        im_v.val[1u] = vaddq_f32(t1_vIm, t3_vIm);
        re_v.val[3u] = vsubq_f32(t1_vRe, t3_vRe);
        im_v.val[3u] = vsubq_f32(t1_vIm, t3_vIm);
#endif
        vst4q_f32(re, re_v);
        vst4q_f32(im, im_v);
        re += 16u;
        im += 16u;
    }
}

static void MC_FUNC_TEMPLATE(MC_FFT_DIRECTION, dit_rad4_mono_depth2_odd, neon) (float * restrict re, float * restrict im, const float * restrict twiddle, uint32_t fftLength) {
    float32x4_t twdB_re = {1.f, twiddle[0], 1.f, twiddle[0]};
    float32x4_t twdB_im = {0.f, twiddle[1], 0.f, twiddle[1]};
    float32x4_t twdC_re = {1.f, twiddle[2], 1.f, twiddle[2]};
    float32x4_t twdC_im = {0.f, twiddle[3], 0.f, twiddle[3]};
    float32x4_t twdD_re = {1.f, twiddle[4], 1.f, twiddle[4]};
    float32x4_t twdD_im = {0.f, twiddle[5], 0.f, twiddle[5]};

    for (uint32_t stepIdx = 0; stepIdx < fftLength; stepIdx += 16u) {
        float32x4_t re_v0 = vld1q_f32(re);     /**< aRe0, aRe1, bRe0, bRe1 */
        float32x4_t re_v1 = vld1q_f32(re+4u);  /**< cRe0, cRe1, dRe0, dRe1 */
        float32x4_t re_v2 = vld1q_f32(re+8u);  /**< aRe2, aRe3, bRe2, bRe3 */
        float32x4_t re_v3 = vld1q_f32(re+12u); /**< cRe2, cRe3, dRe2, dRe3 */
        float32x4_t im_v0 = vld1q_f32(im);
        float32x4_t im_v1 = vld1q_f32(im+4u);
        float32x4_t im_v2 = vld1q_f32(im+8u);
        float32x4_t im_v3 = vld1q_f32(im+12u);

        float32x4_t aRe_v = vcombine_f32(vget_low_f32(re_v0), vget_low_f32(re_v2));
        float32x4_t bRe_v = vcombine_f32(vget_high_f32(re_v0), vget_high_f32(re_v2));
        float32x4_t cRe_v = vcombine_f32(vget_low_f32(re_v1), vget_low_f32(re_v3));
        float32x4_t dRe_v = vcombine_f32(vget_high_f32(re_v1), vget_high_f32(re_v3));
        float32x4_t aIm_v = vcombine_f32(vget_low_f32(im_v0), vget_low_f32(im_v2));
        float32x4_t bIm_v = vcombine_f32(vget_high_f32(im_v0), vget_high_f32(im_v2));
        float32x4_t cIm_v = vcombine_f32(vget_low_f32(im_v1), vget_low_f32(im_v3));
        float32x4_t dIm_v = vcombine_f32(vget_high_f32(im_v1), vget_high_f32(im_v3));

#if MC_INVERSE_FFT
        float32x4_t acc0Re_v = vmlaq_f32(vmulq_f32(bRe_v, twdB_re), bIm_v, twdB_im);
        float32x4_t acc0Im_v = vmlsq_f32(vmulq_f32(bIm_v, twdB_re), bRe_v, twdB_im);
        float32x4_t acc1Re_v = vmlaq_f32(vmulq_f32(cRe_v, twdC_re), cIm_v, twdC_im);
        float32x4_t acc1Im_v = vmlsq_f32(vmulq_f32(cIm_v, twdC_re), cRe_v, twdC_im);
        float32x4_t acc2Re_v = vmlaq_f32(vmulq_f32(dRe_v, twdD_re), dIm_v, twdD_im);
        float32x4_t acc2Im_v = vmlsq_f32(vmulq_f32(dIm_v, twdD_re), dRe_v, twdD_im);
#else
        float32x4_t acc0Re_v = vmlsq_f32(vmulq_f32(bRe_v, twdB_re), bIm_v, twdB_im);
        float32x4_t acc0Im_v = vmlaq_f32(vmulq_f32(bIm_v, twdB_re), bRe_v, twdB_im);
        float32x4_t acc1Re_v = vmlsq_f32(vmulq_f32(cRe_v, twdC_re), cIm_v, twdC_im);
        float32x4_t acc1Im_v = vmlaq_f32(vmulq_f32(cIm_v, twdC_re), cRe_v, twdC_im);
        float32x4_t acc2Re_v = vmlsq_f32(vmulq_f32(dRe_v, twdD_re), dIm_v, twdD_im);
        float32x4_t acc2Im_v = vmlaq_f32(vmulq_f32(dIm_v, twdD_re), dRe_v, twdD_im);
#endif
        re_v0 = vaddq_f32(aRe_v, acc1Re_v);
        im_v0 = vaddq_f32(aIm_v, acc1Im_v);
        re_v1 = vsubq_f32(aRe_v, acc1Re_v);
        im_v1 = vsubq_f32(aIm_v, acc1Im_v);
        re_v2 = vaddq_f32(acc0Re_v, acc2Re_v);
        im_v2 = vaddq_f32(acc0Im_v, acc2Im_v);
        re_v3 = vsubq_f32(acc0Im_v, acc2Im_v); // conj
        im_v3 = vsubq_f32(acc2Re_v, acc0Re_v); // conj

        aRe_v = vaddq_f32(re_v0, re_v2);
        aIm_v = vaddq_f32(im_v0, im_v2);
#if MC_INVERSE_FFT
        bRe_v = vsubq_f32(re_v1, re_v3);
        bIm_v = vsubq_f32(im_v1, im_v3);
#else
        bRe_v = vaddq_f32(re_v1, re_v3);
        bIm_v = vaddq_f32(im_v1, im_v3);
#endif
        cRe_v = vsubq_f32(re_v0, re_v2);
        cIm_v = vsubq_f32(im_v0, im_v2);
#if MC_INVERSE_FFT
        dRe_v = vaddq_f32(re_v1, re_v3);
        dIm_v = vaddq_f32(im_v1, im_v3);
#else
        dRe_v = vsubq_f32(re_v1, re_v3);
        dIm_v = vsubq_f32(im_v1, im_v3);
#endif
        vst1q_f32(re,     vcombine_f32(vget_low_f32(aRe_v), vget_low_f32(bRe_v)));
        vst1q_f32(re+4u,  vcombine_f32(vget_low_f32(cRe_v), vget_low_f32(dRe_v)));
        vst1q_f32(re+8u,  vcombine_f32(vget_high_f32(aRe_v), vget_high_f32(bRe_v)));
        vst1q_f32(re+12u, vcombine_f32(vget_high_f32(cRe_v), vget_high_f32(dRe_v)));

        vst1q_f32(im,     vcombine_f32(vget_low_f32(aIm_v), vget_low_f32(bIm_v)));
        vst1q_f32(im+4u,  vcombine_f32(vget_low_f32(cIm_v), vget_low_f32(dIm_v)));
        vst1q_f32(im+8u,  vcombine_f32(vget_high_f32(aIm_v), vget_high_f32(bIm_v)));
        vst1q_f32(im+12u, vcombine_f32(vget_high_f32(cIm_v), vget_high_f32(dIm_v)));
        re += 16u;
        im += 16u;
    }
}

static void MC_FUNC_TEMPLATE(MC_FFT_DIRECTION, dit_rad4_mono_depth2, neon)(float * restrict re, float * restrict im, const float * restrict twiddle, uint32_t fftLength) {
    float32x4_t twdB_vRe = vld1q_f32(twiddle);
    float32x4_t twdB_vIm = vld1q_f32(twiddle+4u);
    float32x4_t twdC_vRe = vld1q_f32(twiddle+8u);
    float32x4_t twdC_vIm = vld1q_f32(twiddle+12u);
    float32x4_t twdD_vRe = vld1q_f32(twiddle+16u);
    float32x4_t twdD_vIm = vld1q_f32(twiddle+20u);
    
    for (uint32_t stepIdx = 0; stepIdx < fftLength; stepIdx += 16u) {
        float32x4_t re_v0 = vld1q_f32(re);
        float32x4_t re_v1 = vld1q_f32(re+4u);
        float32x4_t re_v2 = vld1q_f32(re+8u);
        float32x4_t re_v3 = vld1q_f32(re+12u);
        float32x4_t im_v0 = vld1q_f32(im);
        float32x4_t im_v1 = vld1q_f32(im+4u);
        float32x4_t im_v2 = vld1q_f32(im+8u);
        float32x4_t im_v3 = vld1q_f32(im+12u);

#if MC_INVERSE_FFT
        float32x4_t acc0_vRe = vmlaq_f32(vmulq_f32(re_v1, twdB_vRe), im_v1, twdB_vIm);
        float32x4_t acc0_vIm = vmlsq_f32(vmulq_f32(im_v1, twdB_vRe), re_v1, twdB_vIm);
        float32x4_t acc1_vRe = vmlaq_f32(vmulq_f32(re_v2, twdC_vRe), im_v2, twdC_vIm);
        float32x4_t acc1_vIm = vmlsq_f32(vmulq_f32(im_v2, twdC_vRe), re_v2, twdC_vIm);
        float32x4_t acc2_vRe = vmlaq_f32(vmulq_f32(re_v3, twdD_vRe), im_v3, twdD_vIm);
        float32x4_t acc2_vIm = vmlsq_f32(vmulq_f32(im_v3, twdD_vRe), re_v3, twdD_vIm);
#else
        float32x4_t acc0_vRe = vmlsq_f32(vmulq_f32(re_v1, twdB_vRe), im_v1, twdB_vIm);
        float32x4_t acc0_vIm = vmlaq_f32(vmulq_f32(im_v1, twdB_vRe), re_v1, twdB_vIm);
        float32x4_t acc1_vRe = vmlsq_f32(vmulq_f32(re_v2, twdC_vRe), im_v2, twdC_vIm);
        float32x4_t acc1_vIm = vmlaq_f32(vmulq_f32(im_v2, twdC_vRe), re_v2, twdC_vIm);
        float32x4_t acc2_vRe = vmlsq_f32(vmulq_f32(re_v3, twdD_vRe), im_v3, twdD_vIm);
        float32x4_t acc2_vIm = vmlaq_f32(vmulq_f32(im_v3, twdD_vRe), re_v3, twdD_vIm);
#endif
        float32x4_t t0_vRe = vaddq_f32(re_v0, acc1_vRe);
        float32x4_t t0_vIm = vaddq_f32(im_v0, acc1_vIm);
        float32x4_t t1_vRe = vsubq_f32(re_v0, acc1_vRe);
        float32x4_t t1_vIm = vsubq_f32(im_v0, acc1_vIm);
        float32x4_t t2_vRe = vaddq_f32(acc0_vRe, acc2_vRe);
        float32x4_t t2_vIm = vaddq_f32(acc0_vIm, acc2_vIm);
        float32x4_t t3_vRe = vsubq_f32(acc0_vIm, acc2_vIm);
        float32x4_t t3_vIm = vsubq_f32(acc2_vRe, acc0_vRe);

        re_v0 = vaddq_f32(t0_vRe, t2_vRe);
#if MC_INVERSE_FFT
        re_v1 = vsubq_f32(t1_vRe, t3_vRe);
        re_v3 = vaddq_f32(t1_vRe, t3_vRe);
        im_v1 = vsubq_f32(t1_vIm, t3_vIm);
        im_v3 = vaddq_f32(t1_vIm, t3_vIm);
#else
        re_v1 = vaddq_f32(t1_vRe, t3_vRe);
        re_v3 = vsubq_f32(t1_vRe, t3_vRe);
        im_v1 = vaddq_f32(t1_vIm, t3_vIm);
        im_v3 = vsubq_f32(t1_vIm, t3_vIm);
#endif
        re_v2 = vsubq_f32(t0_vRe, t2_vRe);
        im_v0 = vaddq_f32(t0_vIm, t2_vIm);
        im_v2 = vsubq_f32(t0_vIm, t2_vIm);

        vst1q_f32(re, re_v0);
        vst1q_f32(re+4u, re_v1);
        vst1q_f32(re+8u, re_v2);
        vst1q_f32(re+12u, re_v3);
        vst1q_f32(im, im_v0);
        vst1q_f32(im+4u, im_v1);
        vst1q_f32(im+8u, im_v2);
        vst1q_f32(im+12u, im_v3);
        re += 16u;
        im += 16u;
    }
}

static void MC_FUNC_TEMPLATE(MC_FFT_DIRECTION, dit_rad4_mono_loop, neon)(float *re, float *im, const float * restrict twiddle, uint32_t fftLength, uint32_t step) {
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
            float32x4_t twdB_vRe = vld1q_f32(twd);
            float32x4_t twdB_vIm = vld1q_f32(twd+8u);
            float32x4_t twdC_vRe = vld1q_f32(twd+16u);
            float32x4_t twdC_vIm = vld1q_f32(twd+24u);
            float32x4_t twdD_vRe = vld1q_f32(twd+32u);
            float32x4_t twdD_vIm = vld1q_f32(twd+40u);

            float32x4_t re_v0 = vld1q_f32(aRe);
            float32x4_t re_v1 = vld1q_f32(bRe);
            float32x4_t re_v2 = vld1q_f32(cRe);
            float32x4_t re_v3 = vld1q_f32(dRe);
            float32x4_t im_v0 = vld1q_f32(aIm);
            float32x4_t im_v1 = vld1q_f32(bIm);
            float32x4_t im_v2 = vld1q_f32(cIm);
            float32x4_t im_v3 = vld1q_f32(dIm);

#if MC_INVERSE_FFT
            float32x4_t acc0_vRe = vmlaq_f32(vmulq_f32(re_v1, twdB_vRe), im_v1, twdB_vIm);
            float32x4_t acc0_vIm = vmlsq_f32(vmulq_f32(im_v1, twdB_vRe), re_v1, twdB_vIm);
            float32x4_t acc1_vRe = vmlaq_f32(vmulq_f32(re_v2, twdC_vRe), im_v2, twdC_vIm);
            float32x4_t acc1_vIm = vmlsq_f32(vmulq_f32(im_v2, twdC_vRe), re_v2, twdC_vIm);
            float32x4_t acc2_vRe = vmlaq_f32(vmulq_f32(re_v3, twdD_vRe), im_v3, twdD_vIm);
            float32x4_t acc2_vIm = vmlsq_f32(vmulq_f32(im_v3, twdD_vRe), re_v3, twdD_vIm);
#else
            float32x4_t acc0_vRe = vmlsq_f32(vmulq_f32(re_v1, twdB_vRe), im_v1, twdB_vIm);
            float32x4_t acc0_vIm = vmlaq_f32(vmulq_f32(im_v1, twdB_vRe), re_v1, twdB_vIm);
            float32x4_t acc1_vRe = vmlsq_f32(vmulq_f32(re_v2, twdC_vRe), im_v2, twdC_vIm);
            float32x4_t acc1_vIm = vmlaq_f32(vmulq_f32(im_v2, twdC_vRe), re_v2, twdC_vIm);
            float32x4_t acc2_vRe = vmlsq_f32(vmulq_f32(re_v3, twdD_vRe), im_v3, twdD_vIm);
            float32x4_t acc2_vIm = vmlaq_f32(vmulq_f32(im_v3, twdD_vRe), re_v3, twdD_vIm);
#endif
            float32x4_t t0_vRe = vaddq_f32(re_v0, acc1_vRe);
            float32x4_t t0_vIm = vaddq_f32(im_v0, acc1_vIm);
            float32x4_t t1_vRe = vsubq_f32(re_v0, acc1_vRe);
            float32x4_t t1_vIm = vsubq_f32(im_v0, acc1_vIm);
            float32x4_t t2_vRe = vaddq_f32(acc0_vRe, acc2_vRe);
            float32x4_t t2_vIm = vaddq_f32(acc0_vIm, acc2_vIm);
            float32x4_t t3_vRe = vsubq_f32(acc0_vIm, acc2_vIm);
            float32x4_t t3_vIm = vsubq_f32(acc2_vRe, acc0_vRe);

            re_v0 = vaddq_f32(t0_vRe, t2_vRe);
#if MC_INVERSE_FFT
            re_v1 = vsubq_f32(t1_vRe, t3_vRe);
            re_v3 = vaddq_f32(t1_vRe, t3_vRe);
            im_v1 = vsubq_f32(t1_vIm, t3_vIm);
            im_v3 = vaddq_f32(t1_vIm, t3_vIm);
#else
            re_v1 = vaddq_f32(t1_vRe, t3_vRe);
            re_v3 = vsubq_f32(t1_vRe, t3_vRe);
            im_v1 = vaddq_f32(t1_vIm, t3_vIm);
            im_v3 = vsubq_f32(t1_vIm, t3_vIm);
#endif
            re_v2 = vsubq_f32(t0_vRe, t2_vRe);
            im_v0 = vaddq_f32(t0_vIm, t2_vIm);
            im_v2 = vsubq_f32(t0_vIm, t2_vIm);

            vst1q_f32(aRe, re_v0);
            vst1q_f32(bRe, re_v1);
            vst1q_f32(cRe, re_v2);
            vst1q_f32(dRe, re_v3);
            vst1q_f32(aIm, im_v0);
            vst1q_f32(bIm, im_v1);
            vst1q_f32(cIm, im_v2);
            vst1q_f32(dIm, im_v3);
            
            aRe += 4u;
            bRe += 4u;
            cRe += 4u;
            dRe += 4u;
            aIm += 4u;
            bIm += 4u;
            cIm += 4u;
            dIm += 4u;
            twd += 4u;

            twdB_vRe = vld1q_f32(twd);
            twdB_vIm = vld1q_f32(twd+8u);
            twdC_vRe = vld1q_f32(twd+16u);
            twdC_vIm = vld1q_f32(twd+24u);
            twdD_vRe = vld1q_f32(twd+32u);
            twdD_vIm = vld1q_f32(twd+40u);

            re_v0 = vld1q_f32(aRe);
            re_v1 = vld1q_f32(bRe);
            re_v2 = vld1q_f32(cRe);
            re_v3 = vld1q_f32(dRe);
            im_v0 = vld1q_f32(aIm);
            im_v1 = vld1q_f32(bIm);
            im_v2 = vld1q_f32(cIm);
            im_v3 = vld1q_f32(dIm);

#if MC_INVERSE_FFT
            acc0_vRe = vmlaq_f32(vmulq_f32(re_v1, twdB_vRe), im_v1, twdB_vIm);
            acc0_vIm = vmlsq_f32(vmulq_f32(im_v1, twdB_vRe), re_v1, twdB_vIm);
            acc1_vRe = vmlaq_f32(vmulq_f32(re_v2, twdC_vRe), im_v2, twdC_vIm);
            acc1_vIm = vmlsq_f32(vmulq_f32(im_v2, twdC_vRe), re_v2, twdC_vIm);
            acc2_vRe = vmlaq_f32(vmulq_f32(re_v3, twdD_vRe), im_v3, twdD_vIm);
            acc2_vIm = vmlsq_f32(vmulq_f32(im_v3, twdD_vRe), re_v3, twdD_vIm);
#else
            acc0_vRe = vmlsq_f32(vmulq_f32(re_v1, twdB_vRe), im_v1, twdB_vIm);
            acc0_vIm = vmlaq_f32(vmulq_f32(im_v1, twdB_vRe), re_v1, twdB_vIm);
            acc1_vRe = vmlsq_f32(vmulq_f32(re_v2, twdC_vRe), im_v2, twdC_vIm);
            acc1_vIm = vmlaq_f32(vmulq_f32(im_v2, twdC_vRe), re_v2, twdC_vIm);
            acc2_vRe = vmlsq_f32(vmulq_f32(re_v3, twdD_vRe), im_v3, twdD_vIm);
            acc2_vIm = vmlaq_f32(vmulq_f32(im_v3, twdD_vRe), re_v3, twdD_vIm);
#endif
            t0_vRe = vaddq_f32(re_v0, acc1_vRe);
            t0_vIm = vaddq_f32(im_v0, acc1_vIm);
            t1_vRe = vsubq_f32(re_v0, acc1_vRe);
            t1_vIm = vsubq_f32(im_v0, acc1_vIm);
            t2_vRe = vaddq_f32(acc0_vRe, acc2_vRe);
            t2_vIm = vaddq_f32(acc0_vIm, acc2_vIm);
            t3_vRe = vsubq_f32(acc0_vIm, acc2_vIm);
            t3_vIm = vsubq_f32(acc2_vRe, acc0_vRe);

            re_v0 = vaddq_f32(t0_vRe, t2_vRe);
#if MC_INVERSE_FFT
            re_v1 = vsubq_f32(t1_vRe, t3_vRe);
            re_v3 = vaddq_f32(t1_vRe, t3_vRe);
            im_v1 = vsubq_f32(t1_vIm, t3_vIm);
            im_v3 = vaddq_f32(t1_vIm, t3_vIm);
#else
            re_v1 = vaddq_f32(t1_vRe, t3_vRe);
            re_v3 = vsubq_f32(t1_vRe, t3_vRe);
            im_v1 = vaddq_f32(t1_vIm, t3_vIm);
            im_v3 = vsubq_f32(t1_vIm, t3_vIm);
#endif
            re_v2 = vsubq_f32(t0_vRe, t2_vRe);
            im_v0 = vaddq_f32(t0_vIm, t2_vIm);
            im_v2 = vsubq_f32(t0_vIm, t2_vIm);

            vst1q_f32(aRe, re_v0);
            vst1q_f32(bRe, re_v1);
            vst1q_f32(cRe, re_v2);
            vst1q_f32(dRe, re_v3);
            vst1q_f32(aIm, im_v0);
            vst1q_f32(bIm, im_v1);
            vst1q_f32(cIm, im_v2);
            vst1q_f32(dIm, im_v3);
            
            aRe += 4u;
            bRe += 4u;
            cRe += 4u;
            dRe += 4u;
            aIm += 4u;
            bIm += 4u;
            cIm += 4u;
            dIm += 4u;
            twd += 44u;
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

static void MC_FUNC_TEMPLATE(MC_FFT_DIRECTION, dif_rad4_mono_depth2_odd, neon) (float * restrict re, float * restrict im, const float * restrict twiddle, uint32_t fftLength) {
    float32x4_t twdB_re = {1.f, twiddle[0], 1.f, twiddle[0]};
    float32x4_t twdB_im = {0.f, twiddle[1], 0.f, twiddle[1]};
    float32x4_t twdC_re = {1.f, twiddle[2], 1.f, twiddle[2]};
    float32x4_t twdC_im = {0.f, twiddle[3], 0.f, twiddle[3]};
    float32x4_t twdD_re = {1.f, twiddle[4], 1.f, twiddle[4]};
    float32x4_t twdD_im = {0.f, twiddle[5], 0.f, twiddle[5]};

    for (uint32_t stepIdx = 0; stepIdx < fftLength; stepIdx += 16u) {
        float32x4_t re_v0 = vld1q_f32(re);     /**< aRe0, aRe1, bRe0, bRe1 */
        float32x4_t re_v1 = vld1q_f32(re+4u);  /**< cRe0, cRe1, dRe0, dRe1 */
        float32x4_t re_v2 = vld1q_f32(re+8u);  /**< aRe2, aRe3, bRe2, bRe3 */
        float32x4_t re_v3 = vld1q_f32(re+12u); /**< cRe2, cRe3, dRe2, dRe3 */
        float32x4_t im_v0 = vld1q_f32(im);
        float32x4_t im_v1 = vld1q_f32(im+4u);
        float32x4_t im_v2 = vld1q_f32(im+8u);
        float32x4_t im_v3 = vld1q_f32(im+12u);

        float32x4_t aRe_v = vcombine_f32(vget_low_f32(re_v0), vget_low_f32(re_v2));
        float32x4_t bRe_v = vcombine_f32(vget_high_f32(re_v0), vget_high_f32(re_v2));
        float32x4_t cRe_v = vcombine_f32(vget_low_f32(re_v1), vget_low_f32(re_v3));
        float32x4_t dRe_v = vcombine_f32(vget_high_f32(re_v1), vget_high_f32(re_v3));
        float32x4_t aIm_v = vcombine_f32(vget_low_f32(im_v0), vget_low_f32(im_v2));
        float32x4_t bIm_v = vcombine_f32(vget_high_f32(im_v0), vget_high_f32(im_v2));
        float32x4_t cIm_v = vcombine_f32(vget_low_f32(im_v1), vget_low_f32(im_v3));
        float32x4_t dIm_v = vcombine_f32(vget_high_f32(im_v1), vget_high_f32(im_v3));

        re_v0 = vaddq_f32(aRe_v, cRe_v);
        im_v0 = vaddq_f32(aIm_v, cIm_v);
        re_v1 = vsubq_f32(aRe_v, cRe_v);
        im_v1 = vsubq_f32(aIm_v, cIm_v);
        re_v2 = vaddq_f32(bRe_v, dRe_v);
        im_v2 = vaddq_f32(bIm_v, dIm_v);
        re_v3 = vsubq_f32(bIm_v, dIm_v); // conj
        im_v3 = vsubq_f32(dRe_v, bRe_v); // conj

        aRe_v = vaddq_f32(re_v0, re_v2);
        aIm_v = vaddq_f32(im_v0, im_v2);
        cRe_v = vsubq_f32(re_v0, re_v2);
        cIm_v = vsubq_f32(im_v0, im_v2);
#if MC_INVERSE_FFT
        bRe_v = vsubq_f32(re_v1, re_v3);
        bIm_v = vsubq_f32(im_v1, im_v3);
        dRe_v = vaddq_f32(re_v1, re_v3);
        dIm_v = vaddq_f32(im_v1, im_v3);

        float32x4_t accRe_v = vmlaq_f32(vmulq_f32(bRe_v, twdB_re), bIm_v, twdB_im);
        float32x4_t accIm_v = vmlsq_f32(vmulq_f32(bIm_v, twdB_re), bRe_v, twdB_im);
        bRe_v = accRe_v;
        bIm_v = accIm_v;
        accRe_v = vmlaq_f32(vmulq_f32(cRe_v, twdC_re), cIm_v, twdC_im);
        accIm_v = vmlsq_f32(vmulq_f32(cIm_v, twdC_re), cRe_v, twdC_im);
        cRe_v = accRe_v;
        cIm_v = accIm_v;
        accRe_v = vmlaq_f32(vmulq_f32(dRe_v, twdD_re), dIm_v, twdD_im);
        accIm_v = vmlsq_f32(vmulq_f32(dIm_v, twdD_re), dRe_v, twdD_im);
        dRe_v = accRe_v;
        dIm_v = accIm_v;
#else
        bRe_v = vaddq_f32(re_v1, re_v3);
        bIm_v = vaddq_f32(im_v1, im_v3);
        dRe_v = vsubq_f32(re_v1, re_v3);
        dIm_v = vsubq_f32(im_v1, im_v3);

        float32x4_t accRe_v = vmlsq_f32(vmulq_f32(bRe_v, twdB_re), bIm_v, twdB_im);
        float32x4_t accIm_v = vmlaq_f32(vmulq_f32(bIm_v, twdB_re), bRe_v, twdB_im);
        bRe_v = accRe_v;
        bIm_v = accIm_v;
        accRe_v = vmlsq_f32(vmulq_f32(cRe_v, twdC_re), cIm_v, twdC_im);
        accIm_v = vmlaq_f32(vmulq_f32(cIm_v, twdC_re), cRe_v, twdC_im);
        cRe_v = accRe_v;
        cIm_v = accIm_v;
        accRe_v = vmlsq_f32(vmulq_f32(dRe_v, twdD_re), dIm_v, twdD_im);
        accIm_v = vmlaq_f32(vmulq_f32(dIm_v, twdD_re), dRe_v, twdD_im);
        dRe_v = accRe_v;
        dIm_v = accIm_v;
#endif
        vst1q_f32(re,     vcombine_f32(vget_low_f32(aRe_v), vget_low_f32(bRe_v)));
        vst1q_f32(re+4u,  vcombine_f32(vget_low_f32(cRe_v), vget_low_f32(dRe_v)));
        vst1q_f32(re+8u,  vcombine_f32(vget_high_f32(aRe_v), vget_high_f32(bRe_v)));
        vst1q_f32(re+12u, vcombine_f32(vget_high_f32(cRe_v), vget_high_f32(dRe_v)));

        vst1q_f32(im,     vcombine_f32(vget_low_f32(aIm_v), vget_low_f32(bIm_v)));
        vst1q_f32(im+4u,  vcombine_f32(vget_low_f32(cIm_v), vget_low_f32(dIm_v)));
        vst1q_f32(im+8u,  vcombine_f32(vget_high_f32(aIm_v), vget_high_f32(bIm_v)));
        vst1q_f32(im+12u, vcombine_f32(vget_high_f32(cIm_v), vget_high_f32(dIm_v)));
        re += 16u;
        im += 16u;
    }
}

static void MC_FUNC_TEMPLATE(MC_FFT_DIRECTION, dif_rad4_mono_depth2, neon)(float * restrict re, float * restrict im, const float * restrict twiddle, uint32_t fftLength) {
    float32x4_t twdB_vRe = vld1q_f32(twiddle);
    float32x4_t twdB_vIm = vld1q_f32(twiddle+4u);
    float32x4_t twdC_vRe = vld1q_f32(twiddle+8u);
    float32x4_t twdC_vIm = vld1q_f32(twiddle+12u);
    float32x4_t twdD_vRe = vld1q_f32(twiddle+16u);
    float32x4_t twdD_vIm = vld1q_f32(twiddle+20u);
    
    for (uint32_t stepIdx = 0; stepIdx < fftLength; stepIdx += 16u) {
        float32x4_t re_v0 = vld1q_f32(re);
        float32x4_t re_v1 = vld1q_f32(re+4u);
        float32x4_t re_v2 = vld1q_f32(re+8u);
        float32x4_t re_v3 = vld1q_f32(re+12u);
        float32x4_t im_v0 = vld1q_f32(im);
        float32x4_t im_v1 = vld1q_f32(im+4u);
        float32x4_t im_v2 = vld1q_f32(im+8u);
        float32x4_t im_v3 = vld1q_f32(im+12u);

        float32x4_t t0_vRe = vaddq_f32(re_v0, re_v2);
        float32x4_t t0_vIm = vaddq_f32(im_v0, im_v2);
        float32x4_t t1_vRe = vsubq_f32(re_v0, re_v2);
        float32x4_t t1_vIm = vsubq_f32(im_v0, im_v2);
        float32x4_t t2_vRe = vaddq_f32(re_v1, re_v3);
        float32x4_t t2_vIm = vaddq_f32(im_v1, im_v3);
        float32x4_t t3_vRe = vsubq_f32(im_v1, im_v3);
        float32x4_t t3_vIm = vsubq_f32(re_v3, re_v1);

        re_v0 = vaddq_f32(t0_vRe, t2_vRe);
        im_v0 = vaddq_f32(t0_vIm, t2_vIm);
        re_v2 = vsubq_f32(t0_vRe, t2_vRe);
        im_v2 = vsubq_f32(t0_vIm, t2_vIm);
#if MC_INVERSE_FFT
        re_v1 = vsubq_f32(t1_vRe, t3_vRe);
        im_v1 = vsubq_f32(t1_vIm, t3_vIm);
        re_v3 = vaddq_f32(t1_vRe, t3_vRe);
        im_v3 = vaddq_f32(t1_vIm, t3_vIm);

        t1_vRe = vmlaq_f32(vmulq_f32(re_v1, twdB_vRe), im_v1, twdB_vIm);
        t1_vIm = vmlsq_f32(vmulq_f32(im_v1, twdB_vRe), re_v1, twdB_vIm);
        t2_vRe = vmlaq_f32(vmulq_f32(re_v2, twdC_vRe), im_v2, twdC_vIm);
        t2_vIm = vmlsq_f32(vmulq_f32(im_v2, twdC_vRe), re_v2, twdC_vIm);
        t3_vRe = vmlaq_f32(vmulq_f32(re_v3, twdD_vRe), im_v3, twdD_vIm);
        t3_vIm = vmlsq_f32(vmulq_f32(im_v3, twdD_vRe), re_v3, twdD_vIm);
#else
        re_v1 = vaddq_f32(t1_vRe, t3_vRe);
        im_v1 = vaddq_f32(t1_vIm, t3_vIm);
        re_v3 = vsubq_f32(t1_vRe, t3_vRe);
        im_v3 = vsubq_f32(t1_vIm, t3_vIm);

        t1_vRe = vmlsq_f32(vmulq_f32(re_v1, twdB_vRe), im_v1, twdB_vIm);
        t1_vIm = vmlaq_f32(vmulq_f32(im_v1, twdB_vRe), re_v1, twdB_vIm);
        t2_vRe = vmlsq_f32(vmulq_f32(re_v2, twdC_vRe), im_v2, twdC_vIm);
        t2_vIm = vmlaq_f32(vmulq_f32(im_v2, twdC_vRe), re_v2, twdC_vIm);
        t3_vRe = vmlsq_f32(vmulq_f32(re_v3, twdD_vRe), im_v3, twdD_vIm);
        t3_vIm = vmlaq_f32(vmulq_f32(im_v3, twdD_vRe), re_v3, twdD_vIm);
#endif

        vst1q_f32(re, re_v0);
        vst1q_f32(re+4u, t1_vRe);
        vst1q_f32(re+8u, t2_vRe);
        vst1q_f32(re+12u, t3_vRe);
        vst1q_f32(im, im_v0);
        vst1q_f32(im+4u, t1_vIm);
        vst1q_f32(im+8u, t2_vIm);
        vst1q_f32(im+12u, t3_vIm);
        re += 16u;
        im += 16u;
    }
}

static void MC_FUNC_TEMPLATE(MC_FFT_DIRECTION, dif_rad4_mono_loop, neon)(float *re, float *im, const float * restrict twiddle, uint32_t fftLength, uint32_t step) {
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
            float32x4_t twdB_vRe = vld1q_f32(twd);
            float32x4_t twdB_vIm = vld1q_f32(twd+8u);
            float32x4_t twdC_vRe = vld1q_f32(twd+16u);
            float32x4_t twdC_vIm = vld1q_f32(twd+24u);
            float32x4_t twdD_vRe = vld1q_f32(twd+32u);
            float32x4_t twdD_vIm = vld1q_f32(twd+40u);

            float32x4_t re_v0 = vld1q_f32(aRe);
            float32x4_t re_v1 = vld1q_f32(bRe);
            float32x4_t re_v2 = vld1q_f32(cRe);
            float32x4_t re_v3 = vld1q_f32(dRe);
            float32x4_t im_v0 = vld1q_f32(aIm);
            float32x4_t im_v1 = vld1q_f32(bIm);
            float32x4_t im_v2 = vld1q_f32(cIm);
            float32x4_t im_v3 = vld1q_f32(dIm);

            float32x4_t t0_vRe = vaddq_f32(re_v0, re_v2);
            float32x4_t t0_vIm = vaddq_f32(im_v0, im_v2);
            float32x4_t t1_vRe = vsubq_f32(re_v0, re_v2);
            float32x4_t t1_vIm = vsubq_f32(im_v0, im_v2);
            float32x4_t t2_vRe = vaddq_f32(re_v1, re_v3);
            float32x4_t t2_vIm = vaddq_f32(im_v1, im_v3);
            float32x4_t t3_vRe = vsubq_f32(im_v1, im_v3);
            float32x4_t t3_vIm = vsubq_f32(re_v3, re_v1);

            re_v0 = vaddq_f32(t0_vRe, t2_vRe);
            im_v0 = vaddq_f32(t0_vIm, t2_vIm);
            re_v2 = vsubq_f32(t0_vRe, t2_vRe);
            im_v2 = vsubq_f32(t0_vIm, t2_vIm);
#if MC_INVERSE_FFT
            re_v1 = vsubq_f32(t1_vRe, t3_vRe);
            im_v1 = vsubq_f32(t1_vIm, t3_vIm);
            re_v3 = vaddq_f32(t1_vRe, t3_vRe);
            im_v3 = vaddq_f32(t1_vIm, t3_vIm);

            t1_vRe = vmlaq_f32(vmulq_f32(re_v1, twdB_vRe), im_v1, twdB_vIm);
            t1_vIm = vmlsq_f32(vmulq_f32(im_v1, twdB_vRe), re_v1, twdB_vIm);
            t2_vRe = vmlaq_f32(vmulq_f32(re_v2, twdC_vRe), im_v2, twdC_vIm);
            t2_vIm = vmlsq_f32(vmulq_f32(im_v2, twdC_vRe), re_v2, twdC_vIm);
            t3_vRe = vmlaq_f32(vmulq_f32(re_v3, twdD_vRe), im_v3, twdD_vIm);
            t3_vIm = vmlsq_f32(vmulq_f32(im_v3, twdD_vRe), re_v3, twdD_vIm);
#else
            re_v1 = vaddq_f32(t1_vRe, t3_vRe);
            im_v1 = vaddq_f32(t1_vIm, t3_vIm);
            re_v3 = vsubq_f32(t1_vRe, t3_vRe);
            im_v3 = vsubq_f32(t1_vIm, t3_vIm);

            t1_vRe = vmlsq_f32(vmulq_f32(re_v1, twdB_vRe), im_v1, twdB_vIm);
            t1_vIm = vmlaq_f32(vmulq_f32(im_v1, twdB_vRe), re_v1, twdB_vIm);
            t2_vRe = vmlsq_f32(vmulq_f32(re_v2, twdC_vRe), im_v2, twdC_vIm);
            t2_vIm = vmlaq_f32(vmulq_f32(im_v2, twdC_vRe), re_v2, twdC_vIm);
            t3_vRe = vmlsq_f32(vmulq_f32(re_v3, twdD_vRe), im_v3, twdD_vIm);
            t3_vIm = vmlaq_f32(vmulq_f32(im_v3, twdD_vRe), re_v3, twdD_vIm);
#endif

            vst1q_f32(aRe, re_v0);
            vst1q_f32(bRe, t1_vRe);
            vst1q_f32(cRe, t2_vRe);
            vst1q_f32(dRe, t3_vRe);
            vst1q_f32(aIm, im_v0);
            vst1q_f32(bIm, t1_vIm);
            vst1q_f32(cIm, t2_vIm);
            vst1q_f32(dIm, t3_vIm);

            aRe += 4u;
            bRe += 4u;
            cRe += 4u;
            dRe += 4u;
            aIm += 4u;
            bIm += 4u;
            cIm += 4u;
            dIm += 4u;
            twd += 4u;

            twdB_vRe = vld1q_f32(twd);
            twdB_vIm = vld1q_f32(twd+8u);
            twdC_vRe = vld1q_f32(twd+16u);
            twdC_vIm = vld1q_f32(twd+24u);
            twdD_vRe = vld1q_f32(twd+32u);
            twdD_vIm = vld1q_f32(twd+40u);

            re_v0 = vld1q_f32(aRe);
            re_v1 = vld1q_f32(bRe);
            re_v2 = vld1q_f32(cRe);
            re_v3 = vld1q_f32(dRe);
            im_v0 = vld1q_f32(aIm);
            im_v1 = vld1q_f32(bIm);
            im_v2 = vld1q_f32(cIm);
            im_v3 = vld1q_f32(dIm);

            t0_vRe = vaddq_f32(re_v0, re_v2);
            t0_vIm = vaddq_f32(im_v0, im_v2);
            t1_vRe = vsubq_f32(re_v0, re_v2);
            t1_vIm = vsubq_f32(im_v0, im_v2);
            t2_vRe = vaddq_f32(re_v1, re_v3);
            t2_vIm = vaddq_f32(im_v1, im_v3);
            t3_vRe = vsubq_f32(im_v1, im_v3);
            t3_vIm = vsubq_f32(re_v3, re_v1);

            re_v0 = vaddq_f32(t0_vRe, t2_vRe);
            im_v0 = vaddq_f32(t0_vIm, t2_vIm);
            re_v2 = vsubq_f32(t0_vRe, t2_vRe);
            im_v2 = vsubq_f32(t0_vIm, t2_vIm);
#if MC_INVERSE_FFT
            re_v1 = vsubq_f32(t1_vRe, t3_vRe);
            im_v1 = vsubq_f32(t1_vIm, t3_vIm);
            re_v3 = vaddq_f32(t1_vRe, t3_vRe);
            im_v3 = vaddq_f32(t1_vIm, t3_vIm);

            t1_vRe = vmlaq_f32(vmulq_f32(re_v1, twdB_vRe), im_v1, twdB_vIm);
            t1_vIm = vmlsq_f32(vmulq_f32(im_v1, twdB_vRe), re_v1, twdB_vIm);
            t2_vRe = vmlaq_f32(vmulq_f32(re_v2, twdC_vRe), im_v2, twdC_vIm);
            t2_vIm = vmlsq_f32(vmulq_f32(im_v2, twdC_vRe), re_v2, twdC_vIm);
            t3_vRe = vmlaq_f32(vmulq_f32(re_v3, twdD_vRe), im_v3, twdD_vIm);
            t3_vIm = vmlsq_f32(vmulq_f32(im_v3, twdD_vRe), re_v3, twdD_vIm);
#else
            re_v1 = vaddq_f32(t1_vRe, t3_vRe);
            im_v1 = vaddq_f32(t1_vIm, t3_vIm);
            re_v3 = vsubq_f32(t1_vRe, t3_vRe);
            im_v3 = vsubq_f32(t1_vIm, t3_vIm);

            t1_vRe = vmlsq_f32(vmulq_f32(re_v1, twdB_vRe), im_v1, twdB_vIm);
            t1_vIm = vmlaq_f32(vmulq_f32(im_v1, twdB_vRe), re_v1, twdB_vIm);
            t2_vRe = vmlsq_f32(vmulq_f32(re_v2, twdC_vRe), im_v2, twdC_vIm);
            t2_vIm = vmlaq_f32(vmulq_f32(im_v2, twdC_vRe), re_v2, twdC_vIm);
            t3_vRe = vmlsq_f32(vmulq_f32(re_v3, twdD_vRe), im_v3, twdD_vIm);
            t3_vIm = vmlaq_f32(vmulq_f32(im_v3, twdD_vRe), re_v3, twdD_vIm);
#endif

            vst1q_f32(aRe, re_v0);
            vst1q_f32(bRe, t1_vRe);
            vst1q_f32(cRe, t2_vRe);
            vst1q_f32(dRe, t3_vRe);
            vst1q_f32(aIm, im_v0);
            vst1q_f32(bIm, t1_vIm);
            vst1q_f32(cIm, t2_vIm);
            vst1q_f32(dIm, t3_vIm);

            aRe += 4u;
            bRe += 4u;
            cRe += 4u;
            dRe += 4u;
            aIm += 4u;
            bIm += 4u;
            cIm += 4u;
            dIm += 4u;
            twd += 44u;
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
