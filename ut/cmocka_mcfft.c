
#include "utils.h"
#include "mcfft.h"
#include "generic/mcfft_generic.h"
#include "reference_signals.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

#include "x86/mcfft_avx.h"
#include "aarch64/mcfft_neon.h"

#ifndef MC_SELECTOR
#define MC_SELECTOR g
#endif

#define MC_TEST_FS (16000.f)

#define MC_TEST_ZEROS_CHECK(array, threshold) { \
        float zero_array_tmp123[MC_ARRAY_LENGTH((array))]; \
        memset(zero_array_tmp123, 0, sizeof(zero_array_tmp123)); \
        assert_true((threshold) > mc_test_mean_error(zero_array_tmp123, (array), MC_ARRAY_LENGTH((array)))); \
    }


static void cmocka_fft_match_response(void **state) {
    float mono_re0[MC_ARRAY_LENGTH(ref_fft_mono_input0)];
    float mono_im0[MC_ARRAY_LENGTH(ref_fft_mono_input0)];
    float mono_re1[MC_ARRAY_LENGTH(ref_fft_mono_input1)];
    float mono_im1[MC_ARRAY_LENGTH(ref_fft_mono_input1)];
    uint8_t fftObjMem[MC_FFT_GET_OBJECT_SIZE(MC_REF_FFT_POW2)];
    mc_fft_object_t fftObj;
    mc_fft_create_object(&fftObj, MC_REF_FFT_POW2, fftObjMem, MC_ARRAY_LENGTH(fftObjMem));
    (void)state;

    memcpy(mono_re0, ref_fft_mono_input0, sizeof(mono_re0));
    memset(mono_im0, 0, sizeof(mono_im0));
    memcpy(mono_re1, ref_fft_mono_input1, sizeof(mono_re1));
    memset(mono_im1, 0, sizeof(mono_im1));

    mc_fft_mono(&fftObj.context, mono_re0, mono_im0, MC_ARRAY_LENGTH(mono_re0));
    mc_fft_mono(&fftObj.context, mono_re1, mono_im1, MC_ARRAY_LENGTH(mono_re1));

    assert_true(1E-6 > mc_test_mean_error(mono_re0, ref_fft_mono_re0, MC_ARRAY_LENGTH(mono_re0)));
    assert_true(1E-6 > mc_test_mean_error(mono_im0, ref_fft_mono_im0, MC_ARRAY_LENGTH(mono_im0)));
    assert_true(1E-6 > mc_test_mean_error(mono_re1, ref_fft_mono_re1, MC_ARRAY_LENGTH(mono_re1)));
    assert_true(1E-6 > mc_test_mean_error(mono_im1, ref_fft_mono_im1, MC_ARRAY_LENGTH(mono_im1)));

    mc_ifft_mono(&fftObj.context, mono_re0, mono_im0, MC_ARRAY_LENGTH(mono_re0));
    mc_fft_norm(mono_re0, mono_im0, MC_ARRAY_LENGTH(mono_re0));
    assert_true(1E-6 > mc_test_mean_error(mono_re0, ref_fft_mono_input0, MC_ARRAY_LENGTH(mono_re0)));
    MC_TEST_ZEROS_CHECK(mono_im0, 1E-6);
    mc_ifft_mono(&fftObj.context, mono_re1, mono_im1, MC_ARRAY_LENGTH(mono_re0));
    mc_fft_norm(mono_re1, mono_im1, MC_ARRAY_LENGTH(mono_re1));
    assert_true(1E-6 > mc_test_mean_error(mono_re1, ref_fft_mono_input1, MC_ARRAY_LENGTH(mono_re1)));
    MC_TEST_ZEROS_CHECK(mono_im1, 1E-6);

    memcpy(mono_re0, ref_fft_mono_input0, sizeof(mono_re0));
    memset(mono_im0, 0, sizeof(mono_im0));
    memcpy(mono_re1, ref_fft_mono_input1, sizeof(mono_re1));
    memset(mono_im1, 0, sizeof(mono_im1));

    mc_fft_dif_mono_core_g(mono_re0, mono_im0, fftObj.context.twiddle, fftObj.context.pow2);
    mc_ifft_dit_mono_core_g(mono_re0, mono_im0, fftObj.context.twiddle, fftObj.context.pow2);
    mc_fft_norm(mono_re0, mono_im0, MC_ARRAY_LENGTH(mono_re0));
    assert_true(1E-6 > mc_test_mean_error(mono_re0, ref_fft_mono_input0, MC_ARRAY_LENGTH(mono_re0)));
    MC_TEST_ZEROS_CHECK(mono_im0, 1E-6);

    MC_FUNC_CALL(fft_dif_mono_core, MC_SELECTOR)(mono_re1, mono_im1, fftObj.context.twiddle, fftObj.context.pow2);
    MC_FUNC_CALL(ifft_dit_mono_core, MC_SELECTOR)(mono_re1, mono_im1, fftObj.context.twiddle, fftObj.context.pow2);
    mc_fft_norm(mono_re1, mono_im1, MC_ARRAY_LENGTH(mono_re1));
    assert_true(1E-6 > mc_test_mean_error(mono_re1, ref_fft_mono_input1, MC_ARRAY_LENGTH(mono_re1)));
    MC_TEST_ZEROS_CHECK(mono_im1, 1E-6);

    memcpy(mono_re0, ref_fft_mono_input0, sizeof(mono_re0));
    memset(mono_im0, 0, sizeof(mono_im0));
    memcpy(mono_re1, ref_fft_mono_input1, sizeof(mono_re1));
    memset(mono_im1, 0, sizeof(mono_im1));

    mc_shuffle_mono_g(mono_re0, mono_im0, fftObj.context.buffer, (const uint16_t*)fftObj.context.digitRev, MC_ARRAY_LENGTH(mono_re0));
    mc_fft_dit_mono_core_g(mono_re0, mono_im0, fftObj.context.twiddle, fftObj.context.pow2);
    assert_true(1E-6 > mc_test_mean_error(mono_re0, ref_fft_mono_re0, MC_ARRAY_LENGTH(mono_re0)));
    assert_true(1E-6 > mc_test_mean_error(mono_im0, ref_fft_mono_im0, MC_ARRAY_LENGTH(mono_im0)));
    mc_ifft_dif_mono_core_g(mono_re0, mono_im0, fftObj.context.twiddle, fftObj.context.pow2);
    mc_shuffle_mono_g(mono_re0, mono_im0, fftObj.context.buffer, 
        (const uint16_t*)&fftObj.context.digitRev[MC_ARRAY_LENGTH(mono_re0)>>1u], // DIF version of shuffle reversed in map
        MC_ARRAY_LENGTH(mono_re0));
    mc_fft_norm(mono_re0, mono_im0, MC_ARRAY_LENGTH(mono_re0));
    assert_true(1E-6 > mc_test_mean_error(mono_re0, ref_fft_mono_input0, MC_ARRAY_LENGTH(mono_re0)));
    MC_TEST_ZEROS_CHECK(mono_im0, 1E-6);

    MC_FUNC_CALL(shuffle_mono, MC_SELECTOR)(mono_re0, mono_im0, fftObj.context.buffer, (const uint16_t*)fftObj.context.digitRev, MC_ARRAY_LENGTH(mono_re0));
    MC_FUNC_CALL(fft_dit_mono_core, MC_SELECTOR)(mono_re0, mono_im0, fftObj.context.twiddle, fftObj.context.pow2);
    assert_true(1E-6 > mc_test_mean_error(mono_re0, ref_fft_mono_re0, MC_ARRAY_LENGTH(mono_re0)));
    assert_true(1E-6 > mc_test_mean_error(mono_im0, ref_fft_mono_im0, MC_ARRAY_LENGTH(mono_im0)));
    MC_FUNC_CALL(ifft_dif_mono_core, MC_SELECTOR)(mono_re0, mono_im0, fftObj.context.twiddle, fftObj.context.pow2);
    MC_FUNC_CALL(shuffle_mono, MC_SELECTOR)(mono_re0, mono_im0, fftObj.context.buffer, 
        (const uint16_t*)&fftObj.context.digitRev[MC_ARRAY_LENGTH(mono_re0)>>1u], // DIF version of shuffle reversed
        MC_ARRAY_LENGTH(mono_re0));
    mc_fft_norm(mono_re0, mono_im0, MC_ARRAY_LENGTH(mono_re0));
    assert_true(1E-6 > mc_test_mean_error(mono_re0, ref_fft_mono_input0, MC_ARRAY_LENGTH(mono_re0)));
    MC_TEST_ZEROS_CHECK(mono_im0, 1E-6);
}

static void cmocka_odd_match_response(void **state) {
    float mono_re2[MC_ARRAY_LENGTH(ref_fft_mono_input2)];
    float mono_im2[MC_ARRAY_LENGTH(ref_fft_mono_input2)];
    uint8_t fftObjMem[MC_FFT_GET_OBJECT_SIZE(MC_REF_FFT_ODD_POW2)];
    mc_fft_object_t fftObj;
    mc_fft_create_object(&fftObj, MC_REF_FFT_ODD_POW2, fftObjMem, MC_ARRAY_LENGTH(fftObjMem));
    (void)state;

    memcpy(mono_re2, ref_fft_mono_input2, sizeof(mono_re2));
    memset(mono_im2, 0, sizeof(mono_im2));

    mc_fft_mono(&fftObj.context, mono_re2, mono_im2, MC_ARRAY_LENGTH(mono_re2));
    assert_true(1E-6 > mc_test_mean_error(mono_re2, ref_fft_mono_re2, MC_ARRAY_LENGTH(mono_re2)));
    assert_true(1E-6 > mc_test_mean_error(mono_im2, ref_fft_mono_im2, MC_ARRAY_LENGTH(mono_im2)));

    mc_ifft_mono(&fftObj.context, mono_re2, mono_im2, MC_ARRAY_LENGTH(mono_re2));
    mc_fft_norm(mono_re2, mono_im2, MC_ARRAY_LENGTH(mono_re2));
    assert_true(1E-6 > mc_test_mean_error(mono_re2, ref_fft_mono_input2, MC_ARRAY_LENGTH(mono_re2)));
    MC_TEST_ZEROS_CHECK(mono_im2, 1E-6);

    memcpy(mono_re2, ref_fft_mono_input2, sizeof(mono_re2));
    memset(mono_im2, 0, sizeof(mono_im2));

    mc_fft_dif_mono_core_g(mono_re2, mono_im2, fftObj.context.twiddle, fftObj.context.pow2);
    mc_ifft_dit_mono_core_g(mono_re2, mono_im2, fftObj.context.twiddle, fftObj.context.pow2);
    mc_fft_norm(mono_re2, mono_im2, MC_ARRAY_LENGTH(mono_re2));
    assert_true(1E-6 > mc_test_mean_error(mono_re2, ref_fft_mono_input2, MC_ARRAY_LENGTH(mono_re2)));
    MC_TEST_ZEROS_CHECK(mono_im2, 1E-6);

    memcpy(mono_re2, ref_fft_mono_input2, sizeof(mono_re2));
    memset(mono_im2, 0, sizeof(mono_im2));

    MC_FUNC_CALL(fft_dif_mono_core, MC_SELECTOR)(mono_re2, mono_im2, fftObj.context.twiddle, fftObj.context.pow2);
    MC_FUNC_CALL(ifft_dit_mono_core, MC_SELECTOR)(mono_re2, mono_im2, fftObj.context.twiddle, fftObj.context.pow2);
    mc_fft_norm(mono_re2, mono_im2, MC_ARRAY_LENGTH(mono_re2));
    assert_true(1E-6 > mc_test_mean_error(mono_re2, ref_fft_mono_input2, MC_ARRAY_LENGTH(mono_re2)));
    MC_TEST_ZEROS_CHECK(mono_im2, 1E-6);

    memcpy(mono_re2, ref_fft_mono_input2, sizeof(mono_re2));
    memset(mono_im2, 0, sizeof(mono_im2));

    mc_shuffle_mono_g(mono_re2, mono_im2, fftObj.context.buffer, (const uint16_t*)fftObj.context.digitRev, MC_ARRAY_LENGTH(mono_re2));
    mc_fft_dit_mono_core_g(mono_re2, mono_im2, fftObj.context.twiddle, fftObj.context.pow2);
    assert_true(1E-6 > mc_test_mean_error(mono_re2, ref_fft_mono_re2, MC_ARRAY_LENGTH(mono_re2)));
    assert_true(1E-6 > mc_test_mean_error(mono_im2, ref_fft_mono_im2, MC_ARRAY_LENGTH(mono_im2)));
    mc_ifft_dif_mono_core_g(mono_re2, mono_im2, fftObj.context.twiddle, fftObj.context.pow2);
    mc_shuffle_mono_g(mono_re2, mono_im2, fftObj.context.buffer, 
        (const uint16_t*)&fftObj.context.digitRev[MC_ARRAY_LENGTH(mono_re2)>>1u], // DIF version of shuffle reversed in map
        MC_ARRAY_LENGTH(mono_re2));
    mc_fft_norm(mono_re2, mono_im2, MC_ARRAY_LENGTH(mono_re2));
    assert_true(1E-6 > mc_test_mean_error(mono_re2, ref_fft_mono_input2, MC_ARRAY_LENGTH(mono_re2)));
    MC_TEST_ZEROS_CHECK(mono_im2, 1E-6);

    memcpy(mono_re2, ref_fft_mono_input2, sizeof(mono_re2));
    memset(mono_im2, 0, sizeof(mono_im2));

    MC_FUNC_CALL(shuffle_mono, MC_SELECTOR)(mono_re2, mono_im2, fftObj.context.buffer, (const uint16_t*)fftObj.context.digitRev, MC_ARRAY_LENGTH(mono_re2));
    MC_FUNC_CALL(fft_dit_mono_core, MC_SELECTOR)(mono_re2, mono_im2, fftObj.context.twiddle, fftObj.context.pow2);
    assert_true(1E-6 > mc_test_mean_error(mono_re2, ref_fft_mono_re2, MC_ARRAY_LENGTH(mono_re2)));
    assert_true(1E-6 > mc_test_mean_error(mono_im2, ref_fft_mono_im2, MC_ARRAY_LENGTH(mono_im2)));
    MC_FUNC_CALL(ifft_dif_mono_core, MC_SELECTOR)(mono_re2, mono_im2, fftObj.context.twiddle, fftObj.context.pow2);
    MC_FUNC_CALL(shuffle_mono, MC_SELECTOR)(mono_re2, mono_im2, fftObj.context.buffer, 
        (const uint16_t*)&fftObj.context.digitRev[MC_ARRAY_LENGTH(mono_re2)>>1u], // DIF version of shuffle reversed in map
        MC_ARRAY_LENGTH(mono_re2));
    mc_fft_norm(mono_re2, mono_im2, MC_ARRAY_LENGTH(mono_re2));
    assert_true(1E-6 > mc_test_mean_error(mono_re2, ref_fft_mono_input2, MC_ARRAY_LENGTH(mono_re2)));
    MC_TEST_ZEROS_CHECK(mono_im2, 1E-6);
}

int main(void)
{
    const struct CMUnitTest utests[] = {
        cmocka_unit_test(cmocka_fft_match_response),
        cmocka_unit_test(cmocka_odd_match_response)
    };

    return cmocka_run_group_tests(utests, NULL, NULL);
}

