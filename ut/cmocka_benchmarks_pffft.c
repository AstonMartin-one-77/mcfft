
/** Only for Unix OS & C11 */

#include "mcfft.h"
#include <math.h>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

#include <time.h>
#include <unistd.h>

#include "pffft.h"

#define MC_TEST_ZEROS_CHECK(array, threshold) { \
    float zero_array_tmp123[MC_ARRAY_LENGTH((array))]; \
    memset(zero_array_tmp123, 0, sizeof(zero_array_tmp123)); \
    assert_true((threshold) > mc_test_mean_error(zero_array_tmp123, (array), MC_ARRAY_LENGTH((array)))); \
}

#define MC_TEST_FS (16000.f)

#define MC_TEST_BENCH_CYCLES (500u)
#define MC_TEST_BATCH_SIZE   (50u)
/* Max power of 2 to test */
#define MC_TEST_FFT_POW2 (10u)
#define MC_TEST_FFT_LEN (1u << MC_TEST_FFT_POW2)

static void cmocka_fft_benchmark(uint32_t power2) {
    float ref_mono_real[MC_TEST_FFT_LEN];
    float mono_re0[MC_TEST_FFT_LEN];
    float mono_im0[MC_TEST_FFT_LEN];
    PFFFT_Setup *fft_spec = NULL;
    float *pffft_input = NULL;
    float *pffft_output = NULL;
    
    struct timespec tms;
    uint64_t startNs = 0, endNs = 0, pffft_minimalNs = (1ull<<63), pffft_median = 0, minimalNs = (1ull<<63), mcfft_median = 0;
    mc_fft_object_t fftObj;
    mc_fft_allocate(&fftObj, power2);
    assert_true(MC_TEST_FFT_LEN >= (1u<<power2));
    assert_true(4u == pffft_simd_size());
    fft_spec = pffft_new_setup((1u<<power2), PFFFT_COMPLEX);
    pffft_input = pffft_aligned_malloc(2u*(1u<<power2)*sizeof(float));
    pffft_output = pffft_aligned_malloc(2u*(1u<<power2)*sizeof(float));
    
    memset(ref_mono_real, 0, sizeof(ref_mono_real));
    mc_test_add_sinwave(ref_mono_real, (1u<<power2), 0.8f, 0.1f, MC_TEST_FS);
    mc_test_add_sinwave(ref_mono_real, (1u<<power2), 0.5f, 0.45f, MC_TEST_FS);

    memcpy(mono_re0, ref_mono_real, sizeof(mono_re0));
    memset(mono_im0, 0, sizeof(mono_im0));

    mc_fft_mono(&fftObj.context, mono_re0, mono_im0, (1u<<power2));

    for (uint32_t i = 0; i < (1u<<power2); ++i) {
        pffft_input[2u*i] = ref_mono_real[i];
        pffft_input[2u*i+1u] = 0.f;
    }
    pffft_transform_ordered(fft_spec, pffft_input, pffft_output, NULL, PFFFT_FORWARD);
    for (uint32_t i = 0; i < (1u<<power2); ++i) {
        pffft_input[i] = pffft_output[2u*i];
        pffft_input[i+(1u<<power2)] = pffft_output[2u*i+1u];
    }
    assert_true(1E-6 > mc_test_mean_error(pffft_input, mono_re0, (1u<<power2)));
    assert_true(1E-6 > mc_test_mean_error(&pffft_input[(1u<<power2)], mono_im0, (1u<<power2)));
    pffft_transform_ordered(fft_spec, pffft_output, pffft_input, NULL, PFFFT_BACKWARD);
    mc_fft_norm(pffft_input, &pffft_input[(1u<<power2)], (1u<<power2));
    for (uint32_t i = 0; i < (1u<<power2); ++i) {
        pffft_output[i] = pffft_input[2u*i];
        pffft_output[i+(1u<<power2)] = pffft_input[2u*i+1u];
    }
    assert_true(1E-6 > mc_test_mean_error(pffft_output, ref_mono_real, (1u<<power2)));

    mc_ifft_mono(&fftObj.context, mono_re0, mono_im0, (1u<<power2));
    mc_fft_norm(mono_re0, mono_im0, (1u<<power2));
    assert_true(1E-6 > mc_test_mean_error(mono_re0, ref_mono_real, (1u<<power2)));
    MC_TEST_ZEROS_CHECK(mono_im0, 1E-6);

    for (uint32_t i = 0; i < (1u<<power2); ++i) {
        pffft_input[2u*i] = ref_mono_real[i];
        pffft_input[2u*i+1u] = 0.f;
    }

    sleep(1);

    for (uint32_t c = 0; c < MC_TEST_BENCH_CYCLES; ++c) {
        timespec_get(&tms, TIME_UTC);
        startNs = tms.tv_nsec;
        for (uint32_t b = 0; b < MC_TEST_BATCH_SIZE; ++b) {
            mc_fft_mono(&fftObj.context, mono_re0, mono_im0, (1u<<power2));
            mc_ifft_mono(&fftObj.context, mono_re0, mono_im0, (1u<<power2));
            mc_fft_norm(mono_re0, mono_im0, (1u<<power2));
        }
        timespec_get(&tms, TIME_UTC);
        endNs = tms.tv_nsec;
        if (endNs > startNs) {
            endNs -= startNs;
            minimalNs = (endNs<minimalNs) ? endNs : minimalNs;
        }
    }

    sleep(1);

    for (uint32_t c = 0; c < MC_TEST_BENCH_CYCLES; ++c) {
        timespec_get(&tms, TIME_UTC);
        startNs = tms.tv_nsec;
        for (uint32_t b = 0; b < MC_TEST_BATCH_SIZE; ++b) {
            pffft_transform_ordered(fft_spec, pffft_input, pffft_output, NULL, PFFFT_FORWARD);
            pffft_transform_ordered(fft_spec, pffft_output, pffft_input, NULL, PFFFT_BACKWARD);
            mc_fft_norm(pffft_input, &pffft_input[(1u<<power2)], (1u<<power2));
        }
        timespec_get(&tms, TIME_UTC);
        endNs = tms.tv_nsec;
        if (endNs > startNs) {
            endNs -= startNs;
            pffft_minimalNs = (endNs<pffft_minimalNs) ? endNs : pffft_minimalNs;
        }
    }

    sleep(1);

    mcfft_median = minimalNs;
    for (uint32_t c = 0; c < MC_TEST_BENCH_CYCLES; ++c) {
        timespec_get(&tms, TIME_UTC);
        startNs = tms.tv_nsec;
        for (uint32_t b = 0; b < MC_TEST_BATCH_SIZE; ++b) {
            mc_fft_mono(&fftObj.context, mono_re0, mono_im0, (1u<<power2));
            mc_ifft_mono(&fftObj.context, mono_re0, mono_im0, (1u<<power2));
            mc_fft_norm(mono_re0, mono_im0, (1u<<power2));
        }
        timespec_get(&tms, TIME_UTC);
        endNs = tms.tv_nsec;
        if (endNs > startNs) {
            endNs -= startNs;
            if (endNs <= (mcfft_median+mcfft_median/8)) { // If in range of +12.5%
                mcfft_median = (mcfft_median-(mcfft_median>>5u))+(endNs>>5u);
            }
            minimalNs = (endNs<minimalNs) ? endNs : minimalNs;
        }
    }

    sleep(1);
    
    pffft_median = pffft_minimalNs;
    for (uint32_t c = 0; c < MC_TEST_BENCH_CYCLES; ++c) {
        timespec_get(&tms, TIME_UTC);
        startNs = tms.tv_nsec;
        for (uint32_t b = 0; b < MC_TEST_BATCH_SIZE; ++b) {
            pffft_transform_ordered(fft_spec, pffft_input, pffft_output, NULL, PFFFT_FORWARD);
            pffft_transform_ordered(fft_spec, pffft_output, pffft_input, NULL, PFFFT_BACKWARD);
            mc_fft_norm(pffft_input, &pffft_input[(1u<<power2)], (1u<<power2));
        }
        timespec_get(&tms, TIME_UTC);
        endNs = tms.tv_nsec;
        if (endNs > startNs) {
            endNs -= startNs;
            if (endNs <= (pffft_median+pffft_median/8)) { // If in range of +12.5%
                pffft_median = (pffft_median-(pffft_median>>5u))+(endNs>>5u);
            }
            pffft_minimalNs = (endNs<pffft_minimalNs) ? endNs : pffft_minimalNs;
        }
    }
    mc_fft_free(&fftObj);
    pffft_destroy_setup(fft_spec);
    pffft_aligned_free(pffft_input);
    pffft_aligned_free(pffft_output);
    printf("Mono time: %d Nsec (min: %d), PFFFT time: %d Nsec (min: %d)\r\n", 
        (int)mcfft_median, (int)minimalNs, 
        (int)pffft_median, (int)pffft_minimalNs);
    assert_true((mcfft_median < pffft_median) || (minimalNs < pffft_minimalNs));
}

static void cmocka_fft_benchmark_32(void **state) {
    (void)state;
    cmocka_fft_benchmark(5);
}

static void cmocka_fft_benchmark_64(void **state) {
    (void)state;
    cmocka_fft_benchmark(6);
}

static void cmocka_fft_benchmark_128(void **state) {
    (void)state;
    cmocka_fft_benchmark(7);
}

static void cmocka_fft_benchmark_256(void **state) {
    (void)state;
    cmocka_fft_benchmark(8);
}

static void cmocka_fft_benchmark_512(void **state) {
    (void)state;
    cmocka_fft_benchmark(9);
}

static void cmocka_fft_benchmark_1024(void **state) {
    (void)state;
    cmocka_fft_benchmark(10);
}

int main(void)
{
    const struct CMUnitTest utests[] = {
        cmocka_unit_test(cmocka_fft_benchmark_64),
        cmocka_unit_test(cmocka_fft_benchmark_128),
        cmocka_unit_test(cmocka_fft_benchmark_256),
        cmocka_unit_test(cmocka_fft_benchmark_512),
        cmocka_unit_test(cmocka_fft_benchmark_1024),
    };

    return cmocka_run_group_tests(utests, NULL, NULL);
}

