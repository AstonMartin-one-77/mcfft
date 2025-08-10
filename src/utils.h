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

#ifndef MC_UTILS_H
#define MC_UTILS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define MC_ASSERT(x) assert((x))
#define MC_NULLPTR_ASSERT(x) assert((NULL != (x)))
#ifdef static_assert
#define MC_STATIC_ASSERT(x) static_assert((x), "(" #x ") failed")
#else
#define ASSERT_CONCAT_(a, b) a##b
#define ASSERT_CONCAT(a, b) ASSERT_CONCAT_(a, b)
#define MC_STATIC_ASSERT(x) typedef char ASSERT_CONCAT(static_assertion_at_line_,__LINE__)[(x) ? 1 : -1]
#endif

/** NOTE: only works with direct array name, not a pointer */
#define MC_ARRAY_LENGTH(array) sizeof((array))/sizeof((array)[0])

#define MC_2D_NULLPTR_ASSERT(array2d, channels) for(uint32_t i = 0; i < (channels); ++i) assert((NULL != (array2d)[i]))

/** Memory alignment can be defined outside */
#ifndef MC_MEM_ALIGNMENT
#define MC_MEM_ALIGNMENT (64u)
#endif
#define MC_GET_ALIGNED_PTR(ptr) ((((uintptr_t)(ptr))+(MC_MEM_ALIGNMENT-1u)) & ~((uintptr_t)(MC_MEM_ALIGNMENT-1u)))
#define MC_GET_ALIGNED_SIZE(bytes) ((((size_t)(bytes))+(MC_MEM_ALIGNMENT-1u)) & ~((size_t)(MC_MEM_ALIGNMENT-1u)))

/** For templated functions */
#define MC_FUNC_T(base, name, ext) st_##base##_##name##_##ext
#define MC_FUNC_TEMPLATE(base, name, ext) MC_FUNC_T(base, name, ext)

/** For templated functions */
#define MC_FUNC_C(name, ext) mc_##name##_##ext
#define MC_FUNC_CALL(name, ext) MC_FUNC_C(name, ext)

#define MC_TO_STRING_(msg) #msg
#define MC_MACRO_TO_STRING(line) MC_TO_STRING_(line)
#define MC_WARNING(msg) (__FILE__ ":["MC_MACRO_TO_STRING(__LINE__)"]:" msg)

#define MC_MAX_FFT_LENGTH (16384u)
#define MC_MIN_FFT_LENGTH (32u)

#define MC_PI (3.141592653589793f)

/** Normalise output signal after Inverse FFT
 * 
 * @param out Pointer to user's buffer to store values (see mc_fft_t.digitRev)
 * @param re Pointer to real part of signal
 * @param im Pointer to imag part of signal
 * @param length Length of Re/Im signal
 */
void mc_fft_norm(float * restrict re, float * restrict im, uint32_t length);

/** Unpack Packed Dual FFT to 2x FFTs in Perm format (see IPP library description)
 * NOTE: Only for real signals
 */
void mc_fftr_unpack_dual_to_perm(float * restrict perm0_re, float * restrict perm0_im,
                                 float * restrict perm1_re, float * restrict perm1_im,
                                 const float * restrict dualRe, const float * restrict dualIm, 
                                 uint32_t length);

void mc_test_add_sinwave(float *output, uint32_t length, float gain, float freq, float fs);

float mc_test_mean_error(const float *v0, const float *v1, uint32_t length);

#ifdef __cplusplus
}
#endif

#endif /* MC_UTILS_H */
