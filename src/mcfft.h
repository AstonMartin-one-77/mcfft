
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

#ifndef MC_FFT_MIXED_H
#define MC_FFT_MIXED_H

#ifdef __cplusplus
extern "C" {
#endif

#include "utils.h"

/** Get the number of elements required for buffer (see mc_fft_t) */
#define MC_BUFFER_LENGTH(power2) ((1u<<((power2)+1u)))
/** Get the number of elements required for digit reverse (see mc_fft_t) */
#define MC_DIGIT_LENGTH(power2) ((1u<<(power2)))
/** Twiddle factors calculated for each loop:
 * For instance, for FFT 64-points (n=3): 1, 4, 16 = (4^n-1)/3 (x6 for Re/Im: angle, 2*angle, 3*angle)
 * NOTE: First twiddle is always skipped, then full formula: (((4^n-1)/3)-1)*6 or 2*(4^n-1)-6
 * NOTE: If power of 2 is odd (not a power of 4) then 2 last values are not used  */
#define MC_TWIDDLE_LENGTH(power2) (((power2)%2u) ? ((((1u<<(power2))-1u)<<1u)-8u) : ((((1u<<(power2))-1u)<<1u)-6u))
/** Get the number of elements required to store twiddle values for specific stage */
#define MC_TWIDDLE_STAGE_SIZE(step) ((step == 8u) ? 6u : (6u*((step)>>2u)))

/** FFT context with pre-calculated values and buffer required */
typedef struct mc_fft_t {
    /** NOTE: All arrays in structure better to align by 64 bytes.
     *        Use mc_fft_allocate()/mc_fft_create_object() if possible */
    float *twiddle;     /* Number of twiddle elements must be == MC_TWIDDLE_LENGTH(power2) */
    uint32_t *digitRev; /* Number of digit elements must be == MC_DIGIT_LENGTH(power2) */
    float *buffer;      /* Buffer required to process FFT */
    uint32_t bufLength; /* Number of buffer elements must be >= MC_BUFFER_LENGTH(power2) */
    uint32_t pow2;      /* length of FFT */

} mc_fft_t;

/** Forward FFT
 * 
 * @param context Pointer to context with pre-calculated values and buffer required
 * @param re Pointer to real part of signal
 * @param im Pointer to imag part of signal
 * @param length Length of Re/Im signal to be processed (must be power of 2 and match FFT context)
 */
void mc_fft_mono(const mc_fft_t *context, float * restrict re, float * restrict im, uint32_t length);

/** Inverse FFT
 * 
 * @param context Pointer to context with pre-calculated values and buffer required
 * @param re Pointer to real part of signal
 * @param im Pointer to imag part of signal
 * @param length Length of Re/Im signal to be processed (must be power of 2 and match FFT context)
 * 
 * NOTE: don't forget to call mc_fft_norm() function after
 */
void mc_ifft_mono(const mc_fft_t *context, float * restrict re, float * restrict im, uint32_t length);

/** Get FFT digit reverse of signal
 * 
 * @param out Pointer to user's buffer to store values (see mc_fft_t.digitRev)
 * @param length Length of user's buffer (see MC_DIGIT_LENGTH(power2))
 * @param power2 Power of 2 which reflects required length of FFT
 */
void mc_fft_get_digitRev(uint32_t *out, uint32_t length, uint32_t power2);

/** Get FFT twiddle factors
 * 
 * @param out Pointer to user's buffer to store values (see mc_fft_t.twiddle)
 * @param length Length of user's buffer (see MC_TWIDDLE_LENGTH(power2))
 * @param power2 Power of 2 which reflects required length of FFT
 */
void mc_fft_get_twiddle(float * restrict out, uint32_t length, uint32_t power2);

/** Get FFT object size in bytes if static/non-malloc allocation is required */
#define MC_FFT_GET_OBJECT_SIZE(power2) (MC_GET_ALIGNED_SIZE(sizeof(float)*MC_BUFFER_LENGTH(power2)) \
                                        + MC_GET_ALIGNED_SIZE(sizeof(uint32_t)*MC_DIGIT_LENGTH(power2)) \
                                        + MC_GET_ALIGNED_SIZE(sizeof(float)*MC_TWIDDLE_LENGTH(power2)) \
                                        + MC_MEM_ALIGNMENT)

/** FFT object to control memory alignment and simplify allocation of memory (see mc_fft_t) */
typedef struct mc_fft_object_t {
    mc_fft_t context;
    void *memory;
} mc_fft_object_t;

/** Create FFT object based on allocated memory (non-malloc API)
 * 
 * @param obj Pointer to user's structure where object will be created
 * @param power2 Power of 2 which reflects required length of FFT
 * @param memory Pointer to user's memory which used to create object
 * @param memSize User's memory size in bytes (see MC_FFT_GET_OBJECT_SIZE(power2))
 */
void mc_fft_create_object(mc_fft_object_t *obj, uint32_t power2, void *memory, size_t memSize);

/** Allocate FFT object via malloc/free API (can be excluded by defining EXCLUDE_MALLOC macro)
 * 
 * @param obj Pointer to user's structure where object will be created
 * @param power2 Power of 2 which reflects required length of FFT
 */
void mc_fft_allocate(mc_fft_object_t *obj, uint32_t power2);

/** Release FFT object via malloc/free API (can be excluded by defining EXCLUDE_MALLOC macro)
 * 
 * @param obj Pointer to user's structure where object is created by mc_fft_allocate() function
 */
void mc_fft_free(mc_fft_object_t *obj);

#ifdef __cplusplus
}
#endif

#endif /* MC_FFT_MIXED_H */
 