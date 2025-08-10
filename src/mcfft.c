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

#include "mcfft.h"
#include "generic/mcfft_generic.h"
#include "x86/mcfft_avx.h"
#include "aarch64/mcfft_neon.h"


#ifndef MC_SELECTOR
#define MC_SELECTOR g
#endif

#ifndef MC_IS_DIF_FFT
#define MC_IS_DIF_FFT (0)
#endif

void mc_fft_mono(const mc_fft_t *context, float * restrict re, float * restrict im, uint32_t length)
{
    MC_NULLPTR_ASSERT(context);
    MC_NULLPTR_ASSERT(context->digitRev);
    MC_NULLPTR_ASSERT(context->twiddle);
    MC_NULLPTR_ASSERT(context->buffer);
    MC_NULLPTR_ASSERT(re);
    MC_NULLPTR_ASSERT(im);
    MC_ASSERT((1U<<context->pow2) == length);
    MC_ASSERT(context->bufLength >= (2u*length));
    MC_ASSERT(MC_MAX_FFT_LENGTH >= length);
    MC_ASSERT(length >= MC_MIN_FFT_LENGTH);
#if MC_IS_DIF_FFT
    MC_FUNC_CALL(fft_dif_mono_core, MC_SELECTOR)(re, im, context->twiddle, context->pow2);
    MC_FUNC_CALL(shuffle_mono, MC_SELECTOR)(re, im, context->buffer, (const uint16_t*)&context->digitRev[length>>1u], length);
#else
    MC_FUNC_CALL(shuffle_mono, MC_SELECTOR)(re, im, context->buffer, (const uint16_t*)&context->digitRev[0], length);
    MC_FUNC_CALL(fft_dit_mono_core, MC_SELECTOR)(re, im, context->twiddle, context->pow2);
#endif
}

void mc_ifft_mono(const mc_fft_t *context, float * restrict re, float * restrict im, uint32_t length) {
    MC_NULLPTR_ASSERT(context);
    MC_NULLPTR_ASSERT(context->digitRev);
    MC_NULLPTR_ASSERT(context->twiddle);
    MC_NULLPTR_ASSERT(context->buffer);
    MC_NULLPTR_ASSERT(re);
    MC_NULLPTR_ASSERT(im);
    MC_ASSERT((1U<<context->pow2) == length);
    MC_ASSERT(context->bufLength >= (2u*length));
    MC_ASSERT(MC_MAX_FFT_LENGTH >= length);
    MC_ASSERT(length >= MC_MIN_FFT_LENGTH);
#if MC_IS_DIF_FFT
    MC_FUNC_CALL(ifft_dif_mono_core, MC_SELECTOR)(re, im, context->twiddle, context->pow2);
    MC_FUNC_CALL(shuffle_mono, MC_SELECTOR)(re, im, context->buffer, (const uint16_t*)&context->digitRev[length>>1u], length);
#else
    MC_FUNC_CALL(shuffle_mono, MC_SELECTOR)(re, im, context->buffer, (const uint16_t*)&context->digitRev[0], length);
    MC_FUNC_CALL(ifft_dit_mono_core, MC_SELECTOR)(re, im, context->twiddle, context->pow2);
#endif
}

void mc_fft_get_digitRev(uint32_t *out, uint32_t length, uint32_t power2) {
    MC_NULLPTR_ASSERT(out);
    MC_ASSERT(MC_DIGIT_LENGTH(power2) == length);
    MC_ASSERT(MC_MAX_FFT_LENGTH >= length);
    uint16_t *dit_map = (uint16_t*)out;
    uint16_t *dif_map = (uint16_t*)&out[length>>1u];
    uint32_t fftLength = 1u<<power2;
    uint32_t base = (power2 % 2u) ? 2u : 4u;

    /** NOTE: Digit reverse is not symmetric for odd power of 2 => swap is not possible */
    /** Digit reverse (DIT version) */
    for (uint32_t i = 0; i < length; ++i) {
        uint32_t k = i;
        uint32_t Nx = base;
        do {
            uint32_t Ny = 4u;
            uint32_t Ni = Ny * Nx;
            k = (k * Ny) % Ni + (k / Nx) % Ny + Ni * (k / Ni);
            Nx = Ni;
        } while (Nx != fftLength);
        dit_map[i] = k;
    }
    /** Digit reverse (DIF version) */
    for (uint32_t i = 0; i < length; ++i) {
        dif_map[dit_map[i]] = i;
    }
}

void mc_fft_get_twiddle(float * restrict out, uint32_t length, uint32_t power2) {
    uint32_t step = (1u<<power2);
    uint32_t totalElements = 0;
    MC_NULLPTR_ASSERT(out);
    MC_ASSERT((1u<<power2) >= MC_MIN_FFT_LENGTH);
    MC_ASSERT(MC_MAX_FFT_LENGTH >= (1u<<power2));
    MC_ASSERT(length == MC_TWIDDLE_LENGTH(power2));
    (void)length;
    do {
        uint32_t saved_elems = mc_fft_rad4_get_twiddle_stage_g(&out[totalElements], step);
        MC_ASSERT(MC_TWIDDLE_STAGE_SIZE(step) == saved_elems);
        totalElements += saved_elems;
        step >>= 2u;
        MC_ASSERT(length >= totalElements);
    } while (step >= 8u);
}

void mc_fft_create_object(mc_fft_object_t *obj, uint32_t power2, void *memory, size_t memSize) {
    MC_NULLPTR_ASSERT(obj);
    MC_NULLPTR_ASSERT(memory);
    MC_ASSERT(MC_MAX_FFT_LENGTH >= (1u<<power2));
    MC_ASSERT((1u<<power2) >= MC_MIN_FFT_LENGTH);
    MC_ASSERT(memSize >= MC_FFT_GET_OBJECT_SIZE(power2));
    uintptr_t memory_addr = 0;
    memset(&obj->context, 0, sizeof(obj->context));
    obj->memory = memory;
    obj->context.pow2 = power2;
    memory_addr = MC_GET_ALIGNED_PTR(obj->memory);
    obj->context.buffer = (float*)memory_addr;
    memory_addr = MC_GET_ALIGNED_PTR(memory_addr+sizeof(obj->context.buffer[0])*MC_BUFFER_LENGTH(power2));
    obj->context.bufLength = MC_BUFFER_LENGTH(power2);
    obj->context.digitRev = (uint32_t*)memory_addr;
    memory_addr = MC_GET_ALIGNED_PTR(memory_addr+sizeof(obj->context.digitRev[0])*MC_DIGIT_LENGTH(power2));
    obj->context.twiddle = (float*)memory_addr;
    memory_addr = MC_GET_ALIGNED_PTR(memory_addr+sizeof(obj->context.twiddle[0])*MC_TWIDDLE_LENGTH(power2));
    MC_ASSERT(memory_addr <= (uintptr_t)obj->memory+memSize);
    mc_fft_get_digitRev(obj->context.digitRev, (1u<<power2), power2);
    mc_fft_get_twiddle(obj->context.twiddle, MC_TWIDDLE_LENGTH(power2), power2);
}

#ifndef MC_EXCLUDE_MALLOC
void mc_fft_allocate(mc_fft_object_t *obj, uint32_t power2) {
    MC_NULLPTR_ASSERT(obj);
    MC_ASSERT(MC_MAX_FFT_LENGTH >= (1u<<power2));
    MC_ASSERT((1u<<power2) >= MC_MIN_FFT_LENGTH);
    size_t memory_size = MC_FFT_GET_OBJECT_SIZE(power2);
    mc_fft_create_object(obj, power2, malloc(memory_size), memory_size);
}

void mc_fft_free(mc_fft_object_t *obj) {
    MC_NULLPTR_ASSERT(obj);
    free(obj->memory);
}
#endif // EXCLUDE_MALLOC

