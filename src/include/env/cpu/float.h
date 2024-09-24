#ifndef FANG_CPU_FLOAT_H
#define FANG_CPU_FLOAT_H

#include <compiler.h>
#include <stdint.h>

/* ================ TYPES ================ */

/* Generally CPUs do not support half-precision and quarter-precision floating
   point operations. Hence, Fang use software acceleration for half and quarter
   precision floats for CPU Environment. */
/* 16-bit IEEE-754 standard half-precision 1.5.10 float. */
typedef uint16_t _fang_float16_t;

/* 8-bit IEEE-754 compatible quarter-precision 1.3.4 float. */
typedef uint8_t  _fang_float8_t;

/* Fang also support 16-bit 1.8.7 Brain Float. */
typedef fang_float16_t _fang_bfloat16_t;

/* ================ TYPES END ================ */


/* ================ INLINE DEFINITIONS ================ */

/* Convert 32-bit IEEE-754 single-precision float to 16-bit 1.5.10
 * quarter-precision IEEE-754 float. */
FANG_HOT static inline _fang_float16_t _fang_float32_to_float16(float f) {
    uint32_t bits = *(uint32_t *) &f;

    /* Sign bit; also can be represented as 16-bit float of positive/negative
       zero. */
    _fang_float16_t sign = (bits >> 16) & 0x8000;

    /* Exponent. Eq.: e_actual = e_stored - bias; where bias = 2 ^ (N - 1) - 1
       where N is the number of exponent. E.g. for 8-bit exponent bias would be
       127. */
    /* Following the exponent formula, 15 should be added to find `e_stored` for
     * 5-bit exponent of 16-bit float. */
    int16_t exp = ((bits >> 23) & 0xFF) - 127) + 15;

    /* Extract first 10-bits fraction/mantissa. */
    uint16_t mann = (bits >> 13) & 0x03FF;

    if(FANG_LIKELY(exp > 0 && exp < 31))
        sign |= ((exp & 0x1F) << 10) | mann;
    else if(FANG_UNLIKELY(exp >= 31))  // Overflow
        sign |= 0x7C00;

out:
    return sign;
}

/* ================ INLINE DEFINITIONS END ================ */

#endif  // FANG_CPU_FLOAT_H
