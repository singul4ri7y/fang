#ifndef FANG_CPU_FLOAT_H
#define FANG_CPU_FLOAT_H

#include <compiler.h>
#include <stdint.h>

/* ================ HELPER MACROS ================ */

/* Single and half precision. */
#define _FANG_S2H(f)     _fang_float32_to_float16(f)
#define _FANG_H2S(f)     _fang_float16_to_float32(f)

/* Brain float and single-precision float. */
#define _FANG_S2BH(f)    _fang_float32_to_bfloat16(f)
#define _FANG_BH2S(f)    _fang_bfloat16_to_float32(f)

/* Single and quarter precision. */
#define _FANG_S2Q(f)     _fang_float32_to_float8(f)
#define _FANG_Q2S(f)     _fang_float8_to_float32(f)

/* ================ HELPER MACROS END ================ */


/* ================ TYPES ================ */

/* Generally CPUs do not support half-precision and quarter-precision floating
   point operations. Hence, Fang use software acceleration for half and quarter
   precision floats for CPU Environment. */
/* 16-bit IEEE-754 standard half-precision 1.5.10 float. */
typedef uint16_t _fang_float16_t;

/* 8-bit IEEE-754 compatible quarter-precision 1.4.3 float. */
typedef uint8_t  _fang_float8_t;

/* Fang also support 16-bit 1.8.7 Brain Float. */
typedef _fang_float16_t _fang_bfloat16_t;

/* ================ TYPES END ================ */


/* ================ INLINE DEFINITIONS ================ */

/* Convert 32-bit IEEE-754 single-precision float to 16-bit 1.5.10
   half-precisoin IEEE-754 float. */
FANG_HOT static inline _fang_float16_t _fang_float32_to_float16(float f) {
    uint32_t bits = *(uint32_t *) &f;

    /* Sign bit; also can be represented as 16-bit float of positive/negative
       zero. */
    _fang_float16_t sign = (bits >> 16) & 0x8000;

    /* Exponent. Eq.: e_actual = e_stored - bias; where bias = 2 ^ (N - 1) - 1
       and N is the number of exponent. E.g. for 8-bit exponent bias would be
       127. */
    /* Following the exponent formula, 15 should be added to find `e_stored` for
       5-bit exponent of 16-bit float. */
    /* From all the possible number combination exponent can have, 0 and
       2 ^ N - 1 is reserved representing for zero/subnormal numbers and
       infinity respectively, forming a set of { 1, ... , (2 ^ N - 1) - 1 }. */
    int16_t exp = (((bits >> 23) & 0xFF) - 127) + 15;  // For 16-bit float

    /* Extract first 10-bits fraction/mantissa. */
    uint16_t mann = (bits >> 13) & 0x03FF;  // For 16-bit float

    if(FANG_LIKELY(exp > 0 && exp < 31))  // Normal numbers
        sign |= ((exp & 0x1F) << 10) | mann;
    else if(FANG_UNLIKELY(exp > 30))  // Overflow, infinity or NaN
        sign |= 0x7C00 | mann;

    /* Normal numbers in 32-bit float might as well be subnormal numbers in
       8-bit float. */
    if(FANG_LIKELY(exp <= 0 && -exp + 1 <= 10)) {  // 10: Amount of mantissa bit
        /* Add leading 1 and shift. */
        mann |= 0x0400;
        mann >>= -exp + 1;
        exp = 0;  // Shift exponent to zero

        sign |= ((exp & 0x1F) << 10) | mann;
    }

    /* If underflow which cannot be represented with 16-bit float subnormals,
       simply converge to zero. */

    return sign;
}

/* Convert 16-bit 1.5.10 IEEE-754 half-precision float to IEEE-754
   32-bit single-precision float. */
FANG_HOT static inline float _fang_float16_to_float32(_fang_float16_t f) {
    /* Sign; can be represented as positive/negative zero as 32-bit float. */
    uint32_t sign = (f & 0x8000) << 16;
    int32_t  exp  = (f >> 10) & 0x1F;  // 5-bit exponent
    uint32_t mann = f & 0x03FF;  // 10-bit fraction/mantissa

    if(exp > 0 && exp < 31) {  // Normal numbers
        /* e_actual = e_stored - bias */
        /* Bias = 2 ^ (N - 1) - 1 */
        exp = (exp - 15) + 127;
        sign |= ((exp & 0xFF) << 23) | (mann << 13);
    }
    else if(exp > 30) {  // Overflow, infinity or NaN
        sign |= (0xFF << 23) | (mann << 13);
    }
    /* Subnormal numbers in half-precision are normal numbers in
       single-precision. */
    else if(mann != 0) {
        /* As subnormals are normal numbers in 32-bit float, leading 1 would be
           added, which require adjustment. */
        while((mann & 0x0400) == 0) {  // Until imaginary 11th bit reached
            mann <<= 1;
            exp--;
        }
        /* Get out of subnormal range for 32-bit float. Subnormal region in
           16-bit float is when e_actual = -15. */
        /* Adding 120 instead of 127 would have 7 deficiency, resulting -7 in
           e_actual. But, extra 1 should be added, to adjust the exponent shift
           because while converting to 16-bit float from 32-bit float, mantissa
           was shifted `-exp + 1` amount. */
        exp += 112 + 1;  // 16-bit subnormal region in 32-bit normal region

        sign |= ((exp & 0xFF) << 23) | ((mann & 0x03FF) << 13);
    }

    return *(float *) &sign;
}

/* Convert 32-bit IEEE-754 single-precision float to 16-bit 1.8.7
   half-precision Brain float. */
FANG_HOT static inline _fang_bfloat16_t _fang_float32_to_bfloat16(float f) {
    uint32_t bits = *(uint32_t *) &f;
    return (_fang_bfloat16_t) (bits >> 16) & 0xFF;  // Shrink mantissa
}

/* Converts 16-bit half-precision 1.8.7 Brain float to 32-bit IEEE-754
   single-precision float. */
FANG_HOT static inline float _fang_bfloat16_to_float32(_fang_bfloat16_t f) {
    uint32_t bits = (uint32_t) f << 16;
    return *(float *) &bits;
}

/* Convert 32-bit IEEE-754 single-precision float to 8-bit 1.4.3
   quarter-precision IEEE-754 compatible float. */
FANG_HOT static inline _fang_float8_t _fang_float32_to_float8(float f) {
    uint32_t bits = *(uint32_t *) &f;

    /* Sign bit; also can be represented as 8-bit float of positive/negative
       zero. */
    _fang_float8_t sign = (bits >> 24) & 0x80;

    /* Exponent. Eq.: e_actual = e_stored - bias; where bias = 2 ^ (N - 1) - 1
       and N is the number of exponent. E.g. for 8-bit exponent bias would be
       127. */
    /* Following the exponent formula, 7 should be added to find `e_stored` for
       4-bit exponent of 8-bit float. */
    /* From all the possible number combination exponent can have, 0 and
       2 ^ N - 1 is reserved representing for zero/subnormal numbers and
       infinity respectively, forming a set of { 1, ... , (2 ^ N - 1) - 1 }. */
    int16_t exp = (((bits >> 23) & 0xFF) - 127) + 7;  // For 8-bit float

    /* Extract first 4-bits fraction/mantissa. */
    uint8_t mann = (bits >> 20) & 0x07;  // For 8-bit float

    if(FANG_LIKELY(exp > 0 && exp < 15))  // Normal numbers
        sign |= ((exp & 0x0F) << 3) | mann;
    else if(FANG_UNLIKELY(exp > 14))  // Overflow, infinity or NaN
        sign |= 0x78 | mann;

    /* Normal numbers in 32-bit float might as well be subnormal numbers in
       8-bit float. */
    if(FANG_LIKELY(exp <= 0 && -exp + 1 <= 3)) {  // 3: Amount of mantissa bit
        /* Add leading 1 and shift. */
        mann |= 0x08;
        mann >>= -exp + 1;
        exp = 0;  // Shift exponent to zero

        sign |= ((exp & 0x0F) << 3) | mann;
    }

    /* If underflow which cannot be represented with 8-bit float subnormals,
       simply converge to zero. */

    return sign;
}

/* Convert 8-bit 1.4.3 IEEE-754 compatible quarter-precision float to IEEE-754
   32-bit single-precision float. */
FANG_HOT static inline float _fang_float8_to_float32(_fang_float8_t f) {
    /* Sign; can be represented as positive/negative zero as 32-bit float. */
    uint32_t sign = (f & 0x80) << 24;
    int32_t  exp  = (f >> 3) & 0x0F;  // 4-bit exponent
    uint32_t mann = f & 0x07;  // 3-bit fraction/mantissa

    if(exp > 0 && exp < 15) {  // Normal numbers
        /* e_actual = e_stored - bias */
        /* Bias = 2 ^ (N - 1) - 1 */
        exp = (exp - 7) + 127;
        sign |= ((exp & 0xFF) << 23) | (mann << 20);
    }
    else if(exp > 14) {  // Overflow, infinity or NaN
        sign |= (0xFF << 23) | (mann << 20);
    }
    /* Subnormal numbers in half-precision are normal numbers in
       single-precision. */
    else if(mann != 0) {
        /* As subnormals are normal numbers in 32-bit float, leading 1 would be
           added, which require adjustment. */
        while((mann & 0x08) == 0) {  // Until imaginary 4th bit reached
            mann <<= 1;
            exp--;
        }
        /* Get out of subnormal range for 32-bit float. Subnormal region in
           8-bit float is when e_actual = -7. */
        /* Adding 120 instead of 127 would have 7 deficiency, resulting -7 in
           e_actual. But, extra 1 should be added, to adjust the exponent shift
           because while converting to 8-bit float from 32-bit float, mantissa
           was shifted `-exp + 1` amount. */
        exp += 120 + 1;  // 8-bit subnormal region in 32-bit float normal region

        sign |= ((exp & 0xFF) << 23) | ((mann & 0x07) << 20);
    }

    return *(float *) &sign;
}

/* ================ INLINE DEFINITIONS END ================ */

#endif  // FANG_CPU_FLOAT_H
