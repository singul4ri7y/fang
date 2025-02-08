#ifndef FANG_CPU_AVXMATH_H
#define FANG_CPU_AVXMATH_H

#include <compiler.h>

#if defined(FANG_USE_AVX512) || defined(FANG_USE_AVX2)
#include <immintrin.h>
#endif  // FANG_USE_AVX512 or FANG_USE_AVX2

/* ================ CONSTANT MACROS ================ */

/* ======== EULER'S CONSTANT EXPONENTIAL ======== */

/* Maximum and minimum values `_fang_expf256` can handle before converging
   to inifinity and zero respectively. It is advisable to not to use such
   large/small values. */
#define _expf32_hi        88.3762626647949f
#define _expf32_lo       -88.3762626647949f

/* Value of `ln2` and `log2(e)`, where `ln` denotes natural log. */
#define _expf32_ln2       0.693147180559945f
#define _expf32_log2e     1.44269504088896341f

/* Maclurin series expansion constants, upto 7 terms. */
#define _expf32_c0        0.001388888888888f
#define _expf32_c1        0.008333333333333f
#define _expf32_c2        0.041666666666666f
#define _expf32_c3        0.166666666666666f
#define _expf32_c4        0.500000000000000f

/* ======== EULER'S CONSTANT EXPONENTIAL END ======== */

/* ================ CONSTANTS MACROS END ================ */


/* ================ INLINE DEFINITIONS ================ */

#ifdef FANG_USE_AVX2

/* Calculates the natural exponent of a AVX2 256-bit float32 vector. */
FANG_HOT FANG_INLINE static inline __m256 _fang_expf32_ps256(__m256 x) {
    /* Values of `x` cannot be larger or smaller than `_expf256_hi` or
       `_expf256_lo` respectively. */
    x = _mm256_max_ps(x, _mm256_set1_ps(_expf32_lo));
    x = _mm256_min_ps(x, _mm256_set1_ps(_expf32_hi));

    /* Here, Maclurin's series can be directly applied to calculate the natural
       exponent. But, in this case if the value of `x` is large, there is a big
       hit in the accuracy, thanks to 32-bit float's limited mantissa. */
    /* Hence, Cody-Waite scheme is being applied here by decomposing `x` into
       two parts (high-percision, low magnitude and low-percision,
       high-magnitude), calculating them separately and finally merging together
       to calculate the final value with relatively good precision and
       magnitude. */

    /* ==== MATH ====
         => x = aln2 + b    [Decomposition based on Cody-Waite scheme]

                            Where, `a` is integer and `b` is the high-precision
                            part with negligible numeric value (magnitude).

         => expf(x) = expf(aln2 + b)
                    = expf(aln2) * expf(b)
                    = expf(ln2^a) * expf(b)
                    = 2^a * expf(b)

       As `b` is small, calculating `expf(b)` will converge easily without large
       series expansion while maintaining good accuracy.

       Ignoring `b`, the first equation can be written as follows:
         => x = aln2    [ `b` is negligible ]
         => a = x / ln2
              = x * log2(e)

       `a` should be integer, so
         => a = round(x * log2(e))

       And finally,
         => b = x - aln2
       ==== MATH END ==== */

    register __m256 a = _mm256_mul_ps(x, _mm256_set1_ps(_expf32_log2e));
                    a = _mm256_round_ps(a, _MM_FROUND_TO_NEAREST_INT |
                                           _MM_FROUND_NO_EXC);
    register __m256 b = _mm256_sub_ps(x, _mm256_mul_ps(a,
                        _mm256_set1_ps(_expf32_ln2)));

    /* Maclurin's series for natural exponent is:
         => expf(x) = 1 + x + x^2/2! + x^3/3! + x^4/4! + x^5/5! + ...

       As `x` here is relatively small (which is `b` in this case), small
       expansion of the series may converge. Hence, the equation stands:

         => expf(x) = x^6/6! + x^5/5! + x^4/4! + x^3/3! + x^2/2! + x + 1

       Diminishing redundant calculation of power of `x` by using Horner's
       method:
         => expf(x) = x * (x * (x * (x * (x * (x/6! + 1/5!) + 1/4!) + 1/3!)
                      + 1/2!) + 1) + 1

       Cleaning up the factorial mess with constants:
         => expf(x) = x * (x * (x * (x * (x * (x * c0 + c1) + c2) + c3) + c4)
                      + one) + one
    */

    /* Prepare constants. */
    register __m256 c0  = _mm256_set1_ps(_expf32_c0);
    register __m256 c1  = _mm256_set1_ps(_expf32_c1);
    register __m256 c2  = _mm256_set1_ps(_expf32_c2);
    register __m256 c3  = _mm256_set1_ps(_expf32_c3);
    register __m256 c4  = _mm256_set1_ps(_expf32_c4);
    register __m256 one = _mm256_set1_ps(1.0f);

    /* Calculate expf(b). */
    register __m256 expf_b = c0;
    /* `x * c0 + c1` */
    expf_b = _mm256_fmadd_ps(b, expf_b, c1);
    /* `x * (x * c0 + c1) + c2` */
    expf_b = _mm256_fmadd_ps(b, expf_b, c2);
    /* `x * (x * (x * c0 + c1) + c2) + c3` */
    expf_b = _mm256_fmadd_ps(b, expf_b, c3);
    /* `x * (x * (x * (x * c0 + c1) + c2) + c3) + c4` */
    expf_b = _mm256_fmadd_ps(b, expf_b, c4);
    /* `x * (x * (x * (x * (x * c0 + c1) + c2) + c3) + c4) + one` */
    expf_b = _mm256_fmadd_ps(b, expf_b, one);
    /* `x * (x * (x * (x * (x * (x * c0 + c1) + c2) + c3) + c4) + one) + one` */
    expf_b = _mm256_fmadd_ps(b, expf_b, one);

    /* Calculate `2^a`. */
    register __m256i a_i32 = _mm256_cvtps_epi32(a);
    /* IEEE-754 floats are generally stored as binary scientific notation:
         1.<mantissa> * 2^(<exponent> - <bias>)
       Hence, to convert `2^a` to float, keeping `a + <bias>` in exponent
       should suffice. */
    a_i32 = _mm256_add_epi32(a_i32, _mm256_set1_epi32(0x7F));  // Add bias, 127
    a_i32 = _mm256_slli_epi32(a_i32, 23);                      // Store exponent

    /* Return 2^a * expf(b) */
    return _mm256_mul_ps(_mm256_castsi256_ps(a_i32), expf_b);
}

#endif  // FANG_USE_AVX2

/* ================ INLINE DEFINITIONS END ================ */

#endif  // FANG_CPU_AVXMATH_H
