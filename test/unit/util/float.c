#include <env/cpu/float.h>
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <math.h>
#include <cmocka.h>

/* ================ HELPER MACROS ================ */

#define ASSERT_FLOAT8(f)         \
assert_float_equal(f, _FANG_Q2S(_FANG_S2Q(f)), 1e-6)

#define ASSERT_FLOAT16(f)        \
assert_float_equal(f, _FANG_H2S(_FANG_S2H(f)), 1e-6)

#define ASSERT_OVERFLOW(x, y)                                      \
{ _fang_float_bitcast_t _ux = { .f32 = x }, _uy = { .f32 = y };    \
assert_int_equal(_ux.u32, _uy.u32); }

/* ================ HELPER MACROS END ================ */


/* ================ TESTS ================ */

/* Test 8-bit IEEE-754 compatible float software acceleration. */
static void fang_float8_test(void **state) {
    /* Zero. */
    ASSERT_FLOAT16(0.0);

    /* Negative zero. */
    ASSERT_FLOAT16(-0.0);

    /* Smallest subnormal number. */
    ASSERT_FLOAT16(0.001953125);
    ASSERT_FLOAT16(-0.001953125);

    /* Largest subnormal number. */
    ASSERT_FLOAT16(0.013671875);
    ASSERT_FLOAT16(-0.013671875);

    /* Smallest normal number. */
    ASSERT_FLOAT16(0.015625);
    ASSERT_FLOAT16(-0.015625);

    /* Nearest value to 1/3. */
    ASSERT_FLOAT16(0.3125);
    ASSERT_FLOAT16(-0.3125);

    /* Largest number smaller than 1. */
    ASSERT_FLOAT16(0.9375);
    ASSERT_FLOAT16(-0.9375);

    /* One. */
    ASSERT_FLOAT16(1.0);
    ASSERT_FLOAT16(-1.0);

    /* Smallest number larger than one. */
    ASSERT_FLOAT16(1.125);
    ASSERT_FLOAT16(-1.125);

    /* Largest number representable. */
    ASSERT_FLOAT16(240.0);
    ASSERT_FLOAT16(-240.0);

    /* ==== Test overflow: infinity and NaN ==== */

    float x = 0.0 / 0.0;
    float y = _FANG_Q2S(_FANG_S2Q(x));
    ASSERT_OVERFLOW(x, y);  // NaN

    x = -0.0 / 0.0;
    y = _FANG_Q2S(_FANG_S2Q(x));
    ASSERT_OVERFLOW(x, y);  // -NaN

    x = 1.0 / 0.0;
    y = _FANG_Q2S(_FANG_S2Q(x));
    ASSERT_OVERFLOW(x, y);  // Infinity

    x = -1.0 / 0.0;
    y = _FANG_Q2S(_FANG_S2Q(x));
    ASSERT_OVERFLOW(x, y);  // -Infinity
}

static void fang_float16_test(void **state) {
    /* Zero. */
    ASSERT_FLOAT16(0.0);

    /* Negative zero. */
    ASSERT_FLOAT16(-0.0);

    /* Smallest subnormal number. */
    ASSERT_FLOAT16(0.000000059604645);
    ASSERT_FLOAT16(-0.000000059604645);

    /* Largest subnormal number. */
    ASSERT_FLOAT16(0.000060975552);
    ASSERT_FLOAT16(-0.000060975552);

    /* Smallest normal number. */
    ASSERT_FLOAT16(0.00006103515625);
    ASSERT_FLOAT16(-0.00006103515625);

    /* Nearest value to 1/3. */
    ASSERT_FLOAT16(0.33325195);
    ASSERT_FLOAT16(-0.33325195);

    /* Largest number smaller than 1. */
    ASSERT_FLOAT16(0.99951172);
    ASSERT_FLOAT16(-0.99951172);

    /* One. */
    ASSERT_FLOAT16(1.0);
    ASSERT_FLOAT16(-1.0);

    /* Smallest number larger than one. */
    ASSERT_FLOAT16(1.00097656);
    ASSERT_FLOAT16(-1.00097656);

    /* Largest number representable. */
    ASSERT_FLOAT16(65504.0);
    ASSERT_FLOAT16(-65504.0);

    /* ==== Test overflow: infinity and NaN ==== */

    float x = 0.0 / 0.0;
    float y = _FANG_H2S(_FANG_S2H(x));
    ASSERT_OVERFLOW(x, y);  // NaN

    x = -0.0 / 0.0;
    y = _FANG_H2S(_FANG_S2H(x));
    ASSERT_OVERFLOW(x, y);  // -NaN

    x = 1.0 / 0.0;
    y = _FANG_H2S(_FANG_S2H(x));
    ASSERT_OVERFLOW(x, y);  // Infinity

    x = -1.0 / 0.0;
    y = _FANG_H2S(_FANG_S2H(x));
    ASSERT_OVERFLOW(x, y);  // -Infinity
}

/* ================ TESTS END ================ */

int main() {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(fang_float8_test),
        cmocka_unit_test(fang_float16_test)
    };

    return cmocka_run_group_tests_name("unit/util/float", tests, NULL, NULL);
}
