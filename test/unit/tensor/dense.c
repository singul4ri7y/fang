#include <fang/env.h>
#include <fang/status.h>
#include <env/cpu/float.h>
#include <stdarg.h>
#include <setjmp.h>
#include <stddef.h>
#include <stdlib.h>
#include <cmocka.h>


/* ================ HELPER MACROS ================ */

/* Array size macro. */
#define _ARR_SIZ(arr)    (sizeof(arr) / sizeof(arr[0]))

/* Check if a tensor initialization is successful or not. */
#define TENCHK(expr)     assert_true(FANG_ISOK(expr))

/* Check whether the flattened tensor data is equal to the data provided. */
/* int16 */
#define ASSERT_TEN_DATA_EQi(ten, eq_data, negate)                               \
for(uint32_t i = 0; i < ten.strides[0] * ten.dims[0]; i++)                      \
    assert_int_equal(((int16_t *) ten.data.dense)[i], negate eq_data[i]);
/* bfloat16 */
#define ASSERT_TEN_DATA_EQbf(ten, eq_data, negate)                              \
for(uint32_t i = 0; i < ten.strides[0] * ten.dims[0]; i++) {                    \
    assert_float_equal(_FANG_BH2S(((_fang_bfloat16_t *) ten.data.dense)[i]),    \
        _FANG_BH2S(_FANG_S2BH(negate eq_data[i])), 1e-6);                       \
}
/* float32 */
#define ASSERT_TEN_DATA_EQf(ten, eq_data, negate)                               \
for(uint32_t i = 0; i < ten.strides[0] * ten.dims[0]; i++) {                    \
    assert_float_equal(((float *) ten.data.dense)[i],                           \
    negate eq_data[i], 1e-6);                                                   \
}

#define _FANG_ARITHMETIC_ERRCHK(op)                                             \
/* Different data types are not allowed. */                                     \
assert_int_equal(fang_ten_##op(&tens->res_8_int16, &tens->ten_8_int16,          \
    &tens->ten_8_float32), -FANG_INVDTYP);                                      \
/* Destination tensor should have valid dimension, conforming to the
   operation. */                                                                \
assert_int_equal(fang_ten_##op(&tens->res_8x8_int16, &tens->ten_3x8x8_int16,    \
    &tens->ten_8x8_int16), -FANG_DESTINVDIM);                                   \
/* Broadcasting should only be possible when either corresponding dimension
   are equal or 1. */                                                           \
assert_int_equal(fang_ten_##op(&tens->res_2x4x8_int16, &tens->ten_8x1_int16,    \
    &tens->ten_2x4x8_int16), -FANG_NOBROAD);

/* ================ HELPER MACROS END ================ */

/* Get the tensor arithmetic operation result data. */
#include "data/dense.c.inc"

/* ================ PRIVATE GLOBALS ================ */



/* ================ PRIVATE GLOBALS END ================ */


/* ================ PRIVATE DATA STRUCTURES ================ */

/* Same, multiple tensors may be needed in several tests. Hence, this structure
   may pass those tensors (provided that the tests ran with appropriate setup()
   and teardown() function) to reduce redundancy. */
typedef struct _fang_ten_tests {
    /* Regular tensors. */
    fang_ten_t ten_8_int16;
    fang_ten_t ten_1_int16;
    fang_ten_t ten_1x1x1x1_int16;
    fang_ten_t ten_8x8_int16;
    fang_ten_t ten_8x1_int16;
    fang_ten_t ten_3x8x8_int16;
    fang_ten_t ten_2x4x8_int16;
    fang_ten_t ten_1x4x1_int16;
    fang_ten_t ten_2x3x4x8_int16;
    fang_ten_t ten_1x3x4x8_int16;
    fang_ten_t ten_1x3x1x8_int16;
    fang_ten_t ten_2x1x4x8_int16;
    fang_ten_t ten_3x1x1x2x1_int16;
    fang_ten_t ten_1x2x3x2x5_int16;

    fang_ten_t ten_8_bfloat16;
    fang_ten_t ten_1_bfloat16;
    fang_ten_t ten_1x1x1x1_bfloat16;
    fang_ten_t ten_8x8_bfloat16;
    fang_ten_t ten_8x1_bfloat16;
    fang_ten_t ten_3x8x8_bfloat16;
    fang_ten_t ten_2x4x8_bfloat16;
    fang_ten_t ten_1x4x1_bfloat16;
    fang_ten_t ten_2x3x4x8_bfloat16;
    fang_ten_t ten_1x3x4x8_bfloat16;
    fang_ten_t ten_1x3x1x8_bfloat16;
    fang_ten_t ten_2x1x4x8_bfloat16;
    fang_ten_t ten_3x1x1x2x1_bfloat16;
    fang_ten_t ten_1x2x3x2x5_bfloat16;

    fang_ten_t ten_8_float32;
    fang_ten_t ten_1_float32;
    fang_ten_t ten_1x1x1x1_float32;
    fang_ten_t ten_8x8_float32;
    fang_ten_t ten_8x1_float32;
    fang_ten_t ten_3x8x8_float32;
    fang_ten_t ten_2x4x8_float32;
    fang_ten_t ten_1x4x1_float32;
    fang_ten_t ten_2x3x4x8_float32;
    fang_ten_t ten_1x3x4x8_float32;
    fang_ten_t ten_1x3x1x8_float32;
    fang_ten_t ten_2x1x4x8_float32;
    fang_ten_t ten_3x1x1x2x1_float32;
    fang_ten_t ten_1x2x3x2x5_float32;

    /* Scalar tensors. */
    fang_ten_t sc1_int16;
    fang_ten_t sc1_bfloat16;
    fang_ten_t sc1_float32;
    fang_ten_t sc2_int16;
    fang_ten_t sc2_bfloat16;
    fang_ten_t sc2_float32;

    /* Resulting tensor shells. */
    fang_ten_t res_8_int16;
    fang_ten_t res_sc_int16;
    fang_ten_t res_1x1x1x8_int16;
    fang_ten_t res_8x8_int16;
    fang_ten_t res_3x8x8_int16;
    fang_ten_t res_2x4x8_int16;
    fang_ten_t res_2x3x4x8_int16;
    fang_ten_t res_3x2x3x2x5_int16;

    fang_ten_t res_8_bfloat16;
    fang_ten_t res_sc_bfloat16;
    fang_ten_t res_8x8_bfloat16;
    fang_ten_t res_1x1x1x8_bfloat16;
    fang_ten_t res_3x8x8_bfloat16;
    fang_ten_t res_2x4x8_bfloat16;
    fang_ten_t res_2x3x4x8_bfloat16;
    fang_ten_t res_3x2x3x2x5_bfloat16;

    fang_ten_t res_8_float32;
    fang_ten_t res_sc_float32;
    fang_ten_t res_1x1x1x8_float32;
    fang_ten_t res_8x8_float32;
    fang_ten_t res_3x8x8_float32;
    fang_ten_t res_2x4x8_float32;
    fang_ten_t res_2x3x4x8_float32;
    fang_ten_t res_3x2x3x2x5_float32;
} _fang_ten_tests_t;

/* ================ PRIVATE DATA STRUCTURES END ================ */


/* ================ SETUP AND TEARDOWN ================ */

/* Setup Environment before every test. */
static int setup(void **state) {
    int env = fang_env_create(FANG_ENV_TYPE_CPU, NULL);
    if(!FANG_ISOK(env))
        return 1;

    *state = (void *) (uint64_t) env;
    return 0;
}

/* Release created Environment after every test. */
static int teardown(void **state) {
    int env = (int) (uint64_t) *state;
    fang_env_release(env);
    return 0;
}

/* Setup for arithmetic tests, where assortment of tensors are required. */
static int setup_arithmetic(void **state) {
    /* The environment already created in grouped setup. */
    int env = (int) (uint64_t) *state;

    /* It is a good idea to fill the tensors with natural numbers. */
    int _DATA_THRESHOLD = 256;
    fang_int_t data_int[_DATA_THRESHOLD];
    fang_float_t data_float[_DATA_THRESHOLD];

    for(int i = 0; i < _DATA_THRESHOLD; i++) {
        data_int[i] = i + 1;
        data_float[i] = (fang_float_t) (i + 1);
    }

    /* Create tensor assortment structure. */
    _fang_ten_tests_t *tens = malloc(sizeof(_fang_ten_tests_t));

    /* Initialize the scalar tensors. */
    TENCHK(fang_ten_scalar(&tens->sc1_int16, env, FANG_TEN_DTYPE_INT16,
        FANG_I2G(48)));
    TENCHK(fang_ten_scalar(&tens->sc1_bfloat16, env, FANG_TEN_DTYPE_BFLOAT16,
        FANG_F2G(48)));
    TENCHK(fang_ten_scalar(&tens->sc1_float32, env, FANG_TEN_DTYPE_FLOAT32,
        FANG_F2G(48)));
    /* `sc2` */
    TENCHK(fang_ten_scalar(&tens->sc2_int16, env, FANG_TEN_DTYPE_INT16,
        FANG_I2G(33)));
    TENCHK(fang_ten_scalar(&tens->sc2_bfloat16, env, FANG_TEN_DTYPE_BFLOAT16,
        FANG_F2G(33)));
    TENCHK(fang_ten_scalar(&tens->sc2_float32, env, FANG_TEN_DTYPE_FLOAT32,
        FANG_F2G(33)));

    /* Initialize regular tensors. */
    /* (8) */
    TENCHK(fang_ten_create(&tens->ten_8_int16, env,
        FANG_TEN_DTYPE_INT16, (uint32_t[]) { 8 }, 1, data_int));
    TENCHK(fang_ten_create(&tens->ten_8_bfloat16, env,
        FANG_TEN_DTYPE_BFLOAT16, (uint32_t[]) { 8 }, 1, data_float));
    TENCHK(fang_ten_create(&tens->ten_8_float32, env,
        FANG_TEN_DTYPE_FLOAT32, (uint32_t[]) { 8 }, 1, data_float));

    /* (1) */
    TENCHK(fang_ten_create(&tens->ten_1_int16, env,
        FANG_TEN_DTYPE_INT16, (uint32_t[]) { 1 }, 1, data_int));
    TENCHK(fang_ten_create(&tens->ten_1_bfloat16, env,
        FANG_TEN_DTYPE_BFLOAT16, (uint32_t[]) { 1 }, 1, data_float));
    TENCHK(fang_ten_create(&tens->ten_1_float32, env,
        FANG_TEN_DTYPE_FLOAT32, (uint32_t[]) { 1 }, 1, data_float));

    /* (1, 1, 1, 1) */
    TENCHK(fang_ten_create(&tens->ten_1x1x1x1_int16, env,
        FANG_TEN_DTYPE_INT16, (uint32_t[]) { 1, 1, 1, 1 }, 4, data_int));
    TENCHK(fang_ten_create(&tens->ten_1x1x1x1_bfloat16, env,
        FANG_TEN_DTYPE_BFLOAT16, (uint32_t[]) { 1, 1, 1, 1 }, 4, data_float));
    TENCHK(fang_ten_create(&tens->ten_1x1x1x1_float32, env,
        FANG_TEN_DTYPE_FLOAT32, (uint32_t[]) { 1, 1, 1, 1 }, 4, data_float));

    /* (8, 8) */
    TENCHK(fang_ten_create(&tens->ten_8x8_int16, env,
        FANG_TEN_DTYPE_INT16, (uint32_t[]) { 8, 8 }, 2, data_int));
    TENCHK(fang_ten_create(&tens->ten_8x8_bfloat16, env,
        FANG_TEN_DTYPE_BFLOAT16, (uint32_t[]) { 8, 8 }, 2, data_float));
    TENCHK(fang_ten_create(&tens->ten_8x8_float32, env,
        FANG_TEN_DTYPE_FLOAT32, (uint32_t[]) { 8, 8 }, 2, data_float));

    /* (8, 1) */
    TENCHK(fang_ten_create(&tens->ten_8x1_int16, env,
        FANG_TEN_DTYPE_INT16, (uint32_t[]) { 8, 1 }, 2, data_int));
    TENCHK(fang_ten_create(&tens->ten_8x1_bfloat16, env,
        FANG_TEN_DTYPE_BFLOAT16, (uint32_t[]) { 8, 1 }, 2, data_float));
    TENCHK(fang_ten_create(&tens->ten_8x1_float32, env,
        FANG_TEN_DTYPE_FLOAT32, (uint32_t[]) { 8, 1 }, 2, data_float));

    /* (3, 8, 8) */
    TENCHK(fang_ten_create(&tens->ten_3x8x8_int16, env,
        FANG_TEN_DTYPE_INT16, (uint32_t[]) { 3, 8, 8 }, 3, data_int));
    TENCHK(fang_ten_create(&tens->ten_3x8x8_bfloat16, env,
        FANG_TEN_DTYPE_BFLOAT16, (uint32_t[]) { 3, 8, 8 }, 3, data_float));
    TENCHK(fang_ten_create(&tens->ten_3x8x8_float32, env,
        FANG_TEN_DTYPE_FLOAT32, (uint32_t[]) { 3, 8, 8 }, 3, data_float));

    /* (2, 4, 8) */
    TENCHK(fang_ten_create(&tens->ten_2x4x8_int16, env,
        FANG_TEN_DTYPE_INT16, (uint32_t[]) { 2, 4, 8 }, 3, data_int));
    TENCHK(fang_ten_create(&tens->ten_2x4x8_bfloat16, env,
        FANG_TEN_DTYPE_BFLOAT16, (uint32_t[]) { 2, 4, 8 }, 3, data_float));
    TENCHK(fang_ten_create(&tens->ten_2x4x8_float32, env,
        FANG_TEN_DTYPE_FLOAT32, (uint32_t[]) { 2, 4, 8 }, 3, data_float));

    /* (1, 4, 1) */
    TENCHK(fang_ten_create(&tens->ten_1x4x1_int16, env,
        FANG_TEN_DTYPE_INT16, (uint32_t[]) { 1, 4, 1 }, 3, data_int));
    TENCHK(fang_ten_create(&tens->ten_1x4x1_bfloat16, env,
        FANG_TEN_DTYPE_BFLOAT16, (uint32_t[]) { 1, 4, 1 }, 3, data_float));
    TENCHK(fang_ten_create(&tens->ten_1x4x1_float32, env,
        FANG_TEN_DTYPE_FLOAT32, (uint32_t[]) { 1, 4, 1 }, 3, data_float));

    /* (2, 3, 4, 8) */
    TENCHK(fang_ten_create(&tens->ten_2x3x4x8_int16, env,
        FANG_TEN_DTYPE_INT16, (uint32_t[]) { 2, 3, 4, 8 }, 4, data_int));
    TENCHK(fang_ten_create(&tens->ten_2x3x4x8_bfloat16, env,
        FANG_TEN_DTYPE_BFLOAT16, (uint32_t[]) { 2, 3, 4, 8 }, 4, data_float));
    TENCHK(fang_ten_create(&tens->ten_2x3x4x8_float32, env,
        FANG_TEN_DTYPE_FLOAT32, (uint32_t[]) { 2, 3, 4, 8 }, 4, data_float));

    /* (1, 3, 4, 8) */
    TENCHK(fang_ten_create(&tens->ten_1x3x4x8_int16, env,
        FANG_TEN_DTYPE_INT16, (uint32_t[]) { 1, 3, 4, 8 }, 4, data_int));
    TENCHK(fang_ten_create(&tens->ten_1x3x4x8_bfloat16, env,
        FANG_TEN_DTYPE_BFLOAT16, (uint32_t[]) { 1, 3, 4, 8 }, 4, data_float));
    TENCHK(fang_ten_create(&tens->ten_1x3x4x8_float32, env,
        FANG_TEN_DTYPE_FLOAT32, (uint32_t[]) { 1, 3, 4, 8 }, 4, data_float));

    /* (1, 3, 1, 8) */
    TENCHK(fang_ten_create(&tens->ten_1x3x1x8_int16, env,
        FANG_TEN_DTYPE_INT16, (uint32_t[]) { 1, 3, 1, 8 }, 4, data_int));
    TENCHK(fang_ten_create(&tens->ten_1x3x1x8_bfloat16, env,
        FANG_TEN_DTYPE_BFLOAT16, (uint32_t[]) { 1, 3, 1, 8 }, 4, data_float));
    TENCHK(fang_ten_create(&tens->ten_1x3x1x8_float32, env,
        FANG_TEN_DTYPE_FLOAT32, (uint32_t[]) { 1, 3, 1, 8 }, 4, data_float));

    /* (2, 1, 4, 8) */
    TENCHK(fang_ten_create(&tens->ten_2x1x4x8_int16, env,
        FANG_TEN_DTYPE_INT16, (uint32_t[]) { 2, 1, 4, 8 }, 4, data_int));
    TENCHK(fang_ten_create(&tens->ten_2x1x4x8_bfloat16, env,
        FANG_TEN_DTYPE_BFLOAT16, (uint32_t[]) { 2, 1, 4, 8 }, 4, data_float));
    TENCHK(fang_ten_create(&tens->ten_2x1x4x8_float32, env,
        FANG_TEN_DTYPE_FLOAT32, (uint32_t[]) { 2, 1, 4, 8 }, 4, data_float));

    /* (3, 1, 1, 2, 1) */
    TENCHK(fang_ten_create(&tens->ten_3x1x1x2x1_int16, env,
        FANG_TEN_DTYPE_INT16, (uint32_t[]) { 3, 1, 1, 2, 1 }, 5,
        data_int));
    TENCHK(fang_ten_create(&tens->ten_3x1x1x2x1_bfloat16, env,
        FANG_TEN_DTYPE_BFLOAT16, (uint32_t[]) { 3, 1, 1, 2, 1 }, 5,
        data_float));
    TENCHK(fang_ten_create(&tens->ten_3x1x1x2x1_float32, env,
        FANG_TEN_DTYPE_FLOAT32, (uint32_t[]) { 3, 1, 1, 2, 1 }, 5,
        data_float));

    /* (1, 2, 3, 2, 5) */
    TENCHK(fang_ten_create(&tens->ten_1x2x3x2x5_int16, env,
        FANG_TEN_DTYPE_INT16, (uint32_t[]) { 1, 2, 3, 2, 5 }, 5,
        data_int));
    TENCHK(fang_ten_create(&tens->ten_1x2x3x2x5_bfloat16, env,
        FANG_TEN_DTYPE_BFLOAT16, (uint32_t[]) { 1, 2, 3, 2, 5 }, 5,
        data_float));
    TENCHK(fang_ten_create(&tens->ten_1x2x3x2x5_float32, env,
        FANG_TEN_DTYPE_FLOAT32, (uint32_t[]) { 1, 2, 3, 2, 5 }, 5,
        data_float));

    /* Resulting tensors. */
    /* (8) */
    TENCHK(fang_ten_create(&tens->res_8_int16, env,
        FANG_TEN_DTYPE_INT16, (uint32_t[]) { 8 }, 1,
        data_int));
    TENCHK(fang_ten_create(&tens->res_8_bfloat16, env,
        FANG_TEN_DTYPE_BFLOAT16, (uint32_t[]) { 8 }, 1,
        data_float));
    TENCHK(fang_ten_create(&tens->res_8_float32, env,
        FANG_TEN_DTYPE_FLOAT32, (uint32_t[]) { 8 }, 1,
        data_float));

    /* () */
    TENCHK(fang_ten_scalar(&tens->res_sc_int16, env,
        FANG_TEN_DTYPE_INT16, NULL));
    TENCHK(fang_ten_scalar(&tens->res_sc_bfloat16, env,
        FANG_TEN_DTYPE_BFLOAT16, NULL));
    TENCHK(fang_ten_scalar(&tens->res_sc_float32, env,
        FANG_TEN_DTYPE_FLOAT32, NULL));

    /* (8, 8) */
    TENCHK(fang_ten_create(&tens->res_8x8_int16, env,
        FANG_TEN_DTYPE_INT16, (uint32_t[]) { 8, 8 }, 2,
        data_int));
    TENCHK(fang_ten_create(&tens->res_8x8_bfloat16, env,
        FANG_TEN_DTYPE_BFLOAT16, (uint32_t[]) { 8, 8 }, 2,
        data_float));
    TENCHK(fang_ten_create(&tens->res_8x8_float32, env,
        FANG_TEN_DTYPE_FLOAT32, (uint32_t[]) { 8, 8 }, 2,
        data_float));

    /* (1, 1, 1, 8) */
    TENCHK(fang_ten_create(&tens->res_1x1x1x8_int16, env,
        FANG_TEN_DTYPE_INT16, (uint32_t[]) { 1, 1, 1, 8 }, 4,
        data_int));
    TENCHK(fang_ten_create(&tens->res_1x1x1x8_bfloat16, env,
        FANG_TEN_DTYPE_BFLOAT16, (uint32_t[]) { 1, 1, 1, 8 }, 4,
        data_float));
    TENCHK(fang_ten_create(&tens->res_1x1x1x8_float32, env,
        FANG_TEN_DTYPE_FLOAT32, (uint32_t[]) { 1, 1, 1, 8 }, 4,
        data_float));

    /* (3, 8, 8) */
    TENCHK(fang_ten_create(&tens->res_3x8x8_int16, env,
        FANG_TEN_DTYPE_INT16, (uint32_t[]) { 3, 8, 8 }, 3,
        data_int));
    TENCHK(fang_ten_create(&tens->res_3x8x8_bfloat16, env,
        FANG_TEN_DTYPE_BFLOAT16, (uint32_t[]) { 3, 8, 8 }, 3,
        data_float));
    TENCHK(fang_ten_create(&tens->res_3x8x8_float32, env,
        FANG_TEN_DTYPE_FLOAT32, (uint32_t[]) { 3, 8, 8 }, 3,
        data_float));

    /* (2, 4, 8) */
    TENCHK(fang_ten_create(&tens->res_2x4x8_int16, env,
        FANG_TEN_DTYPE_INT16, (uint32_t[]) { 2, 4, 8 }, 3,
        data_int));
    TENCHK(fang_ten_create(&tens->res_2x4x8_bfloat16, env,
        FANG_TEN_DTYPE_BFLOAT16, (uint32_t[]) { 2, 4, 8 }, 3,
        data_float));
    TENCHK(fang_ten_create(&tens->res_2x4x8_float32, env,
        FANG_TEN_DTYPE_FLOAT32, (uint32_t[]) { 2, 4, 8 }, 3,
        data_float));

    /* (2, 3, 4, 8) */
    TENCHK(fang_ten_create(&tens->res_2x3x4x8_int16, env,
        FANG_TEN_DTYPE_INT16, (uint32_t[]) { 2, 3, 4, 8 }, 4,
        data_int));
    TENCHK(fang_ten_create(&tens->res_2x3x4x8_bfloat16, env,
        FANG_TEN_DTYPE_BFLOAT16, (uint32_t[]) { 2, 3, 4, 8 }, 4,
        data_float));
    TENCHK(fang_ten_create(&tens->res_2x3x4x8_float32, env,
        FANG_TEN_DTYPE_FLOAT32, (uint32_t[]) { 2, 3, 4, 8 }, 4,
        data_float));

    /* (3, 2, 3, 2, 5) */
    TENCHK(fang_ten_create(&tens->res_3x2x3x2x5_int16, env,
        FANG_TEN_DTYPE_INT16, (uint32_t[]) { 3, 2, 3, 2, 5 }, 5,
        data_int));
    TENCHK(fang_ten_create(&tens->res_3x2x3x2x5_bfloat16, env,
        FANG_TEN_DTYPE_BFLOAT16, (uint32_t[]) { 3, 2, 3, 2, 5 }, 5,
        data_float));
    TENCHK(fang_ten_create(&tens->res_3x2x3x2x5_float32, env,
        FANG_TEN_DTYPE_FLOAT32, (uint32_t[]) { 3, 2, 3, 2, 5 }, 5,
        data_float));

    *state = (void *) tens;

    return 0;
}

/* Teardown for arithmetic tests. */
static int teardown_arithmetic(void **state) {
    /* Lazy and kinda risky way to release the tensors. The `_fang_ten_tests`
     * structure is full of `fang_ten_t` tensor structures. */
    fang_ten_t *tens = (fang_ten_t *) *state;
    int siz = sizeof(_fang_ten_tests_t) / sizeof(fang_ten_t);

    for(int i = 0; i < siz; i++)
        TENCHK(fang_ten_release(tens + i));

    /* Free the test tensor structure. */
    free(tens);
    return 0;
}

/* ================ SETUP AND TEARDOWN END ================ */


/* ================ TESTS ================ */

/* Tensor creation test. */
static void fang_ten_create_test(void **state) {
    /* Get the Environment. */
    int env = (int) (uint64_t) *state;

    uint32_t dims[] = { 3, 4, 7, 2, 8, 5 };
    fang_ten_t ten;

    /* Tensor creation should be successful. */
    TENCHK(fang_ten_create(&ten, env, FANG_TEN_DTYPE_FLOAT16, dims,
        _ARR_SIZ(dims), NULL));

    /* Environment ID should match. */
    assert_int_equal(ten.eid, env);

    /* Dimension count should match. */
    assert_int_equal(ten.ndims, _ARR_SIZ(dims));

    /* Dimension and strides check. */
    uint32_t stride = 1;
    for(int i = 0; i < ten.ndims; i++) {
        /* Dimension should match. */
        assert_int_equal(ten.dims[i], dims[i]);

        /* Stride should match. */
        assert_int_equal(ten.strides[ten.ndims - (i + 1)], stride);
        stride *= ten.dims[ten.ndims - (i + 1)];
    }

    /* Tensor type should be dense. */
    assert_int_equal(ten.typ, FANG_TEN_TYPE_DENSE);

    /* Tensor data type should be float16. */
    assert_int_equal(ten.dtyp, FANG_TEN_DTYPE_FLOAT16);

    /* Number of elements in tensor. */
    int size = ten.strides[0] * ten.dims[0];

    /* Data should be zero-ed out. */
    _fang_float16_t *fdata = ten.data.dense;
    for(int i = 0; i < size; i++)
        assert_float_equal(0.0, _FANG_H2S(fdata[i]), 1e-6);

    fang_int_t input_data[size];
    for(int i = 0; i < size; i++)
        input_data[i] = i - 3360;

    /* Release previous tensor. */
    fang_ten_release(&ten);

    /* Create a new tensor with data. */
    TENCHK(fang_ten_create(&ten, env, FANG_TEN_DTYPE_INT8, dims,
        _ARR_SIZ(dims), input_data));

    /* Check data initialization. */
    int8_t *idata = ten.data.dense;
    for(int i = 0; i < size; i++)
        assert_int_equal(idata[i], (int8_t) input_data[i]);

    /* Test scalar tensor creation. */
    fang_ten_t scalar;
    TENCHK(fang_ten_scalar(&scalar, env, FANG_TEN_DTYPE_INT8, FANG_I2G(69)));
    /* Scalar tensor datum is stored as single element 1-d tensor data. */
    assert_int_equal(69, ((int8_t *) scalar.data.dense)[0]);

    fang_ten_release(&scalar);
    fang_ten_release(&ten);
}

/* Tensor summation test. */
static void fang_ten_sum_test(void **state) {
    /* Get test tensors. */
    _fang_ten_tests_t *tens = (_fang_ten_tests_t *) *state;

    _FANG_ARITHMETIC_ERRCHK(sum);

    /* (8) + (8) */
    TENCHK(fang_ten_sum(&tens->res_8_int16, &tens->ten_8_int16,
        &tens->ten_8_int16));
    TENCHK(fang_ten_sum(&tens->res_8_bfloat16, &tens->ten_8_bfloat16,
        &tens->ten_8_bfloat16));
    TENCHK(fang_ten_sum(&tens->res_8_float32, &tens->ten_8_float32,
        &tens->ten_8_float32));
    ASSERT_TEN_DATA_EQi(tens->res_8_int16, ten_sum_result1_int16,);
    ASSERT_TEN_DATA_EQbf(tens->res_8_bfloat16, ten_sum_result1_float32,);
    ASSERT_TEN_DATA_EQf(tens->res_8_float32, ten_sum_result1_float32,);

    /* () + () */
    TENCHK(fang_ten_sum(&tens->res_sc_int16, &tens->sc1_int16,
        &tens->sc2_int16));
    TENCHK(fang_ten_sum(&tens->res_sc_bfloat16, &tens->sc1_bfloat16,
        &tens->sc2_bfloat16));
    TENCHK(fang_ten_sum(&tens->res_sc_float32, &tens->sc1_float32,
        &tens->sc2_float32));
    assert_int_equal(*(int16_t *) tens->res_sc_int16.data.dense,
        ten_sum_result2_int16[0]);
    assert_float_equal(_FANG_BH2S(*(_fang_bfloat16_t *)
        tens->res_sc_bfloat16.data.dense),
        _FANG_BH2S(_FANG_S2BH(ten_sum_result2_float32[0])), 1e-6);
    assert_float_equal(*(float *) tens->res_sc_float32.data.dense,
        ten_sum_result2_float32[0], 1e-6);

    /* (8) + () */
    TENCHK(fang_ten_sum(&tens->res_8_int16, &tens->ten_8_int16,
        &tens->sc1_int16));
    TENCHK(fang_ten_sum(&tens->res_8_bfloat16, &tens->ten_8_bfloat16,
        &tens->sc1_bfloat16));
    TENCHK(fang_ten_sum(&tens->res_8_float32, &tens->ten_8_float32,
        &tens->sc1_float32));
    ASSERT_TEN_DATA_EQi(tens->res_8_int16, ten_sum_result3_int16,);
    ASSERT_TEN_DATA_EQbf(tens->res_8_bfloat16, ten_sum_result3_float32,);
    ASSERT_TEN_DATA_EQf(tens->res_8_float32, ten_sum_result3_float32,);

    /* (8) + (1) */
    /* NOTE: A single element 1-dimensional tensor can be thought of as a
       scalar tensor. */
    TENCHK(fang_ten_sum(&tens->res_8_int16, &tens->ten_8_int16,
        &tens->ten_1_int16));
    TENCHK(fang_ten_sum(&tens->res_8_bfloat16, &tens->ten_8_bfloat16,
        &tens->ten_1_bfloat16));
    TENCHK(fang_ten_sum(&tens->res_8_float32, &tens->ten_8_float32,
        &tens->ten_1_float32));
    ASSERT_TEN_DATA_EQi(tens->res_8_int16, ten_sum_result4_int16,);
    ASSERT_TEN_DATA_EQbf(tens->res_8_bfloat16, ten_sum_result4_float32,);
    ASSERT_TEN_DATA_EQf(tens->res_8_float32, ten_sum_result4_float32,);

    /* (8) + (1, 1, 1, 1) */
    TENCHK(fang_ten_sum(&tens->res_1x1x1x8_int16, &tens->ten_8_int16,
        &tens->ten_1x1x1x1_int16));
    TENCHK(fang_ten_sum(&tens->res_1x1x1x8_bfloat16, &tens->ten_8_bfloat16,
        &tens->ten_1x1x1x1_bfloat16));
    TENCHK(fang_ten_sum(&tens->res_1x1x1x8_float32, &tens->ten_8_float32,
        &tens->ten_1x1x1x1_float32));
    ASSERT_TEN_DATA_EQi(tens->res_1x1x1x8_int16, ten_sum_result4_int16,);
    ASSERT_TEN_DATA_EQbf(tens->res_1x1x1x8_bfloat16, ten_sum_result4_float32,);
    ASSERT_TEN_DATA_EQf(tens->res_1x1x1x8_float32, ten_sum_result4_float32,);

    /* (8, 8) + (8) */
    TENCHK(fang_ten_sum(&tens->res_8x8_int16, &tens->ten_8x8_int16,
        &tens->ten_8_int16));
    TENCHK(fang_ten_sum(&tens->res_8x8_bfloat16, &tens->ten_8x8_bfloat16,
        &tens->ten_8_bfloat16));
    TENCHK(fang_ten_sum(&tens->res_8x8_float32, &tens->ten_8x8_float32,
        &tens->ten_8_float32));
    ASSERT_TEN_DATA_EQi(tens->res_8x8_int16, ten_sum_result5_int16,);
    ASSERT_TEN_DATA_EQbf(tens->res_8x8_bfloat16, ten_sum_result5_float32,);
    ASSERT_TEN_DATA_EQf(tens->res_8x8_float32, ten_sum_result5_float32,);

    /* (8, 8) + (8, 1) */
    TENCHK(fang_ten_sum(&tens->res_8x8_int16, &tens->ten_8x8_int16,
        &tens->ten_8x1_int16));
    TENCHK(fang_ten_sum(&tens->res_8x8_bfloat16, &tens->ten_8x8_bfloat16,
        &tens->ten_8x1_bfloat16));
    TENCHK(fang_ten_sum(&tens->res_8x8_float32, &tens->ten_8x8_float32,
        &tens->ten_8x1_float32));
    ASSERT_TEN_DATA_EQi(tens->res_8x8_int16, ten_sum_result6_int16,);
    ASSERT_TEN_DATA_EQbf(tens->res_8x8_bfloat16, ten_sum_result6_float32,);
    ASSERT_TEN_DATA_EQf(tens->res_8x8_float32, ten_sum_result6_float32,);

    /* (3, 8, 8) + (8, 8) */
    TENCHK(fang_ten_sum(&tens->res_3x8x8_int16, &tens->ten_3x8x8_int16,
        &tens->ten_8x8_int16));
    TENCHK(fang_ten_sum(&tens->res_3x8x8_bfloat16, &tens->ten_3x8x8_bfloat16,
        &tens->ten_8x8_bfloat16));
    TENCHK(fang_ten_sum(&tens->res_3x8x8_float32, &tens->ten_3x8x8_float32,
        &tens->ten_8x8_float32));
    ASSERT_TEN_DATA_EQi(tens->res_3x8x8_int16, ten_sum_result7_int16,);
    ASSERT_TEN_DATA_EQbf(tens->res_3x8x8_bfloat16, ten_sum_result7_float32,);
    ASSERT_TEN_DATA_EQf(tens->res_3x8x8_float32, ten_sum_result7_float32,);

    /* (2, 4, 8) + (1, 4, 1) */
    TENCHK(fang_ten_sum(&tens->res_2x4x8_int16, &tens->ten_2x4x8_int16,
        &tens->ten_1x4x1_int16));
    TENCHK(fang_ten_sum(&tens->res_2x4x8_bfloat16, &tens->ten_2x4x8_bfloat16,
        &tens->ten_1x4x1_bfloat16));
    TENCHK(fang_ten_sum(&tens->res_2x4x8_float32, &tens->ten_2x4x8_float32,
        &tens->ten_1x4x1_float32));
    ASSERT_TEN_DATA_EQi(tens->res_2x4x8_int16, ten_sum_result8_int16,);
    ASSERT_TEN_DATA_EQbf(tens->res_2x4x8_bfloat16, ten_sum_result8_float32,);
    ASSERT_TEN_DATA_EQf(tens->res_2x4x8_float32, ten_sum_result8_float32,);

    /* (2, 3, 4, 8) + (1, 3, 4, 8) */
    TENCHK(fang_ten_sum(&tens->res_2x3x4x8_int16, &tens->ten_2x3x4x8_int16,
        &tens->ten_1x3x4x8_int16));
    TENCHK(fang_ten_sum(&tens->res_2x3x4x8_bfloat16,
        &tens->ten_2x3x4x8_bfloat16, &tens->ten_1x3x4x8_bfloat16));
    TENCHK(fang_ten_sum(&tens->res_2x3x4x8_float32, &tens->ten_2x3x4x8_float32,
        &tens->ten_1x3x4x8_float32));
    ASSERT_TEN_DATA_EQi(tens->res_2x3x4x8_int16, ten_sum_result9_int16,);
    ASSERT_TEN_DATA_EQbf(tens->res_2x3x4x8_bfloat16, ten_sum_result9_float32,);
    ASSERT_TEN_DATA_EQf(tens->res_2x3x4x8_float32, ten_sum_result9_float32,);

    /* (2, 3, 4, 8) + (1, 3, 1, 8) */
    TENCHK(fang_ten_sum(&tens->res_2x3x4x8_int16, &tens->ten_2x3x4x8_int16,
        &tens->ten_1x3x1x8_int16));
    TENCHK(fang_ten_sum(&tens->res_2x3x4x8_bfloat16,
        &tens->ten_2x3x4x8_bfloat16, &tens->ten_1x3x1x8_bfloat16));
    TENCHK(fang_ten_sum(&tens->res_2x3x4x8_float32, &tens->ten_2x3x4x8_float32,
        &tens->ten_1x3x1x8_float32));
    ASSERT_TEN_DATA_EQi(tens->res_2x3x4x8_int16, ten_sum_result10_int16,);
    ASSERT_TEN_DATA_EQbf(tens->res_2x3x4x8_bfloat16, ten_sum_result10_float32,);
    ASSERT_TEN_DATA_EQf(tens->res_2x3x4x8_float32, ten_sum_result10_float32,);

    /* (2, 1, 4, 8) + (1, 3, 4, 8) */
    TENCHK(fang_ten_sum(&tens->res_2x3x4x8_int16, &tens->ten_2x1x4x8_int16,
        &tens->ten_1x3x4x8_int16));
    TENCHK(fang_ten_sum(&tens->res_2x3x4x8_bfloat16,
        &tens->ten_2x1x4x8_bfloat16, &tens->ten_1x3x4x8_bfloat16));
    TENCHK(fang_ten_sum(&tens->res_2x3x4x8_float32, &tens->ten_2x1x4x8_float32,
        &tens->ten_1x3x4x8_float32));
    ASSERT_TEN_DATA_EQi(tens->res_2x3x4x8_int16, ten_sum_result11_int16,);
    ASSERT_TEN_DATA_EQbf(tens->res_2x3x4x8_bfloat16, ten_sum_result11_float32,);
    ASSERT_TEN_DATA_EQf(tens->res_2x3x4x8_float32, ten_sum_result11_float32,);

    /* (2, 1, 4, 8) + (1, 3, 1, 8) */
    TENCHK(fang_ten_sum(&tens->res_2x3x4x8_int16, &tens->ten_2x1x4x8_int16,
        &tens->ten_1x3x1x8_int16));
    TENCHK(fang_ten_sum(&tens->res_2x3x4x8_bfloat16,
        &tens->ten_2x1x4x8_bfloat16, &tens->ten_1x3x1x8_bfloat16));
    TENCHK(fang_ten_sum(&tens->res_2x3x4x8_float32, &tens->ten_2x1x4x8_float32,
        &tens->ten_1x3x1x8_float32));
    ASSERT_TEN_DATA_EQi(tens->res_2x3x4x8_int16, ten_sum_result12_int16,);
    ASSERT_TEN_DATA_EQbf(tens->res_2x3x4x8_bfloat16, ten_sum_result12_float32,);
    ASSERT_TEN_DATA_EQf(tens->res_2x3x4x8_float32, ten_sum_result12_float32,);

    /* () + (3, 8, 8) */
    TENCHK(fang_ten_sum(&tens->res_3x8x8_int16, &tens->sc1_int16,
        &tens->ten_3x8x8_int16));
    TENCHK(fang_ten_sum(&tens->res_3x8x8_bfloat16, &tens->sc1_bfloat16,
        &tens->ten_3x8x8_bfloat16));
    TENCHK(fang_ten_sum(&tens->res_3x8x8_float32, &tens->sc1_float32,
        &tens->ten_3x8x8_float32));
    ASSERT_TEN_DATA_EQi(tens->res_3x8x8_int16, ten_sum_result13_int16,);
    ASSERT_TEN_DATA_EQbf(tens->res_3x8x8_bfloat16, ten_sum_result13_float32,);
    ASSERT_TEN_DATA_EQf(tens->res_3x8x8_float32, ten_sum_result13_float32,);

    /* (3, 1, 1, 2, 1) + (1, 2, 3, 2, 5) */
    TENCHK(fang_ten_sum(&tens->res_3x2x3x2x5_int16, &tens->ten_3x1x1x2x1_int16,
        &tens->ten_1x2x3x2x5_int16));
    TENCHK(fang_ten_sum(&tens->res_3x2x3x2x5_bfloat16,
        &tens->ten_3x1x1x2x1_bfloat16, &tens->ten_1x2x3x2x5_bfloat16));
    TENCHK(fang_ten_sum(&tens->res_3x2x3x2x5_float32,
        &tens->ten_3x1x1x2x1_float32, &tens->ten_1x2x3x2x5_float32));
    ASSERT_TEN_DATA_EQi(tens->res_3x2x3x2x5_int16,
        ten_sum_result14_int16,);
    ASSERT_TEN_DATA_EQbf(tens->res_3x2x3x2x5_bfloat16,
        ten_sum_result14_float32,);
    ASSERT_TEN_DATA_EQf(tens->res_3x2x3x2x5_float32,
        ten_sum_result14_float32,);
}

/* Tensor multiplication test. */
static void fang_ten_mul_test(void **state) {
    /* Get test tensors. */
    _fang_ten_tests_t *tens = (_fang_ten_tests_t *) *state;

    _FANG_ARITHMETIC_ERRCHK(mul);

    /* (8) * (8) */
    TENCHK(fang_ten_mul(&tens->res_8_int16, &tens->ten_8_int16,
        &tens->ten_8_int16));
    TENCHK(fang_ten_mul(&tens->res_8_bfloat16, &tens->ten_8_bfloat16,
        &tens->ten_8_bfloat16));
    TENCHK(fang_ten_mul(&tens->res_8_float32, &tens->ten_8_float32,
        &tens->ten_8_float32));
    ASSERT_TEN_DATA_EQi(tens->res_8_int16, ten_mul_result1_int16,);
    ASSERT_TEN_DATA_EQbf(tens->res_8_bfloat16, ten_mul_result1_float32,);
    ASSERT_TEN_DATA_EQf(tens->res_8_float32, ten_mul_result1_float32,);

    /* () * () */
    TENCHK(fang_ten_mul(&tens->res_sc_int16, &tens->sc1_int16,
        &tens->sc2_int16));
    TENCHK(fang_ten_mul(&tens->res_sc_bfloat16, &tens->sc1_bfloat16,
        &tens->sc2_bfloat16));
    TENCHK(fang_ten_mul(&tens->res_sc_float32, &tens->sc1_float32,
        &tens->sc2_float32));
    assert_int_equal(*(int16_t *) tens->res_sc_int16.data.dense,
        ten_mul_result2_int16[0]);
    assert_float_equal(_FANG_BH2S(*(_fang_bfloat16_t *)
        tens->res_sc_bfloat16.data.dense),
        _FANG_BH2S(_FANG_S2BH(ten_mul_result2_float32[0])), 1e-6);
    assert_float_equal(*(float *) tens->res_sc_float32.data.dense,
        ten_mul_result2_float32[0], 1e-6);

    /* (8) * () */
    TENCHK(fang_ten_mul(&tens->res_8_int16, &tens->ten_8_int16,
        &tens->sc1_int16));
    TENCHK(fang_ten_mul(&tens->res_8_bfloat16, &tens->ten_8_bfloat16,
        &tens->sc1_bfloat16));
    TENCHK(fang_ten_mul(&tens->res_8_float32, &tens->ten_8_float32,
        &tens->sc1_float32));
    ASSERT_TEN_DATA_EQi(tens->res_8_int16, ten_mul_result3_int16,);
    ASSERT_TEN_DATA_EQbf(tens->res_8_bfloat16, ten_mul_result3_float32,);
    ASSERT_TEN_DATA_EQf(tens->res_8_float32, ten_mul_result3_float32,);

    /* (8) * (1) */
    /* NOTE: A single element 1-dimensional tensor can be thought of as a
       scalar tensor. */
    TENCHK(fang_ten_mul(&tens->res_8_int16, &tens->ten_8_int16,
        &tens->ten_1_int16));
    TENCHK(fang_ten_mul(&tens->res_8_bfloat16, &tens->ten_8_bfloat16,
        &tens->ten_1_bfloat16));
    TENCHK(fang_ten_mul(&tens->res_8_float32, &tens->ten_8_float32,
        &tens->ten_1_float32));
    ASSERT_TEN_DATA_EQi(tens->res_8_int16, ten_mul_result4_int16,);
    ASSERT_TEN_DATA_EQbf(tens->res_8_bfloat16, ten_mul_result4_float32,);
    ASSERT_TEN_DATA_EQf(tens->res_8_float32, ten_mul_result4_float32,);

    /* (8) * (1, 1, 1, 1) */
    TENCHK(fang_ten_mul(&tens->res_1x1x1x8_int16, &tens->ten_8_int16,
        &tens->ten_1x1x1x1_int16));
    TENCHK(fang_ten_mul(&tens->res_1x1x1x8_bfloat16, &tens->ten_8_bfloat16,
        &tens->ten_1x1x1x1_bfloat16));
    TENCHK(fang_ten_mul(&tens->res_1x1x1x8_float32, &tens->ten_8_float32,
        &tens->ten_1x1x1x1_float32));
    ASSERT_TEN_DATA_EQi(tens->res_1x1x1x8_int16, ten_mul_result4_int16,);
    ASSERT_TEN_DATA_EQbf(tens->res_1x1x1x8_bfloat16, ten_mul_result4_float32,);
    ASSERT_TEN_DATA_EQf(tens->res_1x1x1x8_float32, ten_mul_result4_float32,);

    /* (8, 8) * (8) */
    TENCHK(fang_ten_mul(&tens->res_8x8_int16, &tens->ten_8x8_int16,
        &tens->ten_8_int16));
    TENCHK(fang_ten_mul(&tens->res_8x8_bfloat16, &tens->ten_8x8_bfloat16,
        &tens->ten_8_bfloat16));
    TENCHK(fang_ten_mul(&tens->res_8x8_float32, &tens->ten_8x8_float32,
        &tens->ten_8_float32));
    ASSERT_TEN_DATA_EQi(tens->res_8x8_int16, ten_mul_result5_int16,);
    ASSERT_TEN_DATA_EQbf(tens->res_8x8_bfloat16, ten_mul_result5_float32,);
    ASSERT_TEN_DATA_EQf(tens->res_8x8_float32, ten_mul_result5_float32,);

    /* (8, 8) * (8, 1) */
    TENCHK(fang_ten_mul(&tens->res_8x8_int16, &tens->ten_8x8_int16,
        &tens->ten_8x1_int16));
    TENCHK(fang_ten_mul(&tens->res_8x8_bfloat16, &tens->ten_8x8_bfloat16,
        &tens->ten_8x1_bfloat16));
    TENCHK(fang_ten_mul(&tens->res_8x8_float32, &tens->ten_8x8_float32,
        &tens->ten_8x1_float32));
    ASSERT_TEN_DATA_EQi(tens->res_8x8_int16, ten_mul_result6_int16,);
    ASSERT_TEN_DATA_EQbf(tens->res_8x8_bfloat16, ten_mul_result6_float32,);
    ASSERT_TEN_DATA_EQf(tens->res_8x8_float32, ten_mul_result6_float32,);

    /* (3, 8, 8) * (8, 8) */
    TENCHK(fang_ten_mul(&tens->res_3x8x8_int16, &tens->ten_3x8x8_int16,
        &tens->ten_8x8_int16));
    TENCHK(fang_ten_mul(&tens->res_3x8x8_bfloat16, &tens->ten_3x8x8_bfloat16,
        &tens->ten_8x8_bfloat16));
    TENCHK(fang_ten_mul(&tens->res_3x8x8_float32, &tens->ten_3x8x8_float32,
        &tens->ten_8x8_float32));
    ASSERT_TEN_DATA_EQi(tens->res_3x8x8_int16, ten_mul_result7_int16,);
    ASSERT_TEN_DATA_EQbf(tens->res_3x8x8_bfloat16, ten_mul_result7_float32,);
    ASSERT_TEN_DATA_EQf(tens->res_3x8x8_float32, ten_mul_result7_float32,);

    /* (2, 4, 8) * (1, 4, 1) */
    TENCHK(fang_ten_mul(&tens->res_2x4x8_int16, &tens->ten_2x4x8_int16,
        &tens->ten_1x4x1_int16));
    TENCHK(fang_ten_mul(&tens->res_2x4x8_bfloat16, &tens->ten_2x4x8_bfloat16,
        &tens->ten_1x4x1_bfloat16));
    TENCHK(fang_ten_mul(&tens->res_2x4x8_float32, &tens->ten_2x4x8_float32,
        &tens->ten_1x4x1_float32));
    ASSERT_TEN_DATA_EQi(tens->res_2x4x8_int16, ten_mul_result8_int16,);
    ASSERT_TEN_DATA_EQbf(tens->res_2x4x8_bfloat16, ten_mul_result8_float32,);
    ASSERT_TEN_DATA_EQf(tens->res_2x4x8_float32, ten_mul_result8_float32,);

    /* (2, 3, 4, 8) * (1, 3, 4, 8) */
    TENCHK(fang_ten_mul(&tens->res_2x3x4x8_int16, &tens->ten_2x3x4x8_int16,
        &tens->ten_1x3x4x8_int16));
    TENCHK(fang_ten_mul(&tens->res_2x3x4x8_bfloat16,
        &tens->ten_2x3x4x8_bfloat16, &tens->ten_1x3x4x8_bfloat16));
    TENCHK(fang_ten_mul(&tens->res_2x3x4x8_float32, &tens->ten_2x3x4x8_float32,
        &tens->ten_1x3x4x8_float32));
    ASSERT_TEN_DATA_EQi(tens->res_2x3x4x8_int16, ten_mul_result9_int16,);
    ASSERT_TEN_DATA_EQbf(tens->res_2x3x4x8_bfloat16, ten_mul_result9_float32,);
    ASSERT_TEN_DATA_EQf(tens->res_2x3x4x8_float32, ten_mul_result9_float32,);

    /* (2, 3, 4, 8) * (1, 3, 1, 8) */
    TENCHK(fang_ten_mul(&tens->res_2x3x4x8_int16, &tens->ten_2x3x4x8_int16,
        &tens->ten_1x3x1x8_int16));
    TENCHK(fang_ten_mul(&tens->res_2x3x4x8_bfloat16,
        &tens->ten_2x3x4x8_bfloat16, &tens->ten_1x3x1x8_bfloat16));
    TENCHK(fang_ten_mul(&tens->res_2x3x4x8_float32, &tens->ten_2x3x4x8_float32,
        &tens->ten_1x3x1x8_float32));
    ASSERT_TEN_DATA_EQi(tens->res_2x3x4x8_int16, ten_mul_result10_int16,);
    ASSERT_TEN_DATA_EQbf(tens->res_2x3x4x8_bfloat16, ten_mul_result10_float32,);
    ASSERT_TEN_DATA_EQf(tens->res_2x3x4x8_float32, ten_mul_result10_float32,);

    /* (2, 1, 4, 8) * (1, 3, 4, 8) */
    TENCHK(fang_ten_mul(&tens->res_2x3x4x8_int16, &tens->ten_2x1x4x8_int16,
        &tens->ten_1x3x4x8_int16));
    TENCHK(fang_ten_mul(&tens->res_2x3x4x8_bfloat16,
        &tens->ten_2x1x4x8_bfloat16, &tens->ten_1x3x4x8_bfloat16));
    TENCHK(fang_ten_mul(&tens->res_2x3x4x8_float32, &tens->ten_2x1x4x8_float32,
        &tens->ten_1x3x4x8_float32));
    ASSERT_TEN_DATA_EQi(tens->res_2x3x4x8_int16, ten_mul_result11_int16,);
    ASSERT_TEN_DATA_EQbf(tens->res_2x3x4x8_bfloat16, ten_mul_result11_float32,);
    ASSERT_TEN_DATA_EQf(tens->res_2x3x4x8_float32, ten_mul_result11_float32,);

    /* (2, 1, 4, 8) * (1, 3, 1, 8) */
    TENCHK(fang_ten_mul(&tens->res_2x3x4x8_int16, &tens->ten_2x1x4x8_int16,
        &tens->ten_1x3x1x8_int16));
    TENCHK(fang_ten_mul(&tens->res_2x3x4x8_bfloat16,
        &tens->ten_2x1x4x8_bfloat16, &tens->ten_1x3x1x8_bfloat16));
    TENCHK(fang_ten_mul(&tens->res_2x3x4x8_float32, &tens->ten_2x1x4x8_float32,
        &tens->ten_1x3x1x8_float32));
    ASSERT_TEN_DATA_EQi(tens->res_2x3x4x8_int16, ten_mul_result12_int16,);
    ASSERT_TEN_DATA_EQbf(tens->res_2x3x4x8_bfloat16, ten_mul_result12_float32,);
    ASSERT_TEN_DATA_EQf(tens->res_2x3x4x8_float32, ten_mul_result12_float32,);

    /* () * (3, 8, 8) */
    TENCHK(fang_ten_mul(&tens->res_3x8x8_int16, &tens->sc1_int16,
        &tens->ten_3x8x8_int16));
    TENCHK(fang_ten_mul(&tens->res_3x8x8_bfloat16, &tens->sc1_bfloat16,
        &tens->ten_3x8x8_bfloat16));
    TENCHK(fang_ten_mul(&tens->res_3x8x8_float32, &tens->sc1_float32,
        &tens->ten_3x8x8_float32));
    ASSERT_TEN_DATA_EQi(tens->res_3x8x8_int16, ten_mul_result13_int16,);
    ASSERT_TEN_DATA_EQbf(tens->res_3x8x8_bfloat16, ten_mul_result13_float32,);
    ASSERT_TEN_DATA_EQf(tens->res_3x8x8_float32, ten_mul_result13_float32,);

    /* (3, 1, 1, 2, 1) * (1, 2, 3, 2, 5) */
    TENCHK(fang_ten_mul(&tens->res_3x2x3x2x5_int16, &tens->ten_3x1x1x2x1_int16,
        &tens->ten_1x2x3x2x5_int16));
    TENCHK(fang_ten_mul(&tens->res_3x2x3x2x5_bfloat16,
        &tens->ten_3x1x1x2x1_bfloat16, &tens->ten_1x2x3x2x5_bfloat16));
    TENCHK(fang_ten_mul(&tens->res_3x2x3x2x5_float32,
        &tens->ten_3x1x1x2x1_float32, &tens->ten_1x2x3x2x5_float32));
    ASSERT_TEN_DATA_EQi(tens->res_3x2x3x2x5_int16,
        ten_mul_result14_int16,);
    ASSERT_TEN_DATA_EQbf(tens->res_3x2x3x2x5_bfloat16,
        ten_mul_result14_float32,);
    ASSERT_TEN_DATA_EQf(tens->res_3x2x3x2x5_float32,
        ten_mul_result14_float32,);
}

static void fang_ten_diff_test(void **state) {
    /* Get test tensors. */
    _fang_ten_tests_t *tens = (_fang_ten_tests_t *) *state;

    _FANG_ARITHMETIC_ERRCHK(diff);

    /* (8) - (8) */
    TENCHK(fang_ten_diff(&tens->res_8_int16, &tens->ten_8_int16,
        &tens->ten_8_int16));
    TENCHK(fang_ten_diff(&tens->res_8_bfloat16, &tens->ten_8_bfloat16,
        &tens->ten_8_bfloat16));
    TENCHK(fang_ten_diff(&tens->res_8_float32, &tens->ten_8_float32,
        &tens->ten_8_float32));
    ASSERT_TEN_DATA_EQi(tens->res_8_int16, ten_diff_result1_int16,);
    ASSERT_TEN_DATA_EQbf(tens->res_8_bfloat16, ten_diff_result1_float32,);
    ASSERT_TEN_DATA_EQf(tens->res_8_float32, ten_diff_result1_float32,);


    /* () - () */
    TENCHK(fang_ten_diff(&tens->res_sc_int16, &tens->sc1_int16,
        &tens->sc2_int16));
    TENCHK(fang_ten_diff(&tens->res_sc_bfloat16, &tens->sc1_bfloat16,
        &tens->sc2_bfloat16));
    TENCHK(fang_ten_diff(&tens->res_sc_float32, &tens->sc1_float32,
        &tens->sc2_float32));
    assert_int_equal(*(int16_t *) tens->res_sc_int16.data.dense,
        ten_diff_result2_int16[0]);
    assert_float_equal(_FANG_BH2S(*(_fang_bfloat16_t *)
        tens->res_sc_bfloat16.data.dense),
        _FANG_BH2S(_FANG_S2BH(ten_diff_result2_float32[0])), 1e-6);
    assert_float_equal(*(float *) tens->res_sc_float32.data.dense,
        ten_diff_result2_float32[0], 1e-6);
    /* Commute. */
    TENCHK(fang_ten_diff(&tens->res_sc_int16, &tens->sc2_int16,
        &tens->sc1_int16));
    TENCHK(fang_ten_diff(&tens->res_sc_bfloat16, &tens->sc2_bfloat16,
        &tens->sc1_bfloat16));
    TENCHK(fang_ten_diff(&tens->res_sc_float32, &tens->sc2_float32,
        &tens->sc1_float32));
    assert_int_equal(*(int16_t *) tens->res_sc_int16.data.dense,
        -ten_diff_result2_int16[0]);
    assert_float_equal(_FANG_BH2S(*(_fang_bfloat16_t *)
        tens->res_sc_bfloat16.data.dense),
        _FANG_BH2S(_FANG_S2BH(-ten_diff_result2_float32[0])), 1e-6);
    assert_float_equal(*(float *) tens->res_sc_float32.data.dense,
        -ten_diff_result2_float32[0], 1e-6);


    /* (8) - () */
    TENCHK(fang_ten_diff(&tens->res_8_int16, &tens->ten_8_int16,
        &tens->sc1_int16));
    TENCHK(fang_ten_diff(&tens->res_8_bfloat16, &tens->ten_8_bfloat16,
        &tens->sc1_bfloat16));
    TENCHK(fang_ten_diff(&tens->res_8_float32, &tens->ten_8_float32,
        &tens->sc1_float32));
    ASSERT_TEN_DATA_EQi(tens->res_8_int16, ten_diff_result3_int16,);
    ASSERT_TEN_DATA_EQbf(tens->res_8_bfloat16, ten_diff_result3_float32,);
    ASSERT_TEN_DATA_EQf(tens->res_8_float32, ten_diff_result3_float32,);
    /* () - (8) */
    TENCHK(fang_ten_diff(&tens->res_8_int16, &tens->sc1_int16,
        &tens->ten_8_int16));
    TENCHK(fang_ten_diff(&tens->res_8_bfloat16, &tens->sc1_bfloat16,
        &tens->ten_8_bfloat16));
    TENCHK(fang_ten_diff(&tens->res_8_float32, &tens->sc1_float32,
        &tens->ten_8_float32));
    ASSERT_TEN_DATA_EQi(tens->res_8_int16, ten_diff_result3_int16, -);
    ASSERT_TEN_DATA_EQbf(tens->res_8_bfloat16, ten_diff_result3_float32, -);
    ASSERT_TEN_DATA_EQf(tens->res_8_float32, ten_diff_result3_float32, -);


    /* (8) - (1) */
    /* NOTE: A single element 1-dimensional tensor can be thought of as a
       scalar tensor. */
    TENCHK(fang_ten_diff(&tens->res_8_int16, &tens->ten_8_int16,
        &tens->ten_1_int16));
    TENCHK(fang_ten_diff(&tens->res_8_bfloat16, &tens->ten_8_bfloat16,
        &tens->ten_1_bfloat16));
    TENCHK(fang_ten_diff(&tens->res_8_float32, &tens->ten_8_float32,
        &tens->ten_1_float32));
    ASSERT_TEN_DATA_EQi(tens->res_8_int16, ten_diff_result4_int16,);
    ASSERT_TEN_DATA_EQbf(tens->res_8_bfloat16, ten_diff_result4_float32,);
    ASSERT_TEN_DATA_EQf(tens->res_8_float32, ten_diff_result4_float32,);
    /* (1) - (8) */
    TENCHK(fang_ten_diff(&tens->res_8_int16, &tens->ten_1_int16,
        &tens->ten_8_int16));
    TENCHK(fang_ten_diff(&tens->res_8_bfloat16, &tens->ten_1_bfloat16,
        &tens->ten_8_bfloat16));
    TENCHK(fang_ten_diff(&tens->res_8_float32, &tens->ten_1_float32,
        &tens->ten_8_float32));
    ASSERT_TEN_DATA_EQi(tens->res_8_int16, ten_diff_result4_int16, -);
    ASSERT_TEN_DATA_EQbf(tens->res_8_bfloat16, ten_diff_result4_float32, -);
    ASSERT_TEN_DATA_EQf(tens->res_8_float32, ten_diff_result4_float32, -);


    /* (8) - (1, 1, 1, 1) */
    TENCHK(fang_ten_diff(&tens->res_1x1x1x8_int16, &tens->ten_8_int16,
        &tens->ten_1x1x1x1_int16));
    TENCHK(fang_ten_diff(&tens->res_1x1x1x8_bfloat16, &tens->ten_8_bfloat16,
        &tens->ten_1x1x1x1_bfloat16));
    TENCHK(fang_ten_diff(&tens->res_1x1x1x8_float32, &tens->ten_8_float32,
        &tens->ten_1x1x1x1_float32));
    ASSERT_TEN_DATA_EQi(tens->res_1x1x1x8_int16, ten_diff_result4_int16,);
    ASSERT_TEN_DATA_EQbf(tens->res_1x1x1x8_bfloat16, ten_diff_result4_float32,);
    ASSERT_TEN_DATA_EQf(tens->res_1x1x1x8_float32, ten_diff_result4_float32,);
    /* (1, 1, 1, 1) - (8) */
    TENCHK(fang_ten_diff(&tens->res_1x1x1x8_int16, &tens->ten_1x1x1x1_int16,
        &tens->ten_8_int16));
    TENCHK(fang_ten_diff(&tens->res_1x1x1x8_bfloat16, &tens->ten_1x1x1x1_bfloat16,
        &tens->ten_8_bfloat16));
    TENCHK(fang_ten_diff(&tens->res_1x1x1x8_float32, &tens->ten_1x1x1x1_float32,
        &tens->ten_8_float32));
    ASSERT_TEN_DATA_EQi(tens->res_1x1x1x8_int16, ten_diff_result4_int16, -);
    ASSERT_TEN_DATA_EQbf(tens->res_1x1x1x8_bfloat16, ten_diff_result4_float32, -);
    ASSERT_TEN_DATA_EQf(tens->res_1x1x1x8_float32, ten_diff_result4_float32, -);


    /* (8, 8) - (8) */
    TENCHK(fang_ten_diff(&tens->res_8x8_int16, &tens->ten_8x8_int16,
        &tens->ten_8_int16));
    TENCHK(fang_ten_diff(&tens->res_8x8_bfloat16, &tens->ten_8x8_bfloat16,
        &tens->ten_8_bfloat16));
    TENCHK(fang_ten_diff(&tens->res_8x8_float32, &tens->ten_8x8_float32,
        &tens->ten_8_float32));
    ASSERT_TEN_DATA_EQi(tens->res_8x8_int16, ten_diff_result5_int16,);
    ASSERT_TEN_DATA_EQbf(tens->res_8x8_bfloat16, ten_diff_result5_float32,);
    ASSERT_TEN_DATA_EQf(tens->res_8x8_float32, ten_diff_result5_float32,);
    /* (8) - (8, 8) */
    TENCHK(fang_ten_diff(&tens->res_8x8_int16, &tens->ten_8_int16,
        &tens->ten_8x8_int16));
    TENCHK(fang_ten_diff(&tens->res_8x8_bfloat16, &tens->ten_8_bfloat16,
        &tens->ten_8x8_bfloat16));
    TENCHK(fang_ten_diff(&tens->res_8x8_float32, &tens->ten_8_float32,
        &tens->ten_8x8_float32));
    ASSERT_TEN_DATA_EQi(tens->res_8x8_int16, ten_diff_result5_int16, -);
    ASSERT_TEN_DATA_EQbf(tens->res_8x8_bfloat16, ten_diff_result5_float32, -);
    ASSERT_TEN_DATA_EQf(tens->res_8x8_float32, ten_diff_result5_float32, -);


    /* (8, 8) - (8, 1) */
    TENCHK(fang_ten_diff(&tens->res_8x8_int16, &tens->ten_8x8_int16,
        &tens->ten_8x1_int16));
    TENCHK(fang_ten_diff(&tens->res_8x8_bfloat16, &tens->ten_8x8_bfloat16,
        &tens->ten_8x1_bfloat16));
    TENCHK(fang_ten_diff(&tens->res_8x8_float32, &tens->ten_8x8_float32,
        &tens->ten_8x1_float32));
    ASSERT_TEN_DATA_EQi(tens->res_8x8_int16, ten_diff_result6_int16,);
    ASSERT_TEN_DATA_EQbf(tens->res_8x8_bfloat16, ten_diff_result6_float32,);
    ASSERT_TEN_DATA_EQf(tens->res_8x8_float32, ten_diff_result6_float32,);
    /* (8, 1) - (8, 8) */
    TENCHK(fang_ten_diff(&tens->res_8x8_int16, &tens->ten_8x1_int16,
        &tens->ten_8x8_int16));
    TENCHK(fang_ten_diff(&tens->res_8x8_bfloat16, &tens->ten_8x1_bfloat16,
        &tens->ten_8x8_bfloat16));
    TENCHK(fang_ten_diff(&tens->res_8x8_float32, &tens->ten_8x1_float32,
        &tens->ten_8x8_float32));
    ASSERT_TEN_DATA_EQi(tens->res_8x8_int16, ten_diff_result6_int16, -);
    ASSERT_TEN_DATA_EQbf(tens->res_8x8_bfloat16, ten_diff_result6_float32, -);
    ASSERT_TEN_DATA_EQf(tens->res_8x8_float32, ten_diff_result6_float32, -);


    /* (3, 8, 8) - (8, 8) */
    TENCHK(fang_ten_diff(&tens->res_3x8x8_int16, &tens->ten_3x8x8_int16,
        &tens->ten_8x8_int16));
    TENCHK(fang_ten_diff(&tens->res_3x8x8_bfloat16, &tens->ten_3x8x8_bfloat16,
        &tens->ten_8x8_bfloat16));
    TENCHK(fang_ten_diff(&tens->res_3x8x8_float32, &tens->ten_3x8x8_float32,
        &tens->ten_8x8_float32));
    ASSERT_TEN_DATA_EQi(tens->res_3x8x8_int16, ten_diff_result7_int16,);
    ASSERT_TEN_DATA_EQbf(tens->res_3x8x8_bfloat16, ten_diff_result7_float32,);
    ASSERT_TEN_DATA_EQf(tens->res_3x8x8_float32, ten_diff_result7_float32,);
    /* (8, 8) - (3, 8, 8) */
    TENCHK(fang_ten_diff(&tens->res_3x8x8_int16, &tens->ten_8x8_int16,
        &tens->ten_3x8x8_int16));
    TENCHK(fang_ten_diff(&tens->res_3x8x8_bfloat16, &tens->ten_8x8_bfloat16,
        &tens->ten_3x8x8_bfloat16));
    TENCHK(fang_ten_diff(&tens->res_3x8x8_float32, &tens->ten_8x8_float32,
        &tens->ten_3x8x8_float32));
    ASSERT_TEN_DATA_EQi(tens->res_3x8x8_int16, ten_diff_result7_int16, -);
    ASSERT_TEN_DATA_EQbf(tens->res_3x8x8_bfloat16, ten_diff_result7_float32, -);
    ASSERT_TEN_DATA_EQf(tens->res_3x8x8_float32, ten_diff_result7_float32, -);


    /* (2, 4, 8) - (1, 4, 1) */
    TENCHK(fang_ten_diff(&tens->res_2x4x8_int16, &tens->ten_2x4x8_int16,
        &tens->ten_1x4x1_int16));
    TENCHK(fang_ten_diff(&tens->res_2x4x8_bfloat16, &tens->ten_2x4x8_bfloat16,
        &tens->ten_1x4x1_bfloat16));
    TENCHK(fang_ten_diff(&tens->res_2x4x8_float32, &tens->ten_2x4x8_float32,
        &tens->ten_1x4x1_float32));
    ASSERT_TEN_DATA_EQi(tens->res_2x4x8_int16, ten_diff_result8_int16,);
    ASSERT_TEN_DATA_EQbf(tens->res_2x4x8_bfloat16, ten_diff_result8_float32,);
    ASSERT_TEN_DATA_EQf(tens->res_2x4x8_float32, ten_diff_result8_float32,);
    /* (1, 4, 1) - (2, 4, 8) */
    TENCHK(fang_ten_diff(&tens->res_2x4x8_int16, &tens->ten_1x4x1_int16,
        &tens->ten_2x4x8_int16));
    TENCHK(fang_ten_diff(&tens->res_2x4x8_bfloat16, &tens->ten_1x4x1_bfloat16,
        &tens->ten_2x4x8_bfloat16));
    TENCHK(fang_ten_diff(&tens->res_2x4x8_float32, &tens->ten_1x4x1_float32,
        &tens->ten_2x4x8_float32));
    ASSERT_TEN_DATA_EQi(tens->res_2x4x8_int16, ten_diff_result8_int16, -);
    ASSERT_TEN_DATA_EQbf(tens->res_2x4x8_bfloat16, ten_diff_result8_float32, -);
    ASSERT_TEN_DATA_EQf(tens->res_2x4x8_float32, ten_diff_result8_float32, -);


    /* (2, 3, 4, 8) - (1, 3, 4, 8) */
    TENCHK(fang_ten_diff(&tens->res_2x3x4x8_int16, &tens->ten_2x3x4x8_int16,
        &tens->ten_1x3x4x8_int16));
    TENCHK(fang_ten_diff(&tens->res_2x3x4x8_bfloat16,
        &tens->ten_2x3x4x8_bfloat16, &tens->ten_1x3x4x8_bfloat16));
    TENCHK(fang_ten_diff(&tens->res_2x3x4x8_float32, &tens->ten_2x3x4x8_float32,
        &tens->ten_1x3x4x8_float32));
    ASSERT_TEN_DATA_EQi(tens->res_2x3x4x8_int16, ten_diff_result9_int16,);
    ASSERT_TEN_DATA_EQbf(tens->res_2x3x4x8_bfloat16, ten_diff_result9_float32,);
    ASSERT_TEN_DATA_EQf(tens->res_2x3x4x8_float32, ten_diff_result9_float32,);
    /* (1, 3, 4, 8) - (2, 3, 4, 8) */
    TENCHK(fang_ten_diff(&tens->res_2x3x4x8_int16, &tens->ten_1x3x4x8_int16,
        &tens->ten_2x3x4x8_int16));
    TENCHK(fang_ten_diff(&tens->res_2x3x4x8_bfloat16,
        &tens->ten_1x3x4x8_bfloat16, &tens->ten_2x3x4x8_bfloat16));
    TENCHK(fang_ten_diff(&tens->res_2x3x4x8_float32, &tens->ten_1x3x4x8_float32,
        &tens->ten_2x3x4x8_float32));
    ASSERT_TEN_DATA_EQi(tens->res_2x3x4x8_int16,
        ten_diff_result9_int16, -);
    ASSERT_TEN_DATA_EQbf(tens->res_2x3x4x8_bfloat16,
        ten_diff_result9_float32, -);
    ASSERT_TEN_DATA_EQf(tens->res_2x3x4x8_float32, ten_diff_result9_float32, -);


    /* (2, 3, 4, 8) - (1, 3, 1, 8) */
    TENCHK(fang_ten_diff(&tens->res_2x3x4x8_int16, &tens->ten_2x3x4x8_int16,
        &tens->ten_1x3x1x8_int16));
    TENCHK(fang_ten_diff(&tens->res_2x3x4x8_bfloat16,
        &tens->ten_2x3x4x8_bfloat16, &tens->ten_1x3x1x8_bfloat16));
    TENCHK(fang_ten_diff(&tens->res_2x3x4x8_float32, &tens->ten_2x3x4x8_float32,
        &tens->ten_1x3x1x8_float32));
    ASSERT_TEN_DATA_EQi(tens->res_2x3x4x8_int16, ten_diff_result10_int16,);
    ASSERT_TEN_DATA_EQbf(tens->res_2x3x4x8_bfloat16,
        ten_diff_result10_float32,);
    ASSERT_TEN_DATA_EQf(tens->res_2x3x4x8_float32, ten_diff_result10_float32,);
    /* (1, 3, 1, 8) - (2, 3, 4, 8) */
    TENCHK(fang_ten_diff(&tens->res_2x3x4x8_int16, &tens->ten_1x3x1x8_int16,
        &tens->ten_2x3x4x8_int16));
    TENCHK(fang_ten_diff(&tens->res_2x3x4x8_bfloat16,
        &tens->ten_1x3x1x8_bfloat16, &tens->ten_2x3x4x8_bfloat16));
    TENCHK(fang_ten_diff(&tens->res_2x3x4x8_float32, &tens->ten_1x3x1x8_float32,
        &tens->ten_2x3x4x8_float32));
    ASSERT_TEN_DATA_EQi(tens->res_2x3x4x8_int16, ten_diff_result10_int16, -);
    ASSERT_TEN_DATA_EQbf(tens->res_2x3x4x8_bfloat16,
        ten_diff_result10_float32, -);
    ASSERT_TEN_DATA_EQf(tens->res_2x3x4x8_float32,
        ten_diff_result10_float32, -);


    /* (2, 1, 4, 8) - (1, 3, 4, 8) */
    TENCHK(fang_ten_diff(&tens->res_2x3x4x8_int16, &tens->ten_2x1x4x8_int16,
        &tens->ten_1x3x4x8_int16));
    TENCHK(fang_ten_diff(&tens->res_2x3x4x8_bfloat16,
        &tens->ten_2x1x4x8_bfloat16, &tens->ten_1x3x4x8_bfloat16));
    TENCHK(fang_ten_diff(&tens->res_2x3x4x8_float32, &tens->ten_2x1x4x8_float32,
        &tens->ten_1x3x4x8_float32));
    ASSERT_TEN_DATA_EQi(tens->res_2x3x4x8_int16, ten_diff_result11_int16,);
    ASSERT_TEN_DATA_EQbf(tens->res_2x3x4x8_bfloat16,
        ten_diff_result11_float32,);
    ASSERT_TEN_DATA_EQf(tens->res_2x3x4x8_float32, ten_diff_result11_float32,);
    /* (1, 3, 4, 8) - (2, 1, 4, 8) */
    TENCHK(fang_ten_diff(&tens->res_2x3x4x8_int16, &tens->ten_1x3x4x8_int16,
        &tens->ten_2x1x4x8_int16));
    TENCHK(fang_ten_diff(&tens->res_2x3x4x8_bfloat16,
        &tens->ten_1x3x4x8_bfloat16, &tens->ten_2x1x4x8_bfloat16));
    TENCHK(fang_ten_diff(&tens->res_2x3x4x8_float32, &tens->ten_1x3x4x8_float32,
        &tens->ten_2x1x4x8_float32));
    ASSERT_TEN_DATA_EQi(tens->res_2x3x4x8_int16, ten_diff_result11_int16, -);
    ASSERT_TEN_DATA_EQbf(tens->res_2x3x4x8_bfloat16,
        ten_diff_result11_float32, -);
    ASSERT_TEN_DATA_EQf(tens->res_2x3x4x8_float32,
        ten_diff_result11_float32, -);


    /* (2, 1, 4, 8) - (1, 3, 1, 8) */
    TENCHK(fang_ten_diff(&tens->res_2x3x4x8_int16, &tens->ten_2x1x4x8_int16,
        &tens->ten_1x3x1x8_int16));
    TENCHK(fang_ten_diff(&tens->res_2x3x4x8_bfloat16,
        &tens->ten_2x1x4x8_bfloat16, &tens->ten_1x3x1x8_bfloat16));
    TENCHK(fang_ten_diff(&tens->res_2x3x4x8_float32, &tens->ten_2x1x4x8_float32,
        &tens->ten_1x3x1x8_float32));
    ASSERT_TEN_DATA_EQi(tens->res_2x3x4x8_int16, ten_diff_result12_int16,);
    ASSERT_TEN_DATA_EQbf(tens->res_2x3x4x8_bfloat16,
        ten_diff_result12_float32,);
    ASSERT_TEN_DATA_EQf(tens->res_2x3x4x8_float32, ten_diff_result12_float32,);
    /* (1, 3, 1, 8) - (2, 1, 4, 8) */
    TENCHK(fang_ten_diff(&tens->res_2x3x4x8_int16, &tens->ten_1x3x1x8_int16,
        &tens->ten_2x1x4x8_int16));
    TENCHK(fang_ten_diff(&tens->res_2x3x4x8_bfloat16,
        &tens->ten_1x3x1x8_bfloat16, &tens->ten_2x1x4x8_bfloat16));
    TENCHK(fang_ten_diff(&tens->res_2x3x4x8_float32, &tens->ten_1x3x1x8_float32,
        &tens->ten_2x1x4x8_float32));
    ASSERT_TEN_DATA_EQi(tens->res_2x3x4x8_int16, ten_diff_result12_int16, -);
    ASSERT_TEN_DATA_EQbf(tens->res_2x3x4x8_bfloat16,
        ten_diff_result12_float32, -);
    ASSERT_TEN_DATA_EQf(tens->res_2x3x4x8_float32,
        ten_diff_result12_float32, -);


    /* () - (3, 8, 8) */
    TENCHK(fang_ten_diff(&tens->res_3x8x8_int16, &tens->sc1_int16,
        &tens->ten_3x8x8_int16));
    TENCHK(fang_ten_diff(&tens->res_3x8x8_bfloat16, &tens->sc1_bfloat16,
        &tens->ten_3x8x8_bfloat16));
    TENCHK(fang_ten_diff(&tens->res_3x8x8_float32, &tens->sc1_float32,
        &tens->ten_3x8x8_float32));
    ASSERT_TEN_DATA_EQi(tens->res_3x8x8_int16, ten_diff_result13_int16,);
    ASSERT_TEN_DATA_EQbf(tens->res_3x8x8_bfloat16, ten_diff_result13_float32,);
    ASSERT_TEN_DATA_EQf(tens->res_3x8x8_float32, ten_diff_result13_float32,);
    /* (3, 8, 8) - () */
    TENCHK(fang_ten_diff(&tens->res_3x8x8_int16, &tens->ten_3x8x8_int16,
        &tens->sc1_int16));
    TENCHK(fang_ten_diff(&tens->res_3x8x8_bfloat16, &tens->ten_3x8x8_bfloat16,
        &tens->sc1_bfloat16));
    TENCHK(fang_ten_diff(&tens->res_3x8x8_float32, &tens->ten_3x8x8_float32,
        &tens->sc1_float32));
    ASSERT_TEN_DATA_EQi(tens->res_3x8x8_int16, ten_diff_result13_int16, -);
    ASSERT_TEN_DATA_EQbf(tens->res_3x8x8_bfloat16, ten_diff_result13_float32, -);
    ASSERT_TEN_DATA_EQf(tens->res_3x8x8_float32, ten_diff_result13_float32, -);


    /* (3, 1, 1, 2, 1) - (1, 2, 3, 2, 5) */
    TENCHK(fang_ten_diff(&tens->res_3x2x3x2x5_int16, &tens->ten_3x1x1x2x1_int16,
        &tens->ten_1x2x3x2x5_int16));
    TENCHK(fang_ten_diff(&tens->res_3x2x3x2x5_bfloat16,
        &tens->ten_3x1x1x2x1_bfloat16, &tens->ten_1x2x3x2x5_bfloat16));
    TENCHK(fang_ten_diff(&tens->res_3x2x3x2x5_float32,
        &tens->ten_3x1x1x2x1_float32, &tens->ten_1x2x3x2x5_float32));
    ASSERT_TEN_DATA_EQi(tens->res_3x2x3x2x5_int16,
        ten_diff_result14_int16,);
    ASSERT_TEN_DATA_EQbf(tens->res_3x2x3x2x5_bfloat16,
        ten_diff_result14_float32,);
    ASSERT_TEN_DATA_EQf(tens->res_3x2x3x2x5_float32,
        ten_diff_result14_float32,);
    /* (1, 2, 3, 2, 5) - (3, 1, 1, 2, 1) */
    TENCHK(fang_ten_diff(&tens->res_3x2x3x2x5_int16, &tens->ten_1x2x3x2x5_int16,
        &tens->ten_3x1x1x2x1_int16));
    TENCHK(fang_ten_diff(&tens->res_3x2x3x2x5_bfloat16,
        &tens->ten_1x2x3x2x5_bfloat16, &tens->ten_3x1x1x2x1_bfloat16));
    TENCHK(fang_ten_diff(&tens->res_3x2x3x2x5_float32,
        &tens->ten_1x2x3x2x5_float32, &tens->ten_3x1x1x2x1_float32));
    ASSERT_TEN_DATA_EQi(tens->res_3x2x3x2x5_int16,
        ten_diff_result14_int16, -);
    ASSERT_TEN_DATA_EQbf(tens->res_3x2x3x2x5_bfloat16,
        ten_diff_result14_float32, -);
    ASSERT_TEN_DATA_EQf(tens->res_3x2x3x2x5_float32,
        ten_diff_result14_float32, -);
}

/* ================ TESTS END ================ */

int main() {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(fang_ten_create_test),
        cmocka_unit_test_setup_teardown(fang_ten_sum_test, setup_arithmetic,
            teardown_arithmetic),
        cmocka_unit_test_setup_teardown(fang_ten_mul_test, setup_arithmetic,
            teardown_arithmetic),
        cmocka_unit_test_setup_teardown(fang_ten_diff_test, setup_arithmetic,
            teardown_arithmetic),
    };

    return cmocka_run_group_tests_name("tensor/dense", tests, setup, teardown);
}
