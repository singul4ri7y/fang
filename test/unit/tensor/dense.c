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

/* ================ HELPER MACROS END ================ */


/* ================ PRIVATE GLOBALS ================ */

/* Expected data from `ten_x` in Randomizer test. */
static const int16_t expected_data_x[] = { 225, 129, -581, -150, 107, 368, -9,
    187, 313, -829, 625, -42, -275, 913, -352, 713, -391, -488, -568, 972, 999,
    515, 667, 187, 88, -872, -757, 241, -697, 126, -657, -32, 27, 797, 465, 673,
    747, -399, 738, -447, 344, -614, 638, 125, -229, 498, -504, -614 };

/* Expected data from `ten_y` in Randomizer test. */
static const float expected_data_y[] = { 0.5625, 1.5, 1, 0.8125, 0.25, 1.75,
    1.5, 1.375, 0.5, 0.25, 0.625, 1.25, 0.3125, 0.25, 0.050781, 0.4375, 1.625,
    0.4375, 1.375, 1.75, 0.75, 1.75, 1, 0.029297, 0.011719, 1.5, 0.8125, 1.375,
    1.375, 0.0625, 1.125, 1.625, 1, 0.6875, 0.6875, 0.5625, 1, 0.75, 0.5625,
    0.46875, 0.203125, 1, 1.875, 0.171875, 0.203125, 0.8125, 1.25, 0.25, 0.8125,
    0.8125, 1.625, 1.5, 0.625, 0.625, 1, 1.625, 0.375, 1.375, 0.75, 0.035156,
    1.875, 0.875, 1.75, 1.5, 0.46875, 0.34375, 0.5, 0.46875, 1.375, 1.125,
    1.625, 0.6875, 1, 1.875, 0.046875, 0.9375, 0.140625, 0.5625, 1, 1.5, 0.5,
    1.5, 1.25, 0.46875, 1.25, 0.21875, 0.5625, 1.375, 1.75, 0.9375, 0.125,
    0.34375, 0.25, 0.070312, 1.75, 0.5625, 0.75, 1.875, 1.625, 0.171875, 0.5,
    1.375, 0.5, 1.75, 1.625, 1.875, 1, 1, 0.125, 1.5, 1.625, 1.5, 0.078125,
    1.625, 0.9375, 1.625, 1.375, 0.4375, 0.875, 0.3125, 0.5, 0.9375, 0.5, 1.375,
    1.625, 1.125, 1, 1.625, 0.625, 0.0625, 0.34375, 0.375, 0.625, 0.8125, 1.75,
    0.09375, 1.5, 1.25, 1.875, 0.9375, 1, 0.9375, 0.6875, 0.625, 1.75, 1.125,
    0.5, 1.25, 0.75, 0.15625, 0.050781, 0.4375, 1.375, 1, 0.40625, 0.9375,
    0.625, 0.8125, 0.3125, 0.5625, 0.042969, 0.0625, 1.875, 0.6875, 0.875, 1.75,
    1, 1.25, 1.375, 1.25, 0.6875, 1.875, 0.9375, 0.75, 0.28125, 0.035156, 1,
    0.8125, 0.6875, 0.5, 0.015625, 1.125, 1.125, 0.8125, 0.5, 0.070312, 1.25,
    0.050781, 0.6875, 0.625, 1.5, 0.054688, 1.625, 0.46875, 0.625, 1.875, 1.375,
    1.625, 1, 1.875, 0.6875, 0.21875, 0.375, 0.5625, 0.6875, 0.21875, 1.875,
    0.0625, 0.109375, 0.8125, 1.125, 1.5, 0.75, 0.4375, 0.375, 1.5, 1, 1.625,
    1.75, 0.5, 0.171875, 1.125, 1.625, 1.875, 1.375, 0.75, 1.625, 1.5, 1.625,
    0.75, 1.25, 1.375, 0.021484, 1.125, 0.9375, 0.203125, 0.28125, 1.25,
    0.011719, 0.005859, 0.28125, 0.03125, 0.875, 1.5, 1.75, 1.625, 0.8125, 1.5,
    1, 0.5625, 1.125, 1.25, 0.5625, 0.125, 0.5625, 1 };

/* ================ PRIVATE GLOBALS END ================ */


/* ================ SETUP AND TEARDOWN ================ */

/* Setup Environment for every test. */
static int setup(void **state) {
    int *env = malloc(sizeof(int));
    if(env == NULL) return 1;

    *env = fang_env_create(FANG_ENV_TYPE_CPU, NULL);
    if(!FANG_ISOK(*env))
        return 1;

    *state = (void *) env;
    return 0;
}

/* Release the Environment. */
static int teardown(void **state) {
    int *env = (int *) *state;
    fang_env_release(*env);
    free(env);
    return 0;
}

/* ================ SETUP AND TEARDOWN END ================ */


/* ================ TESTS ================ */

/* Tensor creation test. */
void fang_ten_create_test(void **state) {
    /* Get the Environment. */
    int env = *(int *) *state;

    uint32_t dims[] = { 3, 4, 7, 2, 8, 5 };
    fang_ten_t ten;

    /* Tensor creation should be successful. */
    assert_true(FANG_ISOK(fang_ten_create(&ten, env, FANG_TEN_DTYPE_FLOAT16,
        dims, _ARR_SIZ(dims), NULL)));

    /* Environment ID should match. */
    assert_int_equal(ten.eid, env);

    /* Dimension count should match. */
    assert_int_equal(ten.ndims, _ARR_SIZ(dims));

    /* Stridemension check. */
    assert_int_equal(ten.sdims[ten.ndims - 1], dims[0]);
    for(int i = 1; i < ten.ndims; i++) {
        uint32_t dim;
        if(i + 1 == ten.ndims)  // Final dimension
            dim = ten.sdims[i - 1];
        else dim = ten.sdims[i - 1] / ten.sdims[i];

        /* Extracted dimension should match. */
        assert_int_equal(dims[i], dim);
    }

    /* Tensor type should be dense. */
    assert_int_equal(ten.typ, FANG_TEN_TYPE_DENSE);

    /* Tensor data type should be float16. */
    assert_int_equal(ten.dtyp, FANG_TEN_DTYPE_FLOAT16);

    /* Number of elements in tensor. */
    int size = ten.sdims[0] * ten.sdims[ten.ndims - 1];

    /* Data should be zero-ed out. */
    _fang_float16_t *fdata = ten.data.dense;
    for(int i = 0; i < size; i++)
        assert_float_equal(0.0, _FANG_H2S(fdata[i]), 1e-6);

    fang_int input_data[size];
    for(int i = 0; i < size; i++)
        input_data[i] = i - 3360;

    /* Release previous tensor. */
    fang_ten_release(&ten);

    /* Create a new tensor with data. */
    assert_true(FANG_ISOK(fang_ten_create(&ten, env, FANG_TEN_DTYPE_INT8,
        dims, _ARR_SIZ(dims), input_data)));

    /* Check data initialization. */
    int8_t *idata = ten.data.dense;
    for(int i = 0; i < size; i++)
        assert_int_equal(idata[i], (int8_t) input_data[i]);

    fang_ten_release(&ten);
}

/* Tensor randomization test. */
void fang_ten_rand_test(void **state) {
    /* Get the Environment. */
    int env = *(int *) *state;

    /* Limit environment to a single processor diminishing seed variablitiy. */
    assert_true(FANG_ISOK(fang_env_cpu_actproc(env, 0, 1)));

    uint32_t dims_x[] = { 4, 3, 4 };
    uint32_t dims_y[] = { 16, 16 };

    fang_ten_t ten_x;
    fang_ten_t ten_y;
    assert_true(FANG_ISOK(fang_ten_create(&ten_x, env, FANG_TEN_DTYPE_INT16,
        dims_x, _ARR_SIZ(dims_x), NULL)));
    assert_true(FANG_ISOK(fang_ten_create(&ten_y, env, FANG_TEN_DTYPE_FLOAT8,
        dims_y, _ARR_SIZ(dims_y), NULL)));

    fang_ten_rand(&ten_x, FANG_I2G(-1024), FANG_I2G(1024), 69);
    fang_ten_rand(&ten_y, FANG_F2G(0.0), FANG_F2G(2.0), 69);

    /* `ten_x` data should match. */
    int size_x = ten_x.sdims[0] * ten_x.sdims[ten_x.ndims - 1];
    int16_t *data_x = ten_x.data.dense;
    for(int i = 0; i < size_x; i++)
        assert_int_equal(expected_data_x[i], data_x[i]);

    /* `ten_x` data should match. */
    int size_y = ten_y.sdims[0] * ten_y.sdims[ten_y.ndims - 1];
    _fang_float8_t *data_y = ten_y.data.dense;
    for(int i = 0; i < size_y; i++)
        assert_float_equal(expected_data_y[i], _FANG_Q2S(data_y[i]), 1e-6);

    fang_ten_release(&ten_x);
    fang_ten_release(&ten_y);

    /* Revert back to all processors. */
    assert_true(FANG_ISOK(fang_env_cpu_actproc(env, 0, 0)));
}

/* ================ TESTS END ================ */

int main() {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(fang_ten_create_test),
        cmocka_unit_test(fang_ten_rand_test)
    };

    return cmocka_run_group_tests_name("tensor/dense", tests, setup, teardown);
}
