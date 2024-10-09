#include <fang/status.h>
#include <env/cpu/cpu.h>
#include <memory.h>
#include <stdarg.h>
#include <setjmp.h>
#include <cmocka.h>

/* Environment creation test. */
static void fang_env_create_test(void **state) {
    /* Create an Environment using default reallocator. */
    int eid = fang_env_create(FANG_ENV_TYPE_CPU, NULL);
    assert_true(FANG_ISOK(eid));

    /* Should be able to retrieve the Environment structure. */
    fang_env_t *env;
    assert_true(FANG_ISOK(_fang_env_retrieve(&env, eid)));

    /* Type should match. */
    assert_int_equal(env->type, FANG_ENV_TYPE_CPU);

    /* No tensors should be allocated. */
    assert_int_equal(env->ntens, 0);

    /* Default reallocator should be assigned. */
    assert_ptr_equal(env->realloc, _fang_default_reallocator);

    /* Private structure should be allocated. */
    assert_non_null(env->private);

    /* Tensor operators should be assigned. */
    assert_non_null(env->ops);

    fang_env_release(eid);
}

/* Environment structure release test. */
static void fang_env_release_test(void **state) {
    /* Should not be able to release a non-existent Environment. */
    assert_int_equal(fang_env_release(69), -FANG_NOENV);

    /* Invalid environment ID is not allowed. */
    assert_int_equal(fang_env_release(FANG_MAX_ENV + 1), -FANG_INVID);

    int eid = fang_env_create(FANG_ENV_TYPE_CPU, NULL);
    assert_true(FANG_ISOK(eid));

    fang_env_t *env;
    assert_true(FANG_ISOK(_fang_env_retrieve(&env, eid)));

    /* Deliberately set tensor amount. */
    env->ntens = 1;

    /* Environments cannot be released if tensors using it. */
    assert_int_equal(fang_env_release(eid), -FANG_NTENS);

    /* Should be successful. */
    env->ntens = 0;
    assert_true(FANG_ISOK(fang_env_release(eid)));
}

/* CPU type Environment test. */
static void fang_env_cpu_test(void **state) {
    int eid = fang_env_create(FANG_ENV_TYPE_CPU, NULL);
    assert_true(FANG_ISOK(eid));

    /* Retrieve the Environment structure. */
    fang_env_t *env;
    assert_true(FANG_ISOK(_fang_env_retrieve(&env, eid)));

    /* Private CPU structure. */
    _fang_env_cpu_t *cpu_private = (_fang_env_cpu_t *) env->private;

    /* Valid number of processor count should be set. */
    assert_true(cpu_private->nproc > 0);

    /* All processors should be active at first. */
    assert_int_equal(cpu_private->nproc, cpu_private->nact);

    // TODO: Test CPU Environment control functions

    fang_env_release(eid);
}

int main() {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(fang_env_create_test),
        cmocka_unit_test(fang_env_release_test),
        cmocka_unit_test(fang_env_cpu_test)
    };

    return cmocka_run_group_tests_name("unit/environment", tests, NULL, NULL);
}
