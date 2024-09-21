#include <fang/status.h>
#include <env/cpu.h>
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
    _fang_env_cpu_t  *cpu_private = (_fang_env_cpu_t *) env->private;

    /* Should have valid CPU information. */
    assert_non_null(cpu_private->cpu);
    assert_int_not_equal(cpu_private->ncpu, 0);

    /* All physical CPU(s) should be active at first. */
    assert_int_equal(cpu_private->ncpu, cpu_private->ncact);

    /* All physical CPU information should be properly set. */
    _fang_cpu_t *cpu = cpu_private->cpu;
    int sproc = 0;  // To match processor stride calculations
    for(int i = 0; i < cpu_private->ncpu; i++) {
        /* Should have atleast one logical CPU. */
        assert_int_not_equal(cpu[i].nproc, 0);
        /* Initially all logical CPU(s) should be active. */
        assert_int_equal(cpu[i].nproc, cpu[i].nact);

        /* Match logical CPU (CPU cores) stride calculations. */
        assert_int_equal(cpu[i].sproc, sproc);
        sproc += cpu[i].nproc;
    }

    fang_env_release(eid);
}

int main() {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(fang_env_create_test),
        cmocka_unit_test(fang_env_release_test),
        cmocka_unit_test(fang_env_cpu_test)
    };

    return cmocka_run_group_tests_name("environment", tests, NULL, NULL);
}
