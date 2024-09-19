#include <fang/config.h>
#include <memory.h>
#include <stdarg.h>
#include <setjmp.h>
#include <string.h>
#include <cmocka.h>

void fang_default_reallocator_test(void **state) {
    /* Allocate. */
    int *mem = FANG_CREATE(_fang_default_reallocator, int, 64);
    assert_non_null(mem);

    /* Address should be `FANG_MEMALIGN` byte aligned. */
    assert_false((uintptr_t) mem % FANG_MEMALIGN);

    int data[64];
    for(int i = 0; i < 64; i++)
        mem[i] = i - 32;

    /* Copy the data out to the memory. */
    memcpy(mem, data, 64 * sizeof(int));

    /* Extend the memory. */
    mem = _fang_default_reallocator(mem, 128 * sizeof(int));
    assert_non_null(mem);

    /* Address should be aligned. */
    assert_false((uintptr_t) mem % FANG_MEMALIGN);

    /* Memory should stay uncorrupted after extend. */
    assert_int_equal(memcmp(mem, data, 64 * sizeof(int)), 0);

    /* Now shrink the memory. */
    mem = _fang_default_reallocator(mem, 32 * sizeof(int));
    assert_non_null(mem);

    /* Memory should stay uncorrupted after shrink. */
    assert_int_equal(memcmp(mem, data, 32 * sizeof(int)), 0);

    FANG_RELEASE(_fang_default_reallocator, mem);
}

int main() {
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(fang_default_reallocator_test)
    };

    return cmocka_run_group_tests_name("memory", tests, NULL, NULL);
}
