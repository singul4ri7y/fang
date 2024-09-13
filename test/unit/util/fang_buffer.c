#include <criterion/criterion.h>
#include <fang/util/buffer.h>
#include <fang/status.h>
#include <memory.h>

/* Buffer creation */
Test(fang_buffer, create) {
    fang_buffer_t buff;
    int status = FANG_BUFFER_CREATE(&buff, _fang_default_reallocator, int);
    cr_assert_eq(status, FANG_OK, "Buffer creation should succeed");
    cr_assert_eq(buff.realloc, _fang_default_reallocator, "Buffer reallocator "
        "should be properly set up");
    cr_assert_eq(buff.count, 0, "Buffer count should be 0");
    cr_assert_eq(buff.capacity, FANG_BUFFER_INIT_CAPACITY, "Initial buffer "
        "capacity should be equal to %d", FANG_BUFFER_INIT_CAPACITY);
    cr_assert_eq(buff.n, sizeof(int), "Buffer element size should be %d in "
        "this test", sizeof(int));
}
