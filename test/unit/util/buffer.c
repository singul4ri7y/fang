#include <fang/util/buffer.h>
#include <fang/status.h>
#include <memory.h>
#include <stddef.h>
#include <stdarg.h>
#include <setjmp.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>
#include <cmocka.h>

/* Buffer creation test. */
static void fang_buffer_create_test(void **state) {
    fang_buffer_t buff;
    int status = FANG_BUFFER_CREATE(&buff, _fang_default_reallocator, int);

    /* Buffer creation should succeed. */
    assert_int_equal(status, FANG_OK);

    /* Buffer reallocator should be properly set up. */
    assert_ptr_equal(buff.realloc, _fang_default_reallocator);

    /* Buffer count should be 0. */
    assert_int_equal(buff.count, 0);

    /* Initial buffer capacity should be `FANG_BUFFER_INIT_CAPACITY`. */
    assert_int_equal(buff.capacity, FANG_BUFFER_INIT_CAPACITY);

    /* Buffer element size should be `sizeof(int)` in this test. */
    assert_int_equal(buff.n, sizeof(int));

    fang_buffer_release(&buff);
}

/* Adding elements to buffer test. */
static void fang_buffer_add_test(void **state) {
    fang_buffer_t buff;
    FANG_BUFFER_CREATE(&buff, _fang_default_reallocator, int);

    int value = 69;

    /* Adding element should succeed. */
    assert_int_equal(fang_buffer_add(&buff, &value), FANG_OK);

    /* Try adding another_value. */
    int another_value = 1101;
    assert_true(FANG_ISOK(fang_buffer_add(&buff, &another_value)));

    /* Buffer count should be 2. */
    assert_int_equal(buff.count, 2);

    /* The capacity should be greater than 2; no memory should allocate. */
    assert_int_equal(buff.capacity, FANG_BUFFER_INIT_CAPACITY);

    /* Try adding multitude of elements, should allocate more memory. */
    for(int i = 0; i < FANG_BUFFER_INIT_CAPACITY; i++)
        /* Should succeed. */
        assert_true(FANG_ISOK(fang_buffer_add(&buff, &value)));

    /* At this point buffer count should be `FANG_BUFFER_INIT_CAPACITY + 2`. */
    assert_int_equal(buff.count, FANG_BUFFER_INIT_CAPACITY + 2);

    /* Buffer capacity should have 2 factor exponential growth. */
    assert_int_equal(buff.capacity, 2 * FANG_BUFFER_INIT_CAPACITY);

    /* Retrieve the data and compare. */
    int *data = (int *) buff.data;
    /* Values should be equal. */
    assert_int_equal(data[0], value);
    assert_int_equal(data[1], another_value);

    for(int i = 0; i < FANG_BUFFER_INIT_CAPACITY; i++)
        assert_int_equal(data[i + 2  /* Already checked 2 values */], value);

    fang_buffer_release(&buff);
}

/* Concatenating elements (strings most of the cases) to buffer test. */
static void fang_buffer_concat_test(void **state) {
    fang_buffer_t buff;
    FANG_BUFFER_CREATE(&buff, _fang_default_reallocator, char);

    /* Concatenation should succeed. */
    assert_true(FANG_ISOK(fang_buffer_concat(&buff, "WorkHard")));

    /* Buffer count should be 8. */
    assert_int_equal(buff.count, 8);

    /* Should be no allocation. */
    assert_int_equal(buff.capacity, FANG_BUFFER_INIT_CAPACITY);

    /* Data string should be equal. */
    assert_int_equal(strncmp(buff.data, "WorkHard", 8), 0);

    /* Large string concatenation. */
    assert_true(FANG_ISOK(fang_buffer_concat(&buff,
        "LifeIsAnAdventure1234KeepPushingBoundaries5678DiscoverNewHorizon"
        "s9990StayCuriousAndAlwaysAimForGreatness888ExploreWithoutLimits!")));

    /* Buffer count at this point should be 136. */
    assert_int_equal(buff.count, 136);

    /* Capacity should be 4x the initial in this case (2 factor exponential
       growth). */
    assert_int_equal(buff.capacity, 4 * FANG_BUFFER_INIT_CAPACITY);

    /* Add a NULL terminator character and compare. */
    char nc = '\0';
    assert_true(FANG_ISOK(fang_buffer_add(&buff, &nc)));
    assert_int_equal(strcmp(buff.data,
        "WorkHardLifeIsAnAdventure1234KeepPushingBoundaries5678DiscoverNewHor"
        "izons9990StayCuriousAndAlwaysAimForGreatness888ExploreWithoutLimits!"
    ), 0);

    fang_buffer_release(&buff);
}

/* Appending elements to buffer test. */
void fang_buffer_append_test(void **state) {
    fang_buffer_t buff;
    FANG_BUFFER_CREATE(&buff, _fang_default_reallocator, int);

    int BUFFER_SIZE = 136;
    int append_buff[BUFFER_SIZE];
    for(int i = 0; i < BUFFER_SIZE; i++)
        append_buff[i] = rand() % BUFFER_SIZE;

    /* First append 5 elements. */
    assert_true(FANG_ISOK(fang_buffer_append(&buff, append_buff, 5)));

    /* Buffer count should be 5. */
    assert_int_equal(buff.count, 5);

    /* Should be no allocation. */
    assert_int_equal(buff.capacity, FANG_BUFFER_INIT_CAPACITY);

    /* First 5 elements of the buffer should be equal. */
    assert_int_equal(memcmp(buff.data, append_buff, 5 * sizeof(int)), 0);

    /* Now append the whole array. */
    assert_true(FANG_ISOK(fang_buffer_append(&buff, append_buff + 5,
        BUFFER_SIZE - 5)));

    /* Count should be 128. */
    assert_int_equal(buff.count, BUFFER_SIZE);

    /* Capacity should be 4x the initial in this case (2 factor exponential
       growth). */
    assert_int_equal(buff.capacity, 4 * FANG_BUFFER_INIT_CAPACITY);

    /* Final comparison. */
    assert_int_equal(memcmp(buff.data, append_buff,
        BUFFER_SIZE * sizeof(int)), 0);

    fang_buffer_release(&buff);
}

/* Getting element from buffer test. */
void fang_buffer_get_test(void **state) {
    fang_buffer_t buff;
    FANG_BUFFER_CREATE(&buff, _fang_default_reallocator, float);

    /* Append elements. */
    int BUFFER_SIZE = 128;
    float elements[BUFFER_SIZE];
    for(int i = 0; i < BUFFER_SIZE; i++)
        elements[i] = ((float) rand() / RAND_MAX) * BUFFER_SIZE;

    fang_buffer_append(&buff, elements, BUFFER_SIZE);

    /* Comparison using `fang_buffer_get` function. */
    for(int i = 0; i < BUFFER_SIZE; i++) {
        assert_float_equal(*FANG_BUFFER_GET(&buff, float, i),
            elements[i], 1e-6);
    }

    /* Comparison with negative indicies. */
    for(int i = 1; i <= BUFFER_SIZE; i++) {
        assert_float_equal(*FANG_BUFFER_GET(&buff, float, -i),
            elements[BUFFER_SIZE - i], 1e-6);
    }

    fang_buffer_release(&buff);
}

/* Buffer data and size retrieve test. */
void fang_buffer_retrieve_test(void **state) {
    fang_buffer_t buff;
    FANG_BUFFER_CREATE(&buff, _fang_default_reallocator, int);

    /* Append some elements. */
    int elements[] = { 69, 55, 12, 18, 13 };
    fang_buffer_append(&buff, elements, 5);

    size_t size;
    void *data = fang_buffer_retrieve(&buff, &size);

    /* Data pointer should match. */
    assert_ptr_equal(buff.data, data);

    /* Element count should match as well. */
    assert_int_equal(buff.count, size);

    fang_buffer_release(&buff);
}

/* Buffer shrink test. */
void fang_buffer_shrink_to_fit_test(void **state) {
    fang_buffer_t buff;
    FANG_BUFFER_CREATE(&buff, _fang_default_reallocator, int);

    double elements[] = { 69.6969, 3.1416, 2.71828, 0.001 };
    fang_buffer_append(&buff, elements, 4);

    /* Initial capacity should match. */
    assert_int_equal(buff.capacity, FANG_BUFFER_INIT_CAPACITY);

    /* Shrink the capacity to match buffer count. */
    fang_buffer_shrink_to_fit(&buff);

    /* Buffer count should match capacity. */
    assert_int_equal(buff.count, buff.capacity);
    assert_int_equal(buff.count, 4);

    fang_buffer_release(&buff);
}

int main() {
    /* Change random number seed. */
    srand(time(0));

    const struct CMUnitTest tests[] = {
        cmocka_unit_test(fang_buffer_create_test),
        cmocka_unit_test(fang_buffer_add_test),
        cmocka_unit_test(fang_buffer_concat_test),
        cmocka_unit_test(fang_buffer_append_test),
        cmocka_unit_test(fang_buffer_get_test),
        cmocka_unit_test(fang_buffer_retrieve_test),
        cmocka_unit_test(fang_buffer_shrink_to_fit_test)
    };

    return cmocka_run_group_tests_name("fang_buffer", tests, NULL, NULL);
}
