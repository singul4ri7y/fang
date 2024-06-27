#include <fang/fang.h>
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <time.h>

int main() {
    printf("Fang version: %s\n", FANG_VERSION());

    int platform = fang_platform_create(FANG_PLATFORM_TYPE_CPU, NULL);
    uint32_t dims[] = { 15, 15 };
    fang_float *data =  malloc(225 * sizeof(fang_float));;
    for(int i = 0; i < 225; i++) 
        data[i] = (fang_float) i + 1;

    struct timespec start, end;
    fang_ten_t ten_a;
    fang_ten_t ten_b;
    fang_ten_create(&ten_a, platform, FANG_TEN_DTYPE_FLOAT32, dims, sizeof(dims) / sizeof(dims[0]), data);
    fang_ten_create(&ten_b, platform, FANG_TEN_DTYPE_FLOAT32, dims, sizeof(dims) / sizeof(dims[0]), data);
    FANG_TEN_PRINT(&ten_a);
    clock_gettime(CLOCK_MONOTONIC, &start);  // Start the timer
    fang_ten_hadamard(&ten_a, &ten_a, &ten_b);


    
    // fang_ten_rand(&ten, FANG_I2G(0), FANG_I2G(10));
    clock_gettime(CLOCK_MONOTONIC, &end);  // End the timer
    FANG_TEN_PRINT(&ten_b);
    FANG_TEN_PRINT(&ten_a);

    // Calculate the elapsed time
    double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Time taken: %f seconds\n", time_taken);
    
    fang_ten_release(&ten_a);
    fang_ten_release(&ten_b);

    fang_platform_release(platform);
    free(data);

    return 0;
}
