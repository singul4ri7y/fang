#include <fang/fang.h>
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>

int main() {
    printf("Fang version: %s\n", FANG_VERSION());

    int platform = fang_platform_create(FANG_PLATFORM_TYPE_CPU, NULL);
    fang_ten_t ten;
    uint32_t dims[] = { 4, 4, 4, 4, 4};
    fang_float *data = malloc(1024 * sizeof(fang_float));
    for(int i = 0; i < 1024; i++) data[i] = (double) i + 1;
    fang_ten_create(&ten, platform, FANG_TEN_DTYPE_FLOAT64, dims, sizeof(dims) / sizeof(dims[0]), data);
    FANG_TEN_PRINT(&ten);
    fang_ten_release(&ten);

    fang_platform_release(platform);
    free(data);

    return 0;
}
