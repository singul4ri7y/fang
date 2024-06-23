#include <fang/fang.h>
#include <stdio.h>
#include <stddef.h>

int main() {
    printf("Fang version: %s\n", FANG_VERSION());

    int platform = fang_platform_create(FANG_PLATFORM_TYPE_CPU, NULL);
    fang_platform_release(platform);

    return 0;
}
