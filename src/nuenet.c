#include <time.h>

#define NUENET_IMPLEMENTATION
#include "lib/nuenet.h"

int main(void) {
    srand(time(0));

    float data_set[] = {
        0, 0,    0, 0,    0, 0,    0,
        0, 0,    0, 1,    0, 1,    0,
        0, 0,    1, 0,    1, 0,    0,
        0, 0,    1, 1,    1, 1,    0,
        0, 1,    0, 0,    0, 1,    0,
        0, 1,    0, 1,    1, 0,    0,
        0, 1,    1, 0,    1, 1,    0,
        0, 1,    1, 1,    0, 0,    1,
        1, 0,    0, 0,    1, 0,    0,
        1, 0,    0, 1,    1, 1,    0,
        1, 0,    1, 0,    0, 0,    1,
        1, 0,    1, 1,    0, 1,    1,
        1, 1,    0, 0,    1, 1,    0,
        1, 1,    0, 1,    0, 0,    1,
        1, 1,    1, 0,    0, 1,    1,
        1, 1,    1, 1,    1, 0,    1
    };

    nuenet_mat_t data = nuenet_mat_construct(16, 7, data_set);
    nuenet_mat_t input = nuenet_mat_sub(data, 16, 4, 0, 0);
    nuenet_mat_t output = nuenet_mat_sub(data, 16, 3, 0, 4);

    uint64_t blueprint[] = { 4, 4, 3 };

    nuenet_nn_t nn = nuenet_nn_construct(blueprint, NUENET_ARSIZ(blueprint));
    nuenet_nn_randf(nn, 0, 1);

    nuenet_nn_grad_t grad = nuenet_nn_grad_construct(nn);

    printf("cost before = %f\n", nuenet_nn_cost(nn, input, output));

    for(int i = 0; i < 1000000; i++) {
        nuenet_nn_back_prop(nn, grad, input, output);
        nuenet_nn_apply_batch_gd(nn, grad, 1);
    }

    printf("cost after = %f\n", nuenet_nn_cost(nn, input, output));
    for(int i = 0; i < 2; i++) {
    for(int j = 0; j < 2; j++) {
    for(int k = 0; k < 2; k++) {
    for(int l = 0; l < 2; l++) {
        // xor.
        float in[] = { (float) i, (float) j, (float) k, (float) l };
        nuenet_mat_t fp_out = nuenet_nn_ford_prop(nn, nuenet_mat_construct(1, 4, in));

        printf("(%d, %d) + (%d, %d) = (%.2f, %.2f, %0.2f)\n", i, j, k, l,
            nuenet_mat_get(fp_out, 0, 0),
            nuenet_mat_get(fp_out, 0, 1),
            nuenet_mat_get(fp_out, 0, 2));
    }
    }
    }
    }
    nuenet_nn_grad_destruct(grad);
    nuenet_nn_destruct(nn);

    return 0;
}
