#include <time.h>

#define NUENET_IMPLEMENTATION
#include "lib/nuenet.h"

int main(void) {
    srand(time(0));

    float data_set[] = {
        0, 0, 0,
        1, 0, 1,
        0, 1, 1,
        1, 1, 0
    };

    nuenet_mat_t data = nuenet_mat_construct(4, 3, data_set);
    nuenet_mat_t input = nuenet_mat_sub(data, 4, 2, 0, 0);
    nuenet_mat_t output = nuenet_mat_sub(data, 4, 1, 0, 2);

    uint64_t blueprint[] = { 2, 2, 1 };

    nuenet_nn_t nn = nuenet_nn_construct(blueprint, NUENET_ARSIZ(blueprint));
    nuenet_nn_randf(nn, 0, 1);
    NUENET_NN_PRINT(nn);
    nuenet_nn_grad_t grad = nuenet_nn_grad_construct(nn);
    printf("cost before = %f\n", nuenet_nn_cost(nn, input, output));
    for(int i = 0; i < 1000000; i++) {
        // nuenet_nn_fgrad_get(nn, grad, 1e-2, input, output);
        // nuenet_nn_grad_print(grad, "finite diff");
        nuenet_nn_back_prop(nn, grad, input, output);
        // nuenet_nn_grad_print(grad, "backprop");
        nuenet_nn_apply_batch_gd(nn, grad, 1);
    }
    NUENET_NN_PRINT(nn);
    printf("cost after = %f\n", nuenet_nn_cost(nn, input, output));
    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 2; j++) {
            // xor.
            float in[] = { (float) i, (float) j };
            float value = nuenet_mat_get(nuenet_nn_ford_prop(nn, nuenet_mat_construct(1, 2, in)), 0, 0);

            printf("%d ^ %d = %.2f\n", i, j, value);
        }
    }
    nuenet_nn_grad_destruct(grad);
    nuenet_nn_destruct(nn);

    return 0;
}
