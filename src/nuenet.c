#include <time.h>

#define NUENET_IMPLEMENTATION
#include "lib/nuenet.h"

typedef struct Xor {
    nuenet_mat_t *w;
    nuenet_mat_t *b;
    nuenet_mat_t *intm;
    nuenet_mat_t *diff_w;
    nuenet_mat_t *diff_b;
    uint64_t inputs;
    uint64_t outputs;
    uint64_t layers;
} Model;

typedef float sample[3];

void construct(Model *model, int *blueprint, uint64_t len) {
    if(len == 0) return;

    // First layer is the input layers.
    model -> inputs = blueprint[0];
    model -> outputs = blueprint[len - 1];
    len--;
    blueprint++;

    model -> layers = len;

    // Allocation.
    
    model -> w = (nuenet_mat_t *) malloc(len * sizeof(nuenet_mat_t));
    model -> b = (nuenet_mat_t *) malloc(len * sizeof(nuenet_mat_t));
    model -> diff_w = (nuenet_mat_t *) malloc(len * sizeof(nuenet_mat_t));
    model -> diff_b = (nuenet_mat_t *) malloc(len * sizeof(nuenet_mat_t));
    model -> intm = (nuenet_mat_t *) malloc(len * sizeof(nuenet_mat_t));

    for(uint64_t i = 0; i < len; i++) {
        model -> w[i] = nuenet_mat_construct(blueprint[i - 1], blueprint[i], NULL);
        model -> diff_w[i] = nuenet_mat_construct(blueprint[i - 1], blueprint[i], NULL);
        nuenet_mat_randf(model -> w[i], 0, 1);

        model -> b[i] = nuenet_mat_construct(1, blueprint[i], NULL);
        model -> diff_b[i] = nuenet_mat_construct(1, blueprint[i], NULL);
        nuenet_mat_randf(model -> b[i], 0, 1);

        model -> intm[i] = nuenet_mat_construct(1, blueprint[i], NULL);
    }
}

void destruct(Model *model) {
    for(uint64_t i = 0; i < model -> layers; i++) {
        nuenet_mat_destruct(model -> w[i]);
        nuenet_mat_destruct(model -> b[i]);
        nuenet_mat_destruct(model -> diff_w[i]);
        nuenet_mat_destruct(model -> diff_b[i]);
        nuenet_mat_destruct(model -> intm[i]);
    }

    free(model -> w);
    free(model -> b);
    free(model -> diff_w);
    free(model -> diff_b);
    free(model -> intm);
}

nuenet_mat_t ford(Model *model, nuenet_mat_t input) {
    nuenet_mat_shape_t shape = nuenet_mat_get_shape(input);

    assert(shape.row == 1);
    assert(shape.col == model -> inputs);

    nuenet_mat_t acc = input;

    for(uint64_t i = 0; i < model -> layers; i++) {
        nuenet_mat_dot(model -> intm[i], acc, model -> w[i]);
        nuenet_mat_sum(model -> intm[i], model -> intm[i], model -> b[i]);
        nuenet_mat_sigmf(model -> intm[i]);

        acc = model -> intm[i];
    }

    return acc;
}

float cost(Model *model, nuenet_mat_t input, nuenet_mat_t output) {
    assert(input.shape.row == output.shape.row);
    assert(output.shape.col == model -> outputs);

    float result = 0;

    for(uint64_t  i = 0; i < input.shape.row; i++) {
        nuenet_mat_t fp_output = ford(model, nuenet_mat_row(input, i));
        // NUENET_MAT_PRINT(nuenet_mat_row(input, i));
        // NUENET_MAT_PRINT(fp_output);

        for(uint64_t j = 0; j < output.shape.col; j++) {
            float diff = nuenet_mat_get(output, i, j) - nuenet_mat_get(fp_output, 0, j);

            result += diff * diff;
        }
    }

    return sqrt(result / input.shape.row * model -> outputs);
}

void finite_diff(Model *model, float eps, nuenet_mat_t input, nuenet_mat_t output) {
    float saved;
    for(uint64_t i = 0; i < model -> layers; i++) {
        nuenet_mat_shape_t shape = nuenet_mat_get_shape(model -> w[i]);

        float c = cost(model, input, output);

        for(uint64_t row = 0; row < shape.row; row++) {
            for(uint64_t col = 0; col < shape.col; col++) {
                saved = nuenet_mat_get(model -> w[i], row, col);
                nuenet_mat_set(model -> w[i], row, col, saved + eps);
                float new_cost = cost(model, input, output);
                nuenet_mat_set(model -> diff_w[i], row, col, (new_cost - c) / eps);
                nuenet_mat_set(model -> w[i], row, col, saved);
            }
        }

        shape = nuenet_mat_get_shape(model -> b[i]);

        for(uint64_t col = 0; col < shape.col; col++) {
            saved = nuenet_mat_get(model -> b[i], 0, col);
            nuenet_mat_set(model -> b[i], 0, col, saved + eps);
            float new_cost = cost(model, input, output);
            nuenet_mat_set(model -> diff_b[i], 0, col, (new_cost - c) / eps);
            nuenet_mat_set(model -> b[i], 0, col, saved);
        }
    }
}

void learn(Model *model, uint64_t iter, float eps, float rate, nuenet_mat_t input, nuenet_mat_t output) {
    for(uint64_t i = 0; i < iter; i++) {
        finite_diff(model, eps, input, output);

        for(uint64_t j = 0; j < model -> layers; j++) {
            nuenet_mat_shape_t shape = nuenet_mat_get_shape(model -> w[j]);

            for(uint64_t row = 0; row < shape.row; row++) {
                for(uint64_t col = 0; col < shape.col; col++) {
                    nuenet_mat_set(model -> w[j], row, col, nuenet_mat_get(model -> w[j], row, col) - 
                            rate * nuenet_mat_get(model -> diff_w[j], row, col));
                    // printf("w[%lu, %lu] = %f ", row, col, nuenet_mat_get(model -> w[j], row, col));
                }
            }

            // printf("\n");

            shape = nuenet_mat_get_shape(model -> b[j]);
            
            for(uint64_t col = 0; col < shape.col; col++) {
                nuenet_mat_set(model -> b[j], 0, col, nuenet_mat_get(model -> b[j], 0, col) - 
                        rate * nuenet_mat_get(model -> diff_b[j], 0, col));
            }
        }
    }
}

int main(void) {
    srand(time(0));

    float data_set[] = {
        0, 0, 0,
        1, 0, 1,
        0, 1, 1,
        1, 1, 0
    };
    uint64_t size = NUENET_ARSIZ(data_set);

    nuenet_mat_t data = nuenet_mat_construct(4, 3, data_set);
    nuenet_mat_t input = nuenet_mat_sub(data, 4, 2, 0, 0);
    nuenet_mat_t output = nuenet_mat_sub(data, 4, 1, 0, 2);
    NUENET_MAT_PRINT(output);

    uint64_t blueprint[] = { 2, 2, 1 };

    nuenet_nn_t nn = nuenet_nn_construct(blueprint, NUENET_ARSIZ(blueprint));
    nuenet_nn_randf(nn, 0, 1);
    nuenet_nn_fgrad_t grad = nuenet_nn_fgrad_construct(nn);
    printf("cost before = %f\n", nuenet_nn_cost(nn, input, output));
    for(int i = 0; i < 1000000; i++) {
        nuenet_nn_fgrad_get(nn, grad, 1e-1, input, output);
        nuenet_nn_fgrad_apply(nn, grad);
    }
    printf("cost after = %f\n", nuenet_nn_cost(nn, input, output));
    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 2; j++) {
            // xor.
            float in[] = { (float) i, (float) j };
            float value = nuenet_mat_get(nuenet_nn_ford_prop(nn, nuenet_mat_construct(1, 2, in)), 0, 0);

            printf("%d ^ %d = %.2f\n", i, j, value);
        }
    }
    nuenet_nn_fgrad_destruct(grad);
    nuenet_nn_destruct(nn);

    return 0;
}
