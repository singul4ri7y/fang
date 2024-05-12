#ifndef NUENET_H
#define NUENET_H

#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>

/* If NUENET_ALLOC is defined, it is also expected that
 * NUENET_FREE is defined as well.
 */
#ifndef NEUNET_ALLOC
#include <stdlib.h>

#define NUENET_ALLOC(type, x)    malloc((x) * sizeof(type))
#define NUENET_FREE(addr)        free(addr)
#endif // NUENET_ALLOC

/* For assertion. */
#ifndef NUENET_ASSERT
#include <assert.h>

#define NUENET_ASSERT            assert
#endif // NUENET_ASSERT

/* ---------------- MISCELLANEOUS ---------------- */

/* Only use it on C arrays, not pointers. */
#define NUENET_ARSIZ(arr) (sizeof(arr) / sizeof(*arr))

/* Returns a random float value. */
float nuenet_randf();

/* ---------------- MISCELLANEOUS END ---------------- */

/* ---------------- MATRIX ---------------- */

/* A nicer way to print the matrix. */
#define NUENET_MAT_PRINT(mat) nuenet_mat_print(mat, #mat, 0)

/* Stores the dimension of a matrix. */
typedef struct nuenet_mat_shape {
    uint64_t row;
    uint64_t col;
} nuenet_mat_shape_t;

typedef struct nuenet_mat {
    nuenet_mat_shape_t shape;
    struct {
        uint64_t stride;
        uint64_t start_row;
        uint64_t start_col;
    } sub;
    float *data;
} nuenet_mat_t;

/* Consturcts and returns a new matrix. */
/* Pass a NULL pointer to the data to create a new matrix in the heap. */
nuenet_mat_t       nuenet_mat_construct(uint64_t rows, uint64_t cols, void *data);

/* Creates a new matrix structure. */
/* Note: It does not do any sort of allocation, it just fills in the 
 *       'nuenet_mat_t' structure.
 */
nuenet_mat_t       nuenet_mat_new(uint64_t rows, uint64_t cols, uint64_t stride, 
    uint64_t start_row, uint64_t start_col, void *data);

/* Returns a sub-matrix of the given matrix. */
/* Note: Do not call 'nuenet_mat_destruct()' on a sub-matrix. sub-matrices
 *       shares data memory of the parent matrix.
 */
nuenet_mat_t       nuenet_mat_sub(nuenet_mat_t mat, uint64_t rows, uint64_t cols, 
    uint64_t start_row, uint64_t start_col);

/* Returns a specific row matrix sub-matrix. */
nuenet_mat_t       nuenet_mat_row(nuenet_mat_t mat, uint64_t row);

/* Returns a specific row matrix sub-matrix. */
/* 0 based index. */
nuenet_mat_t       nuenet_mat_col(nuenet_mat_t mat, uint64_t col);

/* Gets an element from the matrix. 0 based indexing. */
/* 0 based index. */
float              nuenet_mat_get(nuenet_mat_t mat, uint64_t row, uint64_t col);

/* Sets a value to a matrix. 0 based indexing. */
void               nuenet_mat_set(nuenet_mat_t mat, uint64_t row, uint64_t col, float value);

/* Initializes a matrix with randomized values. */
void               nuenet_mat_randf(nuenet_mat_t mat, float low, float high);

/* Fill the matrix with the provided value. */
void               nuenet_mat_fill(nuenet_mat_t mat, float value);

/* Release if the matrix is created in the heap. */
void               nuenet_mat_destruct(nuenet_mat_t mat);

/* Gets the dimension of a matrix. */
nuenet_mat_shape_t nuenet_mat_get_shape(nuenet_mat_t mat);

/* Dot product of two matrices. Destination matrix is expected to be allocated
 * with proper shape.
 * Returns the dest matrix.
 */
nuenet_mat_t       nuenet_mat_dot(nuenet_mat_t dest, nuenet_mat_t a, nuenet_mat_t b);

/* Summation of two matrices. */
/* Saves the summation to matrix 'dest' and returns the matrix. */
nuenet_mat_t       nuenet_mat_sum(nuenet_mat_t dest, nuenet_mat_t a, nuenet_mat_t b);

/* Returns true of provided matrices conform (same shape) each other. */
bool               nuenet_mat_conf(nuenet_mat_t a, nuenet_mat_t b);

/* Returns true if provided matrices are equal. */
bool               nuenet_mat_equal(nuenet_mat_t a, nuenet_mat_t b);

/* Prints a matrix. */
void               nuenet_mat_print(nuenet_mat_t matrix, 
        const char *const name, int padding);

/* ---------------- MATRIX END ---------------- */

/* ---------------- NEURAL NETWORKS ---------------- */

#define NUENET_NN_PRINT(nn) nuenet_nn_print(nn, #nn)

typedef struct nuenet_nn {
    /* The weight and bias matrices. */
    nuenet_mat_t *weight;
    nuenet_mat_t *bias;

    /* These matrices store the intermediate values of
       operation. */
    nuenet_mat_t *intm;

    /* Number of layers, inputs and outputs. */
    uint64_t ins;
    uint64_t outs;
    uint64_t layers;
} nuenet_nn_t;

/* Stores the finite gradient. */
typedef struct nuenet_nn_fgrad {
    /* We will store finite gradient only for the weights and biases. */
    nuenet_mat_t *weight;
    nuenet_mat_t *bias;

    /* The amount of layers the NN has. */
    uint64_t layers;
} nuenet_nn_fgrad_t;

/* Creates and returns a new neural network. */
nuenet_nn_t       nuenet_nn_construct(uint64_t *blueprint, uint64_t len);

/* Frees an allocated neural network. */
void              nuenet_nn_destruct(nuenet_nn_t nn);

/* Fills the the weights and biases with random values. */
nuenet_nn_t       nuenet_nn_randf(nuenet_nn_t nn, uint64_t lo, uint64_t hi);

/* Prints the whole network. */
void              nuenet_nn_print(nuenet_nn_t nn, const char *const nn_name);

/* Forward propagation. */
nuenet_mat_t      nuenet_nn_ford_prop(nuenet_nn_t nn, nuenet_mat_t input);

/* Returns the current cost of the model for the provided dataset. */
float             nuenet_nn_cost(nuenet_nn_t nn, nuenet_mat_t ins, 
        nuenet_mat_t outs);

/* Creates and returns a new finite gradient structure for a neural net. */
nuenet_nn_fgrad_t nuenet_nn_fgrad_construct(nuenet_nn_t nn);

/* Frees/Releases the finite gradient structure. */
void              nuenet_nn_fgrad_destruct(nuenet_nn_fgrad_t fgrad);

/* Returns the gradient of current neural net using finite difference. */
nuenet_nn_fgrad_t nuenet_nn_fgrad_get(nuenet_nn_t nn, nuenet_nn_fgrad_t fgrad, 
        float eps, nuenet_mat_t ins, nuenet_mat_t outs);

/* Applies the finite gradient. */
nuenet_nn_t       nuenet_nn_fgrad_apply(nuenet_nn_t nn, 
        nuenet_nn_fgrad_t fgrad);

/* ---------------- NEURAL NETWORKS END ---------------- */

/* ---------------- ACTIVATION FUNCTIONS ---------------- */

/* The sigmoid function. */
float        nuenet_sigmf(float x);

/* Apply sigmoid to the whole matrix. */
nuenet_mat_t nuenet_mat_sigmf(nuenet_mat_t mat);

/* ---------------- ACTIVATION FUNCTIONS END ---------------- */

#endif // NUENET_H

#if defined(NUENET_IMPLEMENTATION) && !defined(NUENET_IMPLEMENTED)
#define NUENET_IMPLEMENTED

/* ---------------- PRIVATE ---------------- */

/* Call simplification. */
#define _MAT_GET(mat, row, col)           nuenet_mat_get(mat, row, col)

/* Call simplification. */
#define _MAT_SET(mat, row, col, value)    nuenet_mat_set(mat, row, col, (value))

/* For interation simplification. */
#define _MAT_ITER(mat, dimen)             for(uint64_t dimen = 0u;             \
                                            dimen < (mat).shape.dimen; dimen++)

/* Gets the raw pointer to the matrix data. */
static float   *nuenet__mat_get_data(nuenet_mat_t mat) {
    return (float *) ((uint64_t) mat.data & ~((uint64_t) 1 << 63));
}

/* Calculates size of a matrix or a sub-matrix. */
static uint64_t nuenet__mat_get_size(nuenet_mat_t mat) {
    return (mat.shape.col + mat.sub.start_col) * mat.shape.row 
          + mat.sub.stride * mat.sub.start_row;
}

/* ---------------- PRIVATE END ---------------- */

/* ---------------- MISCELLANEOUS ---------------- */

/* Returns a random float value. */
float nuenet_randf() {
    return (float) rand() / RAND_MAX;
}

/* ---------------- MISCELLANEOUS END ---------------- */

/* ---------------- MATRIX ---------------- */

/* Creates and returns a new matrix. */
/* Pass a NULL pointer to the data to create a new matrix in the heap. */
nuenet_mat_t nuenet_mat_construct(uint64_t rows, uint64_t cols, void *data) {
    nuenet_mat_t mat = nuenet_mat_new(rows, cols, cols, 0, 0, NULL);

    /* If we exclude the last high bit, it's still roughly 
     * 8 Exabytes of addressable memory, which is, well... 
     * still a lot.
     */
    if(data == NULL) {
        void* buffer = NUENET_ALLOC(*mat.data, nuenet__mat_get_size(mat));

        NUENET_ASSERT(buffer != NULL);

        data = (void *) ((uint64_t) 1 << 63 |
                         (uint64_t) buffer);
    }

    mat.data = data;

    return mat;
}

/* Constructs a new matrix structure. */
/* Note: It does not do any sort of allocation, it just fills in the 
 *       'nuenet_mat_t' structure.
 */
nuenet_mat_t nuenet_mat_new(uint64_t rows, uint64_t cols, uint64_t stride, 
        uint64_t start_row, uint64_t start_col, void *data) 
{
    nuenet_mat_t mat = {
        .shape = { .row = rows, .col = cols },
        .sub   = { 
            .stride    = stride,
            .start_col = start_col, 
            .start_row = start_row
        },
        .data = data
    };

    return mat;
}

/* Returns a sub-matrix of the given matrix. */
/* Note: Do not call 'nuenet_mat_destruct()' on a sub-matrix. sub-matrices
 *       shares data memory of the parent matrix.
 */
nuenet_mat_t nuenet_mat_sub(nuenet_mat_t mat, uint64_t rows, uint64_t cols, 
    uint64_t start_row, uint64_t start_col) 
{
    NUENET_ASSERT(start_row + rows <= mat.shape.row);
    NUENET_ASSERT(start_col + cols <= mat.shape.col);

    return nuenet_mat_new(rows, cols, mat.sub.stride, start_row, 
        start_col, mat.data);
}

/* Returns a specific row matrix sub-matrix. */
/* 0 based index. */
nuenet_mat_t nuenet_mat_row(nuenet_mat_t mat, uint64_t row) {
    return nuenet_mat_sub(mat, 1, mat.shape.col, row, 0);
}

/* Returns a specific row matrix sub-matrix. */
/* 0 based index. */
nuenet_mat_t nuenet_mat_col(nuenet_mat_t mat, uint64_t col) {
    return nuenet_mat_sub(mat, mat.shape.row, 1, 0, col);
}

/* Gets an element from the matrix. 0 based indexing. */
float nuenet_mat_get(nuenet_mat_t mat, uint64_t row, uint64_t col) {
    NUENET_ASSERT(row < mat.shape.row);
    NUENET_ASSERT(col < mat.shape.col);

    return nuenet__mat_get_data(mat)[(mat.sub.start_row + row) 
        * mat.sub.stride + col + mat.sub.start_col];
}

/* Sets a value to a matrix. 0 based indexing. */
void nuenet_mat_set(nuenet_mat_t mat, uint64_t row, uint64_t col, float value) {
    NUENET_ASSERT(row < mat.shape.row);
    NUENET_ASSERT(col < mat.shape.col);

    nuenet__mat_get_data(mat)[(mat.sub.start_row + row) 
        * mat.sub.stride + col + mat.sub.start_col] = value;
}

/* Initializes a matrix with randomized values. */
void nuenet_mat_randf(nuenet_mat_t mat, float high, float low) {
    _MAT_ITER(mat, row) {
        _MAT_ITER(mat, col) 
            _MAT_SET(mat, row, col, nuenet_randf() * (high - low) + low);
    }
}

/* Fill the matrix with the provided value. */
void nuenet_mat_fill(nuenet_mat_t mat, float value) {
    _MAT_ITER(mat, row) {
        _MAT_ITER(mat, col) 
            _MAT_SET(mat, row, col, value);
    }
}

/* Free if the matrix is created in the heap. */
void nuenet_mat_destruct(nuenet_mat_t mat) {
    if((uint64_t) mat.data & (uint64_t) 1 << 63) 
        NUENET_FREE(nuenet__mat_get_data(mat));
}

/* Gets the dimension of a matrix. */
nuenet_mat_shape_t nuenet_mat_get_shape(nuenet_mat_t mat) {
    return mat.shape;
}

/* Dot product of two matrices. Destination matrix is expected to be allocated
 * with proper shape.
 * Returns the dest matrix.
 */
nuenet_mat_t nuenet_mat_dot(nuenet_mat_t dest, nuenet_mat_t a, nuenet_mat_t b) {
    NUENET_ASSERT(a.shape.col    == b.shape.row);
    NUENET_ASSERT(dest.shape.row == a.shape.row);
    NUENET_ASSERT(dest.shape.col == b.shape.col);

    _MAT_ITER(dest, row) {
        _MAT_ITER(dest, col) {
            float result = 0;

            for(uint64_t k = 0; k < a.shape.col; k++) 
                result += _MAT_GET(a, row, k) * _MAT_GET(b, k, col);

            _MAT_SET(dest, row, col, result);
        }
    }

    return dest;
}

/* Summation of two matrices. */
/* Saves the summation to matrix 'dest' and returns the matrix. */
nuenet_mat_t nuenet_mat_sum(nuenet_mat_t dest, nuenet_mat_t a, nuenet_mat_t b) {
    NUENET_ASSERT(a.shape.row    == b.shape.row);
    NUENET_ASSERT(a.shape.col    == b.shape.col);
    NUENET_ASSERT(dest.shape.row == a.shape.row);
    NUENET_ASSERT(dest.shape.col == a.shape.col);

    _MAT_ITER(dest, row) {
        _MAT_ITER(dest, col) {
            _MAT_SET(dest, row, col, 
            _MAT_GET(a, row, col) + _MAT_GET(b, row, col));
        }
    }

    return dest;
}

/* Returns true of provided matrices conform (same shape) each other. */
bool nuenet_mat_conf(nuenet_mat_t a, nuenet_mat_t b) {
    return a.shape.row == b.shape.row && a.shape.col == b.shape.col;
}

/* Returns true if provided matrices are equal. */
bool nuenet_mat_equal(nuenet_mat_t a, nuenet_mat_t b) {
    if(a.shape.row != b.shape.row || a.shape.col != b.shape.col) 
        return false;

    for(uint64_t i = 0; i < a.shape.row; i++) {
        for(uint64_t j = 0; j < a.shape.row; j++) {
            if(_MAT_GET(a, i, j) != _MAT_GET(b, i, j)) 
                return false;
        }
    }

    return true;
}

/* Prints a matrix. */
void nuenet_mat_print(nuenet_mat_t mat, const char *const name, int padding) {
    printf("%*s%s = [\n", padding, "", name);
    _MAT_ITER(mat, row) {
        printf("%*s", padding, "");
        _MAT_ITER(mat, col) 
            printf("%8.2f ", _MAT_GET(mat, row, col));

        printf("\n");
    }
    printf("%*s]\n", padding, "");
}

/* ---------------- MATRIX END ---------------- */

/* ---------------- NEURAL NETWORKS ---------------- */

/* Creates and returns a new neural network. */
nuenet_nn_t nuenet_nn_construct(uint64_t *blueprint, uint64_t len) {
    assert(len >= 2);

    nuenet_nn_t nn;

    /* The first layer are the inputs */
    nn.ins  = blueprint[0];
    nn.outs = blueprint[len - 1];

    /* We don't need to take the input layer into account now. */
    blueprint++, len--;
    nn.layers = len;

    /* Allocate the matrice buffers. */
    nn.weight = NUENET_ALLOC(nuenet_mat_t, len);
    NUENET_ASSERT(nn.weight != NULL);

    nn.bias = NUENET_ALLOC(nuenet_mat_t, len);
    NUENET_ASSERT(nn.bias != NULL);

    nn.intm = NUENET_ALLOC(nuenet_mat_t, len);
    NUENET_ASSERT(nn.intm != NULL);

    for(uint64_t i = 0; i < len; i++) {
        nn.weight[i] = nuenet_mat_construct(blueprint[i - 1], blueprint[i], NULL);
        nn.bias[i]   = nuenet_mat_construct(1, blueprint[i], NULL);
        nn.intm[i]   = nuenet_mat_construct(1, blueprint[i], NULL);
    }

    return nn;
}

/* Frees an allocated neural network. */
void nuenet_nn_destruct(nuenet_nn_t nn) {
    for(uint64_t i = 0; i < nn.layers; i++) {
        nuenet_mat_destruct(nn.weight[i]);
        nuenet_mat_destruct(nn.bias[i]);
        nuenet_mat_destruct(nn.intm[i]);
    }

    NUENET_FREE(nn.weight);
    NUENET_FREE(nn.bias);
    NUENET_FREE(nn.intm);
}

/* Fills the the weights and biases with random values. */
nuenet_nn_t nuenet_nn_randf(nuenet_nn_t nn, uint64_t lo, uint64_t hi) {
    for(uint64_t i = 0; i < nn.layers; i++) {
        nuenet_mat_randf(nn.weight[i], lo, hi);
        nuenet_mat_randf(nn.bias[i], lo, hi);
        nuenet_mat_randf(nn.intm[i], lo, hi);
    }

    return nn;
}

/* Prints the whole network. */
void nuenet_nn_print(nuenet_nn_t nn, const char *const nn_name) {
    char buf[128];

    printf("%s = [\n", nn_name);
    for(uint64_t i = 0; i < nn.layers; i++) {
        snprintf(buf, 128 * sizeof(char), "nn.weight[%lu]", i);
        nuenet_mat_print(nn.weight[i], buf, 4);
        snprintf(buf, 128 * sizeof(char), "nn.bias[%lu]", i);
        nuenet_mat_print(nn.bias[i], buf, 4);
        snprintf(buf, 128 * sizeof(char), "nn.intm[%lu]", i);
        nuenet_mat_print(nn.intm[i], buf, 4);
    }
    printf("]\n");
}

/* Forward propagation. */
nuenet_mat_t nuenet_nn_ford_prop(nuenet_nn_t nn, nuenet_mat_t input) {
    nuenet_mat_shape_t shape = nuenet_mat_get_shape(input);

    assert(shape.row == 1);
    assert(shape.col == nn.ins);

    nuenet_mat_t acc = input;

    for(uint64_t i = 0; i < nn.layers; i++) {
        nuenet_mat_dot(nn.intm[i], acc, nn.weight[i]);
        nuenet_mat_sum(nn.intm[i], nn.intm[i], nn.bias[i]);
        nuenet_mat_sigmf(nn.intm[i]);

        acc = nn.intm[i];
    }

    NUENET_ASSERT(acc.shape.col == nn.outs);

    return acc;
}

/* Returns the current cost of the model for the provided dataset. */
float nuenet_nn_cost(nuenet_nn_t nn, nuenet_mat_t ins, nuenet_mat_t outs) {
    nuenet_mat_shape_t shape = nuenet_mat_get_shape(outs);

    NUENET_ASSERT(ins.shape.row == shape.row);
    NUENET_ASSERT(shape.col == nn.outs);

    float result = 0.0f;

    for(uint64_t i = 0; i < shape.row; i++) {
        nuenet_mat_t fp_out = nuenet_nn_ford_prop(nn, nuenet_mat_row(ins, i));

        for(uint64_t j = 0; j < shape.col; j++) {
            float diff = _MAT_GET(outs, i, j) - _MAT_GET(fp_out, 0, j);

            result += diff * diff;
        }
    }

    return sqrt(result / shape.row * nn.outs);
}

/* Creates and returns a new finite gradient structure for a neural net. */
nuenet_nn_fgrad_t nuenet_nn_fgrad_construct(nuenet_nn_t nn) {
    nuenet_nn_fgrad_t fgrad;

    fgrad.weight = NUENET_ALLOC(nuenet_mat_t, nn.layers);
    NUENET_ASSERT(fgrad.weight != NULL);

    fgrad.bias = NUENET_ALLOC(nuenet_mat_t, nn.layers);
    NUENET_ASSERT(fgrad.bias != NULL);

    /* We need to know the amount of layers we have, usefull when freeing. */
    fgrad.layers = nn.layers;

    for(uint64_t i = 0; i < nn.layers; i++) {
        nuenet_mat_shape_t shape = nuenet_mat_get_shape(nn.weight[i]);

        fgrad.weight[i] = nuenet_mat_construct(shape.row, shape.col, NULL);
        fgrad.bias[i]   = nuenet_mat_construct(1, nn.bias[i].shape.col, NULL);
    }

    return fgrad;
}

/* Frees/Releases the finite gradient structure. */
void nuenet_nn_fgrad_destruct(nuenet_nn_fgrad_t fgrad) {
    for(uint64_t i = 0; i < fgrad.layers; i++) {
        nuenet_mat_destruct(fgrad.weight[i]);
        nuenet_mat_destruct(fgrad.bias[i]);
    }
    
    NUENET_FREE(fgrad.weight);
    NUENET_FREE(fgrad.bias);
}

/* Returns the gradient of current neural net using finite difference. */
nuenet_nn_fgrad_t nuenet_nn_fgrad_get(nuenet_nn_t nn, nuenet_nn_fgrad_t fgrad, 
    float eps, nuenet_mat_t ins, nuenet_mat_t outs) 
{
    float saved;

    /* The initial cost function we are going to compare with. */
    float cost = nuenet_nn_cost(nn, ins, outs);

    for(uint64_t i = 0; i < nn.layers; i++) {
        nuenet_mat_t w = nn.weight[i];
        nuenet_mat_t b = nn.bias[i];

        nuenet_mat_shape_t shape = nuenet_mat_get_shape(w);

        /* First find the difference for the weights. */
        for(uint64_t row = 0; row < shape.row; row++) {
            for(uint64_t col = 0; col < shape.col; col++) {
                /* Save the initial value and change the original 
                   value by 'eps'. */
                saved = _MAT_GET(w, row, col);
                _MAT_SET(w, row, col, saved + eps);

                /* Get the new cost and find the difference of the gradient. */
                float new_cost = nuenet_nn_cost(nn, ins, outs);

                /* Store the value to the finite gradient structure. */
                _MAT_SET(fgrad.weight[i], row, col, 
                        (new_cost - cost) / eps);

                /* Now restore the original value. */
                _MAT_SET(w, row, col, saved);
            }
        }

        /* Now the exactly same stuff with the biases. */
        /* By design, all the biases going to have a single row. */
        for(uint64_t col = 0; col < b.shape.col; col++) {
            /* Save the initial value and change the original 
               value by 'eps'. */
            saved = _MAT_GET(b, 0, col);
            _MAT_SET(b, 0, col, saved + eps);

            /* Get the new cost and find the difference of the gradient. */
            float new_cost = nuenet_nn_cost(nn, ins, outs);

            /* Store the value to the finite gradient structure. */
            _MAT_SET(fgrad.bias[i], 0, col, (new_cost - cost) / eps);

            /* Now restore the original value. */
            _MAT_SET(b, 0, col, saved);
        }
    }

    return fgrad;
}

/* Applies the finite gradient. */
nuenet_nn_t nuenet_nn_fgrad_apply(nuenet_nn_t nn, nuenet_nn_fgrad_t fgrad) {
    for(uint64_t i = 0; i < nn.layers; i++) {
        nuenet_mat_t w = nn.weight[i];
        nuenet_mat_t b = nn.bias[i];

        nuenet_mat_shape_t shape = nuenet_mat_get_shape(w);

        /* First apply for all the weights. */
        for(uint64_t row = 0; row < shape.row; row++) {
            for(uint64_t col = 0; col < shape.col; col++) {
                _MAT_SET(w, row, col, _MAT_GET(w, row, col) - 
                        _MAT_GET(fgrad.weight[i], row, col));
            }
        }

        /* Now the biases. */
        for(uint64_t col = 0; col < b.shape.col; col++) {
            _MAT_SET(b, 0, col, _MAT_GET(b, 0, col) - 
                    _MAT_GET(fgrad.bias[i], 0, col));
        }
    }

    return nn;
}

/* ---------------- NEURAL NETWORKS END ---------------- */

/* ---------------- ACTIVATION FUNCTIONS ---------------- */

/* The sigmoid function. */
float nuenet_sigmf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/* Apply sigmoid to the whole matrix. */
nuenet_mat_t nuenet_mat_sigmf(nuenet_mat_t mat) {
    _MAT_ITER(mat, row) {
        _MAT_ITER(mat, col) 
            _MAT_SET(mat, row, col, nuenet_sigmf(_MAT_GET(mat, row, col)));
    }

    return mat;
}

/* ---------------- ACTIVATION FUNCTIONS END ---------------- */

#endif // NUENET_IMPLEMENTATION
