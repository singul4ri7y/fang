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
    uint64_t stride;
    float *data;
} nuenet_mat_t;

/* Which one to transpose? */
/* Used in functions like 'nuenet_mat_dot_transp()'. */
typedef enum nuenet_mat_transp_which {
    NUENET_MAT_TRANSPOSE_A,
    NUENET_MAT_TRANSPOSE_B
} nuenet_mat_transp_which_t;

/* Consturcts and returns a new matrix. */
/* Pass a NULL pointer to the data to create a new matrix in the heap. */
nuenet_mat_t       nuenet_mat_construct(uint64_t rows, uint64_t cols, void *data);

/* Creates a new matrix structure. */
/* Note: It does not do any sort of allocation, it just fills in the 
 *       'nuenet_mat_t' structure.
 */
nuenet_mat_t       nuenet_mat_new(uint64_t rows, uint64_t cols, 
        uint64_t stride, void *data);

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
float              nuenet_mat_set(nuenet_mat_t mat, uint64_t row, uint64_t col, float value);

/* Initializes a matrix with randomized values. */
void               nuenet_mat_randf(nuenet_mat_t mat, float low, float high);

/* Fill the matrix with the provided value. */
nuenet_mat_t       nuenet_mat_fill(nuenet_mat_t mat, float value);

/* Release if the matrix is created in the heap. */
void               nuenet_mat_destruct(nuenet_mat_t mat);

/* Gets the dimension of a matrix. */
nuenet_mat_shape_t nuenet_mat_get_shape(nuenet_mat_t mat);

/* Dot product of two matrices. Destination matrix is expected to be allocated
   with proper shape. */
/* Returns the dest matrix. */
nuenet_mat_t       nuenet_mat_dot(nuenet_mat_t dest, 
        nuenet_mat_t a, nuenet_mat_t b);

/* Element-wise matrix multiplication (Hadamard Product). */
/* Returns the resulant matrix. */
nuenet_mat_t       nuenet_mat_hadamard(nuenet_mat_t dest,
        nuenet_mat_t a, nuenet_mat_t b);

/* Dot product of two matrices, but one of them being transposed. */
/* Returns the resultant matrix. */
nuenet_mat_t       nuenet_mat_dot_transp(nuenet_mat_t dest,
        nuenet_mat_t a, nuenet_mat_t b, nuenet_mat_transp_which_t which);

/* Multiplies a scalar value to each of the element in the matrix. 
   it basically scales the matrix/vector. */
/* Returns the resultant matrix. */
nuenet_mat_t       nuenet_mat_scale(nuenet_mat_t dest, nuenet_mat_t a,
        float scalar);

/* Summation of two matrices. */
/* Saves the summation to matrix 'dest' and returns the matrix. */
/* Returns the resultant matrix. */
nuenet_mat_t       nuenet_mat_sum(nuenet_mat_t dest, nuenet_mat_t a, 
        nuenet_mat_t b);

/* Subtraction of two matrices. */
/* Saves the resultant matrix to destination matrix and returns the matrix. */
/* Returns the resultant matrix. */
nuenet_mat_t       nuenet_mat_diff(nuenet_mat_t dest, nuenet_mat_t a, 
        nuenet_mat_t b);

/* Transposes the provided matrix. */
/* The destination matrix is expected to be in proper transposed shape. */
/* Returns the transposed matrix. */
nuenet_mat_t       nuenet_mat_transp(nuenet_mat_t dest, nuenet_mat_t mat);

/* Returns true of provided matrices conform (same shape) each other. */
bool               nuenet_mat_conf(nuenet_mat_t a, nuenet_mat_t b);

/* Returns true if provided matrices are equal. */
bool               nuenet_mat_equal(nuenet_mat_t a, nuenet_mat_t b);

/* Prints a matrix. */
void               nuenet_mat_print(nuenet_mat_t matrix, 
        const char *const name, int padding);

/* ---------------- MATRIX END ---------------- */

/* ---------------- NEURAL NETWORKS ---------------- */

#define NUENET_NN_PRINT(nn)        nuenet_nn_print(nn, #nn)
#define NUENET_NN_GRAD_PRINT(grad) nuenet_nn_grad_print(grad, #grad)

typedef struct nuenet_nn {
    /* The weight and bias matrices. */
    nuenet_mat_t *weight;
    nuenet_mat_t *bias;

    /* These matrices store the intermediate values of
       operation. */
    /* This matrix also may hold the value of output. */
    nuenet_mat_t *act;

    /* Number of layers, inputs and outputs. */
    uint64_t layers;
    uint64_t ins;
    uint64_t outs;
} nuenet_nn_t;

/* Stores the gradient. */
typedef struct nuenet_nn_grad {
    /* We will store gradient only for the weights and biases. */
    nuenet_mat_t *weight;

    /* Which may also store the error term calculations. */
    nuenet_mat_t *bias;

    /* To store the intermediate calculations during backpropagation. */
    nuenet_mat_t *intm_weight;
    nuenet_mat_t *intm_bias;

    /* The amount of layers the NN has. */
    uint64_t layers;
} nuenet_nn_grad_t;

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

/* Creates and returns a new gradient structure for a neural net. */
nuenet_nn_grad_t  nuenet_nn_grad_construct(nuenet_nn_t nn);

/* Frees/Releases the gradient structure. */
void              nuenet_nn_grad_destruct(nuenet_nn_grad_t grad);

/* Initializes/zeros out the gradient structure. */
void              nuenet_nn_grad_init(nuenet_nn_grad_t grad);

/* Prints the gradient structure. */
void              nuenet_nn_grad_print(nuenet_nn_grad_t grad, 
        const char *const name);

/* Calculate gradients using Backward Propagation on the Neural Network. */
void              nuenet_nn_back_prop(nuenet_nn_t nn, nuenet_nn_grad_t grad,
        nuenet_mat_t ins, nuenet_mat_t outs);

/* Applies calculated gradients using Batch Gradient Descent. */
void              nuenet_nn_apply_batch_gd(nuenet_nn_t nn, 
        nuenet_nn_grad_t grad, float rate);

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
    nuenet_mat_t mat = nuenet_mat_new(rows, cols, cols, NULL);

    /* If we exclude the last high bit, it's still roughly 
     * 8 Exabytes of addressable memory, which is, well... 
     * still a lot.
     */
    if(data == NULL) {
        void* buffer = NUENET_ALLOC(*mat.data, rows * cols);

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
nuenet_mat_t nuenet_mat_new(uint64_t rows, uint64_t cols, 
    uint64_t stride,  void *data) 
{
    nuenet_mat_t mat = {
        .shape  = { .row = rows, .col = cols },
        .stride = stride,
        .data   = data
    };

    return mat;
}

/* Returns a sub-matrix of the given matrix. */
/* Note: Do not call 'nuenet_mat_destruct()' on sub-matrices. Sub matrices
 *       share data memory of the parent matrix.
 */
nuenet_mat_t nuenet_mat_sub(nuenet_mat_t mat, uint64_t rows, uint64_t cols, 
    uint64_t start_row, uint64_t start_col) 
{
    nuenet_mat_shape_t shape = nuenet_mat_get_shape(mat);

    NUENET_ASSERT(start_row + rows <= shape.row);
    NUENET_ASSERT(start_col + cols <= shape.col);

    return nuenet_mat_new(rows, cols, mat.stride, mat.data + 
        start_row * mat.stride + start_col);
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
    nuenet_mat_shape_t shape = nuenet_mat_get_shape(mat);

    NUENET_ASSERT(row < shape.row);
    NUENET_ASSERT(col < shape.col);

    return nuenet__mat_get_data(mat)[row * mat.stride + col];
}

/* Sets a value to a matrix. 0 based indexing. */
float nuenet_mat_set(nuenet_mat_t mat, uint64_t row, uint64_t col, float value) {
    NUENET_ASSERT(row < mat.shape.row);
    NUENET_ASSERT(col < mat.shape.col);

    nuenet__mat_get_data(mat)[row * mat.stride + col] = value;

    return value;
}

/* Initializes a matrix with randomized values. */
void nuenet_mat_randf(nuenet_mat_t mat, float high, float low) {
    _MAT_ITER(mat, row) {
        _MAT_ITER(mat, col) 
            _MAT_SET(mat, row, col, nuenet_randf() * (high - low) + low);
    }
}

/* Fill the matrix with the provided value. */
nuenet_mat_t nuenet_mat_fill(nuenet_mat_t mat, float value) {
    _MAT_ITER(mat, row) {
        _MAT_ITER(mat, col) 
            _MAT_SET(mat, row, col, value);
    }

    return mat;
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

/* Element-wise matrix multiplication (Hadamard Product). */
/* Returns the resulant matrix. */
nuenet_mat_t nuenet_mat_hadamard(nuenet_mat_t dest, nuenet_mat_t a, 
    nuenet_mat_t b) 
{
    NUENET_ASSERT(a.shape.row    == b.shape.row);
    NUENET_ASSERT(a.shape.col    == b.shape.col);
    NUENET_ASSERT(dest.shape.row == a.shape.row);
    NUENET_ASSERT(dest.shape.col == a.shape.col);

    _MAT_ITER(a, row) {
        _MAT_ITER(a, col) {
            _MAT_SET(dest, row, col, 
                _MAT_GET(a, row, col) * _MAT_GET(b, row, col));
        }
    }

    return dest;
}

/* Dot product of two matrices, but one of them being transposed. */
/* Returns the resultant matrix. */
nuenet_mat_t nuenet_mat_dot_transp(nuenet_mat_t dest,
    nuenet_mat_t a, nuenet_mat_t b, nuenet_mat_transp_which_t which) 
{
    NUENET_ASSERT(which == NUENET_MAT_TRANSPOSE_A || 
        which == NUENET_MAT_TRANSPOSE_B);

    uint64_t a_row = a.shape.row;
    uint64_t a_col = a.shape.col;
    uint64_t b_row = b.shape.row;
    uint64_t b_col = b.shape.col;

    if(which == NUENET_MAT_TRANSPOSE_A) 
        a_row = a.shape.col, a_col = a.shape.row;
    else if(which == NUENET_MAT_TRANSPOSE_B) 
        b_row = b.shape.col, b_col = b.shape.row;

    NUENET_ASSERT(a_col == b_row);
    NUENET_ASSERT(a_row == dest.shape.row);
    NUENET_ASSERT(b_col == dest.shape.col);

    for(uint64_t row = 0; row < a_row; row++) {
        for(uint64_t col = 0; col < b_col; col++) {
            float result = 0.0f;

            for(uint64_t k = 0; k < a_col; k++) {
                result += (which == NUENET_MAT_TRANSPOSE_A) 
                    ? _MAT_GET(a, k, row) * _MAT_GET(b, k, col)
                    : _MAT_GET(a, row, k) *_MAT_GET(b, col, k);
            }

            _MAT_SET(dest, row, col, result);
        }
    }

    return dest;
}

/* Multiplies a scalar value to each of the element in the matrix. 
   it basically scales the matrix/vector. */
/* Returns the resultant matrix. */
nuenet_mat_t nuenet_mat_scale(nuenet_mat_t dest, nuenet_mat_t a, float scalar) {
    NUENET_ASSERT(a.shape.row == dest.shape.row);
    NUENET_ASSERT(a.shape.col == dest.shape.col);

    _MAT_ITER(a, row) {
        _MAT_ITER(a, col) {
            _MAT_SET(dest, row, col, scalar * _MAT_GET(a, row, col));
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

/* Subtraction of two matrices. */
/* Saves the resultant matrix to matrix 'dest' and returns the matrix. */
nuenet_mat_t nuenet_mat_diff(nuenet_mat_t dest, nuenet_mat_t a, 
        nuenet_mat_t b) 
{
    NUENET_ASSERT(a.shape.row    == b.shape.row);
    NUENET_ASSERT(a.shape.col    == b.shape.col);
    NUENET_ASSERT(dest.shape.row == a.shape.row);
    NUENET_ASSERT(dest.shape.col == a.shape.col);

    _MAT_ITER(dest, row) {
        _MAT_ITER(dest, col) {
            _MAT_SET(dest, row, col, 
                _MAT_GET(a, row, col) - _MAT_GET(b, row, col));
        }
    }

    return dest;
}

/* Transposes the provided matrix. */
/* The destination matrix is expected to be in proper transposed shape. */
/* Returns the transposed matrix. */
nuenet_mat_t nuenet_mat_transp(nuenet_mat_t dest, nuenet_mat_t mat) {
    NUENET_ASSERT(mat.shape.row == dest.shape.col);
    NUENET_ASSERT(mat.shape.col == dest.shape.row);

    _MAT_ITER(mat, row) {
        _MAT_ITER(mat, col) {
            _MAT_SET(dest, col, row, _MAT_GET(mat, row, col));
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
    NUENET_ASSERT(len >= 2);

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

    nn.act = NUENET_ALLOC(nuenet_mat_t, len);
    NUENET_ASSERT(nn.act != NULL);

    for(uint64_t i = 0; i < len; i++) {
        nn.weight[i] = nuenet_mat_construct(blueprint[i - 1], blueprint[i], NULL);
        nn.bias[i]   = nuenet_mat_construct(1, blueprint[i], NULL);
        nn.act[i]    = nuenet_mat_construct(1, blueprint[i], NULL);
    }

    return nn;
}

/* Frees an allocated neural network. */
void nuenet_nn_destruct(nuenet_nn_t nn) {
    for(uint64_t i = 0; i < nn.layers; i++) {
        nuenet_mat_destruct(nn.weight[i]);
        nuenet_mat_destruct(nn.bias[i]);
        nuenet_mat_destruct(nn.act[i]);
    }

    NUENET_FREE(nn.weight);
    NUENET_FREE(nn.bias);
    NUENET_FREE(nn.act);
}

/* Fills the the weights and biases with random values. */
nuenet_nn_t nuenet_nn_randf(nuenet_nn_t nn, uint64_t lo, uint64_t hi) {
    for(uint64_t i = 0; i < nn.layers; i++) {
        nuenet_mat_randf(nn.weight[i], lo, hi);
        nuenet_mat_randf(nn.bias[i], lo, hi);
        nuenet_mat_randf(nn.act[i], lo, hi);
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
        snprintf(buf, 128 * sizeof(char), "nn.act[%lu]", i);
        nuenet_mat_print(nn.act[i], buf, 4);
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
        nuenet_mat_dot(nn.act[i], acc, nn.weight[i]);
        nuenet_mat_sum(nn.act[i], nn.act[i], nn.bias[i]);
        nuenet_mat_sigmf(nn.act[i]);

        acc = nn.act[i];
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
            float diff = _MAT_GET(fp_out, 0, j) - _MAT_GET(outs, i, j);

            result += diff * diff;
        }
    }

    return result / (shape.row * shape.col);
}

/* Creates and returns a new gradient structure for a neural net. */
nuenet_nn_grad_t nuenet_nn_grad_construct(nuenet_nn_t nn) {
    nuenet_nn_grad_t grad;

    grad.weight = NUENET_ALLOC(nuenet_mat_t, nn.layers);
    NUENET_ASSERT(grad.weight != NULL);

    grad.intm_weight = NUENET_ALLOC(nuenet_mat_t, nn.layers);
    NUENET_ASSERT(grad.intm_weight != NULL);

    grad.bias = NUENET_ALLOC(nuenet_mat_t, nn.layers);
    NUENET_ASSERT(grad.bias != NULL);

    grad.intm_bias = NUENET_ALLOC(nuenet_mat_t, nn.layers);
    NUENET_ASSERT(grad.intm_bias != NULL);

    /* We need to know the amount of layers we have, useful when freeing. */
    grad.layers = nn.layers;

    for(uint64_t l = 0; l < grad.layers; l++) {
        nuenet_mat_shape_t shape = nuenet_mat_get_shape(nn.weight[l]);

        grad.weight[l]      = nuenet_mat_construct(shape.row, shape.col, NULL);
        grad.intm_weight[l] = nuenet_mat_construct(shape.row, shape.col, NULL);
        grad.bias[l]        = nuenet_mat_construct(1, shape.col, NULL);
        grad.intm_bias[l]   = nuenet_mat_construct(1, shape.col, NULL);
    }

    return grad;
}

/* Frees/Releases the gradient structure. */
void nuenet_nn_grad_destruct(nuenet_nn_grad_t grad) {
    for(uint64_t l = 0; l < grad.layers; l++) {
        nuenet_mat_destruct(grad.weight[l]);
        nuenet_mat_destruct(grad.intm_weight[l]);
        nuenet_mat_destruct(grad.bias[l]);
        nuenet_mat_destruct(grad.intm_bias[l]);
    }
    
    NUENET_FREE(grad.weight);
    NUENET_FREE(grad.intm_weight);
    NUENET_FREE(grad.bias);
    NUENET_FREE(grad.intm_bias);
}

/* Initializes/zeros out the gradient structure. */
void nuenet_nn_grad_init(nuenet_nn_grad_t grad) {
    for(uint64_t l = 0; l < grad.layers; l++) {
        nuenet_mat_fill(grad.weight[l], 0.0f);
        nuenet_mat_fill(grad.bias[l], 0.0f);
    }
}

/* Prints the gradient structure. */
void nuenet_nn_grad_print(nuenet_nn_grad_t grad, const char *const grad_name) {
    char buf[128];

    printf("%s = [\n", grad_name);
    for(uint64_t i = 0; i < grad.layers; i++) {
        snprintf(buf, 128 * sizeof(char), "grad.weight[%lu]", i);
        nuenet_mat_print(grad.weight[i], buf, 4);
        snprintf(buf, 128 * sizeof(char), "grad.bias[%lu]", i);
        nuenet_mat_print(grad.bias[i], buf, 4);
    }
    printf("]\n");
}

/* Calculate gradients using Backward Propagation on the Neural Network. */
void nuenet_nn_back_prop(nuenet_nn_t nn, nuenet_nn_grad_t grad,
    nuenet_mat_t ins, nuenet_mat_t outs) 
{
    NUENET_ASSERT(ins.shape.row == outs.shape.row);

    /* Initialize the graident structure. */
    nuenet_nn_grad_init(grad);

    /* For each of the sample. */
    for(uint64_t i = 0; i < ins.shape.row; i++) {
        nuenet_mat_t input  = nuenet_mat_row(ins, i);
        nuenet_mat_t output = nuenet_mat_row(outs, i);
        nuenet_mat_t fp_out = nuenet_nn_ford_prop(nn, input);

        /* First we need to process the output layers. */
        uint64_t ol_idx = grad.layers - 1;

        /* Calculate gradient for the activation fn. */
        nuenet_mat_hadamard(grad.intm_bias[ol_idx], 
            nuenet_mat_diff(grad.intm_bias[ol_idx], 
                nuenet_mat_fill(grad.intm_bias[ol_idx], 1.0f),
                /* Output layer activation are the output of the forward propagation. */
                fp_out),
            fp_out);

        /* Now calculate the gradient w.r.t. bias for current sample. */
        nuenet_mat_hadamard(grad.intm_bias[ol_idx], 
            /* We do not need 'fp_out' anymore. */
            nuenet_mat_diff(fp_out, fp_out, output),
            grad.intm_bias[ol_idx]);

        /* Add current bias gradients for current sample. */
        nuenet_mat_sum(grad.bias[ol_idx], grad.bias[ol_idx], 
            grad.intm_bias[ol_idx]);

        /* Calculate the graidents w.r.t. weights. */
        nuenet_mat_dot_transp(grad.intm_weight[ol_idx], 
            (ol_idx == 0) ? input : nn.act[ol_idx - 1], 
            /* The gradient of the biases are also our error terms. */
            grad.intm_bias[ol_idx], NUENET_MAT_TRANSPOSE_A);

        /* Now add the weight gradients for the current sample. */
        nuenet_mat_sum(grad.weight[ol_idx], grad.weight[ol_idx],
            grad.intm_weight[ol_idx]);

        /* Now calculate the gradients w.r.t. biases and weights for
           the hidden layers. */
        for(int64_t l = (int64_t) grad.layers - 2; l >= 0; l--) {
            /* We basically store the error term as gradient of bias. */
            nuenet_mat_dot_transp(grad.intm_bias[l], grad.intm_bias[l + 1], 
                nn.weight[l + 1], NUENET_MAT_TRANSPOSE_B);

            /* As the column number of the bias and weights matrices are 
               the same and on top of that we don't need the intermediate
               weights now, we can use it's first row as the another 
               intermediate bias matrix. */
            nuenet_mat_t intm_weight_row = nuenet_mat_row(grad.intm_weight[l], 0);

            nuenet_mat_hadamard(grad.intm_bias[l], grad.intm_bias[l],
                nuenet_mat_hadamard(intm_weight_row, 
                    nuenet_mat_diff(intm_weight_row,
                        nuenet_mat_fill(intm_weight_row, 1.0f),
                        nn.act[l]),
                    nn.act[l]));

            /* Now add the gradient w.r.t. bias sample. */
            nuenet_mat_sum(grad.bias[l], grad.bias[l], grad.intm_bias[l]);

            /* Calculate the graidents w.r.t. weights. */
            nuenet_mat_dot_transp(grad.intm_weight[l], 
                (l == 0) ? input : nn.act[l - 1], 
                /* The gradient of the biases are also our error terms. */
                grad.intm_bias[l], NUENET_MAT_TRANSPOSE_A);

            /* Now add the weight gradients for the current sample. */
            nuenet_mat_sum(grad.weight[l], grad.weight[l],
                grad.intm_weight[l]);
        }
    }
}

/* Applies calculated gradients using Batch Gradient Descent. */
void nuenet_nn_apply_batch_gd(nuenet_nn_t nn, 
    nuenet_nn_grad_t grad, float rate) 
{
    for(uint64_t l = 0; l < nn.layers; l++) {
        /* First the weights. */
        nuenet_mat_diff(nn.weight[l], nn.weight[l], 
            nuenet_mat_scale(grad.intm_weight[l], 
                grad.weight[l], rate));

        /* Then the biases. */
        nuenet_mat_diff(nn.bias[l], nn.bias[l], 
            nuenet_mat_scale(grad.intm_bias[l],
                grad.bias[l], rate));
    }
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
