#include <stdio.h>
#include <fang/tensor.h>
#include <fang/env.h>
#include <fang/type.h>

int main(int argc, char const *argv[])
{
    // this creates environment for CPU. do not bother this is routione for now.
    // gpu support is coming soon.
    int env = fang_env_create(FANG_ENV_TYPE_CPU, NULL);

    /** Lets create some tensors, assign random numbers, print and release them */
    fang_ten_t ten;
    fang_ten_create(&ten, env, FANG_TEN_DTYPE_FLOAT32, FANG_DIM(3,3), NULL);
    fang_ten_rand(&ten, FANG_F2G(10), FANG_F2G(20), 69);
    FANG_TEN_PRINT(&ten);
    fang_ten_release(&ten);

    // scalar tensor
    fang_ten_scalar(&ten, env, FANG_TEN_DTYPE_FLOAT32, FANG_F2G(10));
    FANG_TEN_PRINT(&ten);
    fang_ten_release(&ten);

    /**What about some tensor operations */
    fang_ten_t x, y, answer;
    fang_ten_dim_t xdim = FANG_DIM(3,3);
    fang_ten_dim_t ydim = FANG_DIM(3,3);
    fang_ten_dim_t adim = FANG_DIM(3,3);
    fang_ten_create(&x, env, FANG_TEN_DTYPE_FLOAT32, xdim, NULL);
    fang_ten_create(&y, env, FANG_TEN_DTYPE_FLOAT32, ydim, NULL);
    fang_ten_create(&answer, env, FANG_TEN_DTYPE_FLOAT32, adim, NULL);

    fang_ten_rand(&x, FANG_F2G(10), FANG_F2G(20), 69);
    fang_ten_rand(&y, FANG_F2G(10), FANG_F2G(20), 96);
    FANG_TEN_PRINT(&x);
    FANG_TEN_PRINT(&y);

    // sum of two tensors
    fang_ten_sum(&answer, &x, &y);
    FANG_TEN_PRINT(&answer);

    // difference of two tensors
    fang_ten_diff(&answer, &x, &y);
    FANG_TEN_PRINT(&answer);

    // element wise multiplication of two tensors
    fang_ten_mul(&answer, &x, &y);
    FANG_TEN_PRINT(&answer);
    fang_ten_release(&x);
    fang_ten_release(&y);
    fang_ten_release(&answer);

    /** Now is the time for Matix Multiplication
     * we are multiplying simple 2x3 and 3x2 matrices
     * the result should be 2x2 matrix
     */
    fang_ten_t matmul_result;
    fang_ten_create(&x, env, FANG_TEN_DTYPE_FLOAT32, FANG_DIM(2,3), NULL);
    fang_ten_create(&y, env, FANG_TEN_DTYPE_FLOAT32, FANG_DIM(3,2), NULL);
    fang_ten_create(&matmul_result, env, FANG_TEN_DTYPE_FLOAT32, FANG_DIM(2,2), NULL);

    fang_ten_rand(&x, FANG_F2G(10), FANG_F2G(20), 69);
    fang_ten_rand(&y, FANG_F2G(10), FANG_F2G(20), 96);
    FANG_TEN_PRINT(&x);
    FANG_TEN_PRINT(&y);

    fang_ten_matmul(&matmul_result, &x, &y);
    FANG_TEN_PRINT(&matmul_result);

    fang_ten_release(&x);
    fang_ten_release(&y);
    fang_ten_release(&matmul_result);

    /** GEMM(General Matrix Multiplication) using the same matmul function
     * Lets multiply two 2x3x4 and 4x3x2 matrices
     * the result should be 2x3x2 matrix
     */
    fang_ten_t gemm_result;
    fang_ten_create(&x, env, FANG_TEN_DTYPE_FLOAT32, FANG_DIM(2,3,4), NULL);
    fang_ten_create(&y, env, FANG_TEN_DTYPE_FLOAT32, FANG_DIM(2,4,5), NULL);
    fang_ten_create(&gemm_result, env, FANG_TEN_DTYPE_FLOAT32, FANG_DIM(2,3,5), NULL);

    fang_ten_rand(&x, FANG_F2G(10), FANG_F2G(20), 69);
    fang_ten_rand(&y, FANG_F2G(10), FANG_F2G(20), 96);
    FANG_TEN_PRINT(&x);
    FANG_TEN_PRINT(&y);
    
    /* Performs matrix-multiplication between two tensors. It's just a wrapper
    * around `fang_ten_gemm()`, which is much more genralized. */
    fang_ten_matmul(&gemm_result, &x, &y);
    FANG_TEN_PRINT(&gemm_result);

    fang_ten_release(&x);
    fang_ten_release(&y);
    fang_ten_release(&gemm_result);


    /** Example of broadcasting */
    fang_ten_create(&x, env, FANG_TEN_DTYPE_FLOAT32, FANG_DIM(2,3,4), NULL);
    fang_ten_create(&y, env, FANG_TEN_DTYPE_FLOAT32, FANG_DIM(3,4), NULL);
    fang_ten_create(&answer, env, FANG_TEN_DTYPE_FLOAT32, FANG_DIM(2,3,4), NULL);

    fang_ten_rand(&x, FANG_F2G(10), FANG_F2G(20), 69);
    fang_ten_rand(&y, FANG_F2G(10), FANG_F2G(20), 96);
    FANG_TEN_PRINT(&x);
    FANG_TEN_PRINT(&y);

    fang_ten_sum(&answer, &x, &y);
    FANG_TEN_PRINT(&answer);

    fang_ten_release(&x);
    fang_ten_release(&y);
    fang_ten_release(&answer);

    return 0;
}