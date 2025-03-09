#include <stdio.h>
#include <fang/tensor.h>
#include <fang/env.h>
#include <fang/type.h>

/**
 * To compile this use or edit CMakelists 
 */

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

    /** If we want to create a tensor manually... */
    
    fang_int_t my_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9}; // Example data

    fang_ten_t tensor;
    int res = fang_ten_create(&tensor, env, FANG_TEN_DTYPE_INT32, FANG_DIM(2,2), my_data);
    FANG_TEN_PRINT(&tensor);
    fang_ten_release(&tensor);

    fang_float_t my_data_f[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    res = fang_ten_create(&tensor, env, FANG_TEN_DTYPE_FLOAT32, FANG_DIM(2,2), my_data_f);
    FANG_TEN_PRINT(&tensor);
    fang_ten_release(&tensor);

}