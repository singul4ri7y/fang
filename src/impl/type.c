#include <fang/type.h> 

/* ---------------- DEFINITIONS ---------------- */ 

/* Converts floating value to generic void pointer. */
fang_gen fang_float_to_generic(fang_float value) {
    fang_gen *res = (fang_gen *) &value;
    return *res;
}

/* Converts generic void pointer to floating value. */
fang_float fang_generic_to_float(fang_gen value) {
    fang_float *res = (fang_float *) &value;
    return *res;
}

/* Converts signed integer value to generic void pointer. */
fang_gen fang_int_to_generic(fang_int value) {
    fang_gen *res = (fang_gen *) &value;
    return *res;
}

/* Converts generic void pointer to signed integer value. */
fang_int fang_generic_to_int(fang_gen value) {
    fang_int *res = (fang_int *) &value;
    return *res;
}

/* Converts unsigned integer value to generic void pointer. */
fang_gen fang_uint_to_generic(fang_uint value) {
    fang_gen *res = (fang_gen *) &value;
    return *res;
}

/* Converts generic void pointer to unsigned integer value. */
fang_uint fang_generic_to_uint(fang_gen value) {
    fang_uint *res = (fang_uint *) &value;
    return *res;
}

/* ---------------- DEFINITIONS END ---------------- */ 
