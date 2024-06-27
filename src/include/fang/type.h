#ifndef FANG_TYPE_H
#define FANG_TYPE_H

#include <stdint.h>

/* ---------------- TYPE DEFINITIONS ---------------- */

/*
 * Note: If tensor type is any of the floating type, 
 * any input data will be considered of type 'fang_float'.
 *
 * If tensor type is any of the signed/unsigned integer
 * type, the input data will be considered 'fang_(u)int'
 * depending on the signedness.
 *
 * This is applicable to all the external interaction with 
 * tensors, e.g. creating an input tensor with external data.
 */

/* Most precised floating type Fang supports. */
typedef double fang_float;

/* Largest integer type Fang supports. */
typedef int64_t fang_int;
typedef uint64_t fang_uint;

/* Pointer is the largest singular data-type a machine can handle. */
/* You can pass anything to it, starting from pointer of any structure to 
   any type of bitcasted value, e.g. 32-bit float. */
typedef void *fang_gen;

/* ---------------- TYPE DEFINITIONS END ---------------- */

/* ---------------- HELPER MACROS ---------------- */

/* Helper macro for coversion between general datatypes to void pointer 
   which is internally usable and representable. */
#define FANG_F2G(value)        fang_float_to_generic(value)
#define FANG_I2G(value)        fang_int_to_generic(value)
#define FANG_U2G(value)        fang_uint_to_generic(value)
#define FANG_G2F(value)        fang_generic_to_float(value)
#define FANG_G2I(value)        fang_generic_to_int(value)
#define FANG_G2U(value)        fang_generic_to_uint(value)

#define FANG_TEN_PRINT(ten)    fang_ten_print(ten, #ten, 0)

/* ---------------- HELPER MACROS END ---------------- */

/* ---------------- DECLARATIONS---------------- */

/* Converts floating value to generic void pointer. */
fang_gen   fang_float_to_generic(fang_float value);

/* Converts generic void pointer to floating value. */
fang_float fang_generic_to_float(fang_gen value);

/* Converts signed integer value to generic void pointer. */
fang_gen   fang_int_to_generic(fang_int value);

/* Converts generic void pointer to signed integer value. */
fang_int   fang_generic_to_int(fang_gen value);

/* Converts unsigned integer value to generic void pointer. */
fang_gen   fang_uint_to_generic(fang_uint value);

/* Converts generic void pointer to unsigned integer value. */
fang_uint  fang_generic_to_uint(fang_gen value);

/* ---------------- DECLARATIONS END ---------------- */

#endif // FANG_TYPE_H
