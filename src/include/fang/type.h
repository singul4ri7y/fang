#ifndef FANG_TYPE_H
#define FANG_TYPE_H

#include <stdint.h>
#include <compiler.h>

/* ================ HELPER MACROS ================ */

/* Helper macros for conversion between generic and general data types. */
#define FANG_F2G(value)        fang_float_to_generic(value)
#define FANG_I2G(value)        fang_int_to_generic(value)
#define FANG_U2G(value)        fang_uint_to_generic(value)
#define FANG_G2F(value)        fang_generic_to_float(value)
#define FANG_G2I(value)        fang_generic_to_int(value)
#define FANG_G2U(value)        fang_generic_to_uint(value)

/* ================ HELPER MACROS END ================ */


/* ================ TYPE DEFINITIONS ================ */

/* NOTE: If tensor type is any of the floating types, any input data would be
 * considered `fang_float`.
 *
 * If tensor type is any of the signed/unsigned integer types, the input data
 * would be cosidered `fang_(u)int`, depending on the signedness.
 *
 * This is applicable to all external interaction with tensors, e.g. creating
 * an input tensor with external data.
 */

/* Most precised floating type Fang supports. */
typedef double fang_float;

/* Largest integer type Fang supports. */
typedef int64_t fang_int;
typedef uint64_t fang_uint;

/* Pointer has the size of a single general purpose register of a machine, thus
   being the largest data type machine can handle. */
/* Any form of singular data can be represented through it, e.g. 32-bit
   float or pointer to a structure. Hence, pointer values can be thought
   of as an abstract generic data type. */
typedef void *fang_gen;

/* ================ TYPE DEFINITIONS END ================ */


/* ================ INLINE DEFINITIONS ================ */

/* Ugly; stay happy and live a long life. */
#define _FANG_TYPE_DEF(postfix, ityp, otyp)                               \
FANG_HOT static inline fang_##otyp fang_##postfix(fang_##ityp value) {    \
    fang_##otyp *res = (fang_##otyp *) &value;                            \
    return *res;                                                          \
}

/* Converts float to generic data. */
_FANG_TYPE_DEF(float_to_generic, float, gen)

/* Converts generic data to float. */
_FANG_TYPE_DEF(generic_to_float, gen, float)

/* Converts signed integer to generic data. */
_FANG_TYPE_DEF(int_to_generic, int, gen)

/* Converts generic data to signed integer. */
_FANG_TYPE_DEF(generic_to_int, gen, int)

/* Converts unsigned integer to generic data. */
_FANG_TYPE_DEF(uint_to_generic, uint, gen)

/* Converts generic data to unsigned integer. */
_FANG_TYPE_DEF(generic_to_uint, gen, uint)

#undef _FANG_TYPE_DEF

/* ================ INLINE DEFINITIONS END ================ */

#endif  // FANG_TYPE_H
