#ifndef FANG_CONFIG_H
#define FANG_CONFIG_H

/* ================ CONFIGURATION MACROS ================ */

/* Default memory address alignment used in Fang. */
/* This alignment should be maintained regardless of reallocator function.
 * Cache lines are generally 64 bytes and fetched from address aligned in 64
 * bytes boundrary. Also, Fang uses AVX512 SIMD which requires 64 byte aligned
 * memory. */
/* Should be 2^n. */
#define FANG_MEMALIGN    64

/* Maximum Environments can be used. */
#define FANG_MAX_ENV     128

/* ================ CONFIGURATION MACROS END ================ */


/* ================ API MACROS ================ */

#ifdef FANG_LIB
#ifdef _WIN32
#define FANG_API    __declspec(dllexport)
#else
#define FANG_API
#endif  // _WIN32

#else  // FANG_LIB
#ifdef _WIN32
#define FANG_API    __declspec(dllimport)
#else
#define FANG_API    extern
#endif  // _WIN32
#endif  // FANG_LIB

/* ================ API MACROS END ================ */

#endif  // FANG_CONFIG_H
