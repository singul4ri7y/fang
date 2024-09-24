#ifndef FANG_COMPILER_H
#define FANG_COMPILER_H

/* ================ COMPILER OPTIMIZATION MACROS ================ */

#if defined(__GNUC__) || defined(__clang__)
/* Branch predictions. */
#define FANG_LIKELY(x)      __builtin_expect(!!(x), 1)
#define FANG_UNLIKELY(x)    __builtin_expect(!!(x), 0)

/* For optimizing hot/cold code sections and routines (functions). */
#define FANG_HOT            __attribute__((hot))
#define FANG_COLD           __attribute__((cold))

/* Inline every function call, reducing overhead. */
#define FANG_FLATTEN        __attribute__((flatten))

/* To be used with reallocator functions. */
#define FANG_MALLOC         __attribute__((malloc))

#else  // __GNUC__ or __clang__
#define FANG_LIKELY(x)       x
#define FANG_UNLIKELY(x)    !x

#define FANG_HOT
#define FANG_COLD

#define FANG_FLATTEN

#define FANG_MALLOC
#endif // __GNUC__ or __clang__

/* ================ COMPILER OPTIMIZATION MACROS END ================ */

#endif  // FANG_COMPILER_H
