#ifndef FANG_COMPILER_H
#define FANG_COMPILER_H

/* ================ COMPILER OPTIMIZATION MACROS ================ */

#if defined(__GNUC__) || defined(__clang__)
/* Branch predictions. */
#define FANG_LIKELY(x)                 __builtin_expect(!!(x), 1)
#define FANG_UNLIKELY(x)               __builtin_expect(!!(x), 0)

/* Tell compiler to assume some memory is N-byte aligned. */
#define FANG_ASSUME_ALIGNED(m, b)      __builtin_assume_aligned((void *) m, b)

/* N-byte boundary aligned stack-memory allocation. */
#define FANG_ALIGNAS(b)                __attribute__((aligned(b)))

/* Prefetch memory to cache. */
#define FANG_PREFETCH(ptr, rw, loc)    __builtin_prefetch(ptr, rw, loc)

/* Whether prefetch to read or write. */
#define FANG_PREFETCH_READ             0
#define FANG_PREFETCH_WRITE            1
#define FANG_PREFETCH_SHARED_READ      2

/* Prefetch temporal locality. */
#define FANG_PREFETCH_LOCALITY_D0      0  // No temporal locality
#define FANG_PREFETCH_LOCALITY_D1      1  // Minimal locality, likely L3 cache
#define FANG_PREFETCH_LOCALITY_D2      2  // Moderate locality, likely L2 cache
#define FANG_PREFETCH_LOCALITY_D3      3  // Maximum locality, likely L1 cache

/* For optimizing hot/cold code sections and routines (functions). */
#define FANG_HOT                       __attribute__((hot))
#define FANG_COLD                      __attribute__((cold))

/* Inline every function call, reducing overhead. */
#define FANG_FLATTEN                   __attribute__((flatten))

/* To be used with reallocator functions. */
#define FANG_MALLOC                    __attribute__((malloc))

/* Suppress unused parameter. */
#define FANG_UNUSED                    __attribute__((unused))

/* Always inline a function. */
#define FANG_INLINE                    __attribute__((always_inline))

#else  // __GNUC__ or __clang__
#define FANG_LIKELY(x)                 !!(x)
#define FANG_UNLIKELY(x)               !!(x)

#define FANG_ASSUME_ALIGNED(m, b)      (m)

#define FANG_ALIGNAS(b)

#define FANG_PREFETCH(ptr, rw, loc)

#define FANG_PREFETCH_READ
#define FANG_PREFETCH_WRITE
#define FANG_PREFETCH_SHARED_READ

#define FANG_PREFETCH_LOCALITY_D0
#define FANG_PREFETCH_LOCALITY_D1
#define FANG_PREFETCH_LOCALITY_D2
#define FANG_PREFETCH_LOCALITY_D3

#define FANG_HOT
#define FANG_COLD

#define FANG_FLATTEN

#define FANG_MALLOC

#define FANG_UNUSED

#define FANG_INLINE
#endif // __GNUC__ or __clang__

/* ================ COMPILER OPTIMIZATION MACROS END ================ */

#endif  // FANG_COMPILER_H
