#ifndef FANG_BUFFER_H
#define FANG_BUFFER_H

#include <fang/config.h>
#include <compiler.h>
#include <memory.h>

/* ================ HELPER MACRO ================ */

/* Initial capacity. */
#define FANG_BUFFER_INIT_CAPACITY             64

/* Helper macro to create buffer of specific type. */
#define FANG_BUFFER_CREATE(buff, realloc, type)    fang_buffer_create(buff,    \
    realloc, sizeof(type));

/* Get an element from buffer. */
#define FANG_BUFFER_GET(buff, type, index)    (type *) fang_buffer_get(    \
    buff, index)

/* ================ HELPER MACRO END ================ */


/* ================ DATA STRUCTURES ================ */

/* Represent any type of buffer of any type. */
typedef struct fang_buffer {
    void *data;                  // The buffer
    fang_reallocator_t realloc;  // Reallocator used for this buffer
    size_t capacity;             // How many elements were allocated?
    size_t count;                // Element count in the buffer
    int n;                       // Size of a single element
} fang_buffer_t;

/* ================ DATA STRUCTURES END ================ */


/* ================ DECLARATIONS ================ */

/* Initializes buffer structure. */
FANG_API int fang_buffer_create(fang_buffer_t *restrict buff,
    fang_reallocator_t realloc, int n);

/* Pushes a single element to the buffer. */
FANG_API FANG_HOT int fang_buffer_add(fang_buffer_t *restrict buff,
    void *restrict data);

/* Pushes list of elements to the buffer. */
FANG_API FANG_HOT int fang_buffer_concat(fang_buffer_t *restrict buff,
    void *data, size_t count);

/* Get element data denoted by index. */
FANG_API FANG_HOT void *fang_buffer_get(fang_buffer_t *restrict buff,
    ptrdiff_t index);

/* Retrieve buffer pointer and count from structure. */
FANG_API void *fang_buffer_retrieve(fang_buffer_t *restrict buff,
    size_t *restrict count);

/* Shrink buffer capacity to fit element count. */
/* May mess up memory size alignment. Hence, recommended to use when buffer is
 * being returned or data is getting retrieved. */
FANG_API void fang_buffer_shrink_to_fit(fang_buffer_t *restrict buff);

/* Destroy a buffer. */
FANG_API void fang_buffer_release(fang_buffer_t *restrict buff);

/* ================ DECLARATIONS END ================ */

#endif  // FANG_BUFFER_H
