#include <fang/util/buffer.h>
#include <fang/status.h>
#include <string.h>

/* ================ PRIVATE MACROS ================ */

/* Responsible for exponential growth of capacity (by twice). */
#define _FANG_GROW_CAPACITY(x)    ((x) < FANG_BUFFER_INIT_CAPACITY ?    \
    FANG_BUFFER_INIT_CAPACITY : (x) * 2)

#define _FANG_REPORT(buff, result)                                      \
    if(FANG_UNLIKELY(buff == NULL)) {                                   \
        result = -FANG_NOMEM;                                           \
        goto out;                                                       \
    }

/* ================ PRIVATE MACROS END ================ */


/* ================ DEFINITIONS ================ */

/* Initializes buffer structure. */
int fang_buffer_create(fang_buffer_t *restrict buff,
    fang_reallocator_t realloc, int n)
{
    int res = FANG_OK;

    buff->realloc = realloc;
    buff->count = 0;
    buff->n = n;

    /* Initial allocation. */
    buff->capacity = _FANG_GROW_CAPACITY(0);
    buff->data = realloc(NULL, buff->capacity * buff->n);
    _FANG_REPORT(buff->data, res);

out:
    return res;
}

/* Pushes a single element to the buffer. */
int fang_buffer_add(fang_buffer_t *restrict buff, void *restrict data) {
    int res = FANG_OK;

    /* Increase buffer capacity if need be. */
    if(FANG_LIKELY(buff->count + 1 > buff->capacity)) {
        buff->capacity = _FANG_GROW_CAPACITY(buff->capacity);
        buff->data     = buff->realloc(buff->data, buff->capacity * buff->n);
        _FANG_REPORT(buff->data, res);
    }

    memcpy((char *) buff->data + (buff->count * buff->n), data, buff->n);
    buff->count++;

out:
    return res;
}

/* Concatenates a NULL terminated buffer. */
int fang_buffer_concat(fang_buffer_t *restrict buff, void *data) {
    int res = FANG_OK;

    /* Think of the buffer as a pure string buffer not taking termination
       element/character into account. */
    int count = strlen((const char *) data);
    /* Keep increasing buffer until it's OK. */
    while(FANG_LIKELY(buff->count + count > buff->capacity)) {
        buff->capacity = _FANG_GROW_CAPACITY(buff->capacity);
        buff->data     = buff->realloc(buff->data, buff->capacity * buff->n);
        _FANG_REPORT(buff->data, res);
    }

    /* Copy as pure string buffer. */
    memcpy((char *) buff->data + (buff->count * buff->n), data, count);
    buff->count += count;

out:
    return res;
}

/* Pushes list of elements to the buffer. */
int fang_buffer_append(fang_buffer_t *restrict buff, void *data, size_t count) {
    int res = FANG_OK;

    /* Keep increasing buffer until it's OK. */
    while(FANG_LIKELY(buff->count + count > buff->capacity)) {
        buff->capacity = _FANG_GROW_CAPACITY(buff->capacity);
        buff->data     = buff->realloc(buff->data, buff->capacity * buff->n);
        _FANG_REPORT(buff->data, res);
    }

    memcpy((char *) buff->data + (buff->count * buff->n), data,
        count * buff->n);
    buff->count += count;

out:
    return res;
}

/* Get element data denoted by index. */
void *fang_buffer_get(fang_buffer_t *restrict buff, ptrdiff_t index) {
    void *res = NULL;

    if(index < 0)
        index += buff->count;

    if(FANG_UNLIKELY(index < 0 || (size_t) index >= buff->count))
        goto out;

    res = (char *) buff->data + (index * buff->n);

out:
    return res;
}

/* Retrieve buffer pointer and count from structure. */
void *fang_buffer_retrieve(fang_buffer_t *buff, size_t *restrict count) {
    if(FANG_LIKELY(count != NULL))
        *count = buff->count;

    return buff->data;
}

/* Shrink buffer capacity to fit element count. */
void fang_buffer_shrink_to_fit(fang_buffer_t *restrict buff) {
    buff->data = buff->realloc(buff->data, buff->count * buff->n);
    buff->capacity = buff->count;
}

/* Destroy a buffer. */
void fang_buffer_release(fang_buffer_t *restrict buff) {
    if(FANG_LIKELY(buff->data))
        FANG_RELEASE(buff->realloc, buff->data);
}

/* ================ DEFINITIONS END ================ */
