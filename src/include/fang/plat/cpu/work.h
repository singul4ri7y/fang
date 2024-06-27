#ifndef FANG_PLAT_CPU_WORK_H
#define FANG_PLAT_CPU_WORK_H

/* Bitcast. */
#define _FANG_BITCAST(type, vtype, value)    ({                  \
    union { vtype src; type dest; } _u = { .src = value };       \
    _u.dest;                                                     \
})

#ifdef __linux__
/* ---------------- WORKER FUNCTIONS (KERNEL) ---------------- */

void *_fang_worker_fn_randf64(void *arg);
void *_fang_worker_fn_randf32(void *arg);
void *_fang_worker_fn_randi64(void *arg);
void *_fang_worker_fn_randi32(void *arg);
void *_fang_worker_fn_randi16(void *arg);
void *_fang_worker_fn_randi8(void *arg);
void *_fang_worker_fn_randu64(void *arg);
void *_fang_worker_fn_randu32(void *arg);
void *_fang_worker_fn_randu16(void *arg);
void *_fang_worker_fn_randu8(void *arg);

void *_fang_worker_fn_sumf64(void *arg);
void *_fang_worker_fn_sumf32(void *arg);
void *_fang_worker_fn_sumi64(void *arg);
void *_fang_worker_fn_sumi32(void *arg);
void *_fang_worker_fn_sumi16(void *arg);
void *_fang_worker_fn_sumi8(void *arg);
void *_fang_worker_fn_sumu64(void *arg);
void *_fang_worker_fn_sumu32(void *arg);
void *_fang_worker_fn_sumu16(void *arg);
void *_fang_worker_fn_sumu8(void *arg);

void *_fang_worker_fn_difff64(void *arg);
void *_fang_worker_fn_difff32(void *arg);
void *_fang_worker_fn_diffi64(void *arg);
void *_fang_worker_fn_diffi32(void *arg);
void *_fang_worker_fn_diffi16(void *arg);
void *_fang_worker_fn_diffi8(void *arg);
void *_fang_worker_fn_diffu64(void *arg);
void *_fang_worker_fn_diffu32(void *arg);
void *_fang_worker_fn_diffu16(void *arg);
void *_fang_worker_fn_diffu8(void *arg);

void *_fang_worker_fn_hadamardf64(void *arg);
void *_fang_worker_fn_hadamardf32(void *arg);
void *_fang_worker_fn_hadamardi64(void *arg);
void *_fang_worker_fn_hadamardi32(void *arg);
void *_fang_worker_fn_hadamardi16(void *arg);
void *_fang_worker_fn_hadamardi8(void *arg);
void *_fang_worker_fn_hadamardu64(void *arg);
void *_fang_worker_fn_hadamardu32(void *arg);
void *_fang_worker_fn_hadamardu16(void *arg);
void *_fang_worker_fn_hadamardu8(void *arg);

/* ---------------- WORKER FUNCTIONS (KERNEL) END ---------------- */
#endif // __linux__
#endif // FANG_PLAT_CPU_WORK_H
