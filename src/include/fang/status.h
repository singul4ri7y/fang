#ifndef FANG_STATUS_H
#define FANG_STATUS_H

/* Check whether the status is OK. */
#define FANG_OK(exp)    ((exp) >= 0)

/* Generic OK. Something greater or equal to 0 should be OK. */
#define FANG_GENOK      0

/* Generic error. */
#define FANG_GEN        1

/* Invalid ID. */
#define FANG_NOID       2

/* No memory. */
#define FANG_NOMEM      4

/* ---------------- PLATFORM ---------------- */

/* No platform found. */
#define FANG_NOPL      10

/* Platform is still holding onto some tensors. */
#define FANG_NTEN      11

/* No info. */
#define FANG_NOINFO    12

/* ---------------- PLATFORM END ---------------- */

/* ---------------- TENSOR ---------------- */

/* Invalid dimension. */
#define FANG_INVDIM    22

/* ---------------- TENSOR END ---------------- */

#endif    // FANG_STATUS_H
