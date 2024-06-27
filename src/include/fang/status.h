#ifndef FANG_STATUS_H
#define FANG_STATUS_H

/* Check whether the status is OK. */
#define FANG_OK(exp)    ((exp) >= 0)

/* Generic OK. Something greater or equal to 0 should be OK. */
#define FANG_GENOK       0

/* Generic error. */
#define FANG_GEN         1

/* Invalid ID. */
#define FANG_NOID        2

/* No memory. */
#define FANG_NOMEM       4

/* ---------------- PLATFORM ---------------- */

/* No platform found. */
#define FANG_NOPL      100

/* Platform is still holding onto some tensors. */
#define FANG_NTEN      101

/* No info. */
#define FANG_NOINFO    102

/* Invalid platform or platform mismatch. */
#define FANG_INVPL     103

/* ---------------- PLATFORM END ---------------- */

/* ---------------- TENSOR ---------------- */

/* Invalid datatype. */
#define FANG_INVTYP    201

/* Could not create or start CPU threads. */
#define FANG_NOTHRD    202

/* Could not pin thread to a specific CPU core. */
#define FANG_NOPIN     203

/* Invalid dimension. */
#define FANG_INVDIM    204

/* ---------------- TENSOR END ---------------- */

#endif    // FANG_STATUS_H
