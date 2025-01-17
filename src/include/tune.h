/* ALL THE FINE-TUNING PARAMETERS GO HERE. */

#ifndef FANG_TUNE_H
#define FANG_TUNE_H

/* ============================================= */
/*                      CPU                      */
/* ============================================= */


/* ================ GEMM ================ */

/* ======== SINGLE-PRECISION GEMM ======== */

/* Cache blocking parameters. Adjust accordingly with empirical analysis. */
#define FANG_SGEMM_MC              4032  // Ensure `MR` alignment
#define FANG_SGEMM_NC              128   // Ensure `NR` alignment
#define FANG_SGEMM_KC              228

/* Micro-kernel index (MRxNR):
 *     _fang_sgemm_6x16_ukernel: 0 (MR = 6, NR = 16)
 */
#define FANG_SGEMM_KERNEL          0
// TODO: Add more kernels.

/* How much threads to allocate to each of the loops (excluding loop 1 and 4)
   for parallelization. */
#define FANG_SGEMM_LOOP5_NT        1
#define FANG_SGEMM_LOOP3_NT        1
#define FANG_SGEMM_LOOP2_NT        1

/* ======== SINGLE-PRECISION GEMM END ======== */

/* ================ GEMM END ================ */


/* ============================================= */
/*                      CPU END                  */
/* ============================================= */

#endif  // FANG_TUNE_H
