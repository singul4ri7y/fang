#include <fang/status.h>
#include <platform/env/cpu.h>
#include <unistd.h>
#include <cpuid.h>

/* ================ DEFINITIONS ================ */

/* Gets CPU information (how many cores, start index etc.) in Linux. */
int _fang_env_cpu_getinfo(int *restrict nproc) {
    /* Get number of proccessors. */
    *nproc = sysconf(_SC_NPROCESSORS_ONLN);
    return FANG_OK;
}

/* ================ DEFINITIONS END ================ */
