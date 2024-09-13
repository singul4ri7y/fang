#include <fang/status.h>
#include <fang/util/buffer.h>
#include <platform/env/cpu.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* ================ PRIVATE MACROS ================ */



/* ================ PRIVATE MACROS END ================ */


/* ================ DEFINITIONS ================ */

/* Gets CPU information (how many cores, start index etc.) in Linux. */
int _fang_env_cpu_getinfo(_fang_cpu_t **restrict cpu_buff, int *restrict ncpu,
    fang_reallocator_t realloc)
{
    int res = FANG_OK;

    int _BUFSIZ = 1024;
    char cbuff[_BUFSIZ];

    FILE *info = fopen("/proc/cpuinfo", "r");
    if(info == NULL) {
        res = -FANG_NOINFO;
        goto out;
    }

    fang_buffer_t buff;
    FANG_BUFFER_CREATE(&buff, realloc, _fang_cpu_t);

    int curr_id = -1;

    /* Reconnaissance :). */
    while(fgets(cbuff, _BUFSIZ, info)) {
        /* Potential physical CPU. */
        if(strncmp(cbuff, "physical id", 11) == 0) {
            /* Is it a proper physical ID field? */
            char *colon = strchr(cbuff, ':');
            if(colon == NULL)
                continue;

            /* In Linux, physical CPU IDs are increamenting. */
            int phy_id = atoi(colon + 2);  // Skip ": "
            /* New physical CPU. */
            if(phy_id > curr_id) {
                _fang_cpu_t cpu = { 0 };
                if(!FANG_ISOK(res = fang_buffer_add(&buff, &cpu)))
                    goto out;

                curr_id = phy_id;
            }
        }
        /* Processors count the physical CPU holds. */
        /* It's a good thing "siblings: " field comes after "physical id: "
           field. */
        else if(strncmp(cbuff, "siblings", 8) == 0) {
            /* Proper "siblings: " field? */
            char *colon = strchr(cbuff, ':');
            if(colon && curr_id != -1) {
                _fang_cpu_t *cpu = FANG_BUFFER_GET(&buff, _fang_cpu_t, curr_id);
                if(cpu ->nproc == 0) {
                    int nproc = atoi(colon + 2);  // Skip ": "

                    cpu->nproc = nproc;
                    cpu->nact  = nproc;  // All processors are active by default

                    if(curr_id > 0) {
                        _fang_cpu_t *prev_cpu =
                            FANG_BUFFER_GET(&buff, _fang_cpu_t, curr_id - 1);

                        cpu->sproc = prev_cpu->sproc
                            + prev_cpu->nproc;
                    } else cpu->sproc = 0;  // Very first CPU.
                }
            }
        }
    }

    /* No element would be pushed anymore. */
    fang_buffer_shrink_to_fit(&buff);

    size_t size;
    *cpu_buff = fang_buffer_retrieve(&buff, &size);
    *ncpu = (int) size;

out:
    fclose(info);

    return res;
}

/* ================ DEFINITIONS END ================ */
