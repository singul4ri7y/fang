#include <fang/plat/cpu.h>
#include <fang/status.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* Creates a CPU platform specific private structure. */
/* The structure itself is explicitly allocated. */
int _fang_platform_cpu_create(_fang_platform_cpu_t **restrict cpup, 
    fang_reallocator_t realloc) 
{
    int res = FANG_GENOK;

    _fang_platform_cpu_t *cpu_plat = FANG_CREATE(realloc, 
        sizeof(_fang_platform_cpu_t));

    if(cpu_plat == NULL) {
        res = -FANG_NOMEM;
        goto out;
    }

#ifdef __linux__
    {
        int bufsiz = 1024;
        char buf[bufsiz];
    
        FILE *file = fopen("/proc/cpuinfo", "r");

        if(file == NULL) {
            res = -FANG_NOINFO;
            goto out;
        }

        _fang_cpu_t *cpus = NULL;
        int curr_id = -1;
        int ncpu    = 0;

        while(fgets(buf, bufsiz, file)) {
            /* If we find "physical id : ", try processing it. If we find an ID 
               greater than the previous one, we got a new CPU. */
            if(strncmp(buf, "physical id\t: ", 14) == 0) {
                int phy_id = atoi(strchr(buf, ':') + 2);    // Skip colon and space
                
                /* We've got ourselves a new CPU. */
                if(phy_id > curr_id) {
                    /* In Linux, physical ids are increamenting. */
                    cpus = realloc(cpus, (phy_id + 1) * sizeof(_fang_cpu_t));

                    if(cpus == NULL) {
                        res = -FANG_NOMEM;
                        goto out;
                    }

                    memset(cpus + phy_id, 0, sizeof(_fang_cpu_t));
                    ncpu++;
                    curr_id = phy_id;
                }
            }
            /* If we find a model name, set the model name if it's not already set. */
            else if(strncmp(buf, "model name\t: ", 13) == 0) {
                if(curr_id != -1 && cpus[curr_id].name == NULL) {
                    /* There is a newline '\n' character at the end. */
                    char *start = strchr(buf, ':') + 2;    // Skip ": "

                    size_t len = strlen(start);

                    /* We will ignore the newline, but we also have to 
                       take account the terminating character. So, length
                       of both cancel out :). */
                    cpus[curr_id].name = FANG_CREATE(realloc, len);

                    if(cpus[curr_id].name == NULL) {
                        res = -FANG_NOMEM;
                        goto out;
                    }

                    strncpy(cpus[curr_id].name, start, len - 1);
                    cpus[curr_id].name[len - 1] = '\0';
                }
            }
            /* We also need total number threads a CPU hold. */
            else if(strncmp(buf, "siblings\t: ", 11) == 0) {
                if(curr_id != -1 && cpus[curr_id].nthread == 0) {
                    int threads = atoi(strchr(buf, ':') + 2);

                    cpus[curr_id].nthread = threads;
                    cpus[curr_id].nact    = threads;
                }
            }
        }

        cpu_plat -> ncpu = ncpu;
        cpu_plat -> cpu  = cpus;

        fclose(file);
    }
#endif

    *cpup = cpu_plat;

out: 
    return res;
}

/* Release CPU platform. */
void _fang_platform_cpu_release(void *restrict private, 
    fang_reallocator_t realloc) 
{
    _fang_platform_cpu_t *cpu_plat = (_fang_platform_cpu_t *) private;

    for(int i = 0; i < cpu_plat -> ncpu; i++) 
        FANG_RELEASE(realloc, cpu_plat -> cpu[i].name);

    FANG_RELEASE(realloc, cpu_plat -> cpu);
    FANG_RELEASE(realloc, cpu_plat);
}
