#ifndef ERRORCHECK_H
#define ERRORCHECK_H

#ifdef DEBUG

#include <stdio.h>
#include <stdlib.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

__host__
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#else /* DEBUG */

#define gpuErrchk(ans) ans;

#endif /* DEBUG */
#endif /* ERRORCHECK_H */ 