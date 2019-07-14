/*
 * Copyright (c) 2013-2015 Mellanox Technologies, Inc.
 *                         All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */
#include "oshmem_config.h"

#include "oshmem/constants.h"
#include "oshmem/include/shmem.h"

#include "oshmem/shmem/shmem_api_logger.h"

#include "oshmem/runtime/runtime.h"
#include "oshmem/mca/memheap/memheap.h"

#if OSHMEM_PROFILING
#include "oshmem/include/pshmem.h"
#pragma weak shmem_malloc = pshmem_malloc
#pragma weak shmem_calloc = pshmem_calloc
#pragma weak shmalloc = pshmalloc
#include "oshmem/shmem/c/profile/defines.h"
#endif

static inline void* _shmalloc(size_t size);

void* shmem_malloc(size_t size)
{
    return _shmalloc(size);
}

void* shmem_calloc(size_t count, size_t size)
{
    size_t req_sz = count * size;
    void *ptr = _shmalloc(req_sz);
    if (ptr) {
        memset(ptr, 0, req_sz);
    }
    return ptr;
}

void* shmalloc(size_t size)
{
    return _shmalloc(size);
}

static inline void* _shmalloc(size_t size)
{
    int rc;
    void* pBuff = NULL;

    RUNTIME_CHECK_INIT();
    RUNTIME_CHECK_WITH_MEMHEAP_SIZE(size);

    SHMEM_MUTEX_LOCK(shmem_internal_mutex_alloc);

    rc = MCA_MEMHEAP_CALL(alloc(size, &pBuff));

    SHMEM_MUTEX_UNLOCK(shmem_internal_mutex_alloc);

    if (OSHMEM_SUCCESS != rc) {
        SHMEM_API_VERBOSE(10,
                          "Allocation with shmalloc(size=%lu) failed.",
                          (unsigned long)size);
        return NULL ;
    }
#if OSHMEM_SPEC_COMPAT == 1
    shmem_barrier_all();
#endif
    return pBuff;
}
