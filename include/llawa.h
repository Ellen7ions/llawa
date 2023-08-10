//
// Created by Ellen7ions on 2023/8/9.
//

#ifndef LLAWA_LLAWA_H
#define LLAWA_LLAWA_H


#include "llawa_ops.h"

#define LLAWA_MAX_DIM       4
#define LLAWA_MAX_OP_SRC    4

// Memory management
#define LLAWA_ALLOC_MEM malloc
#define LLAWA_FREE_MEME free

// DEBUG


#ifdef __cplusplus
extern "C" {
#endif

#ifdef LLAWA_DEBUG

#include <assert.h>

#define LLAWA_ASSERT(x) assert(x)
#endif

#include <stdlib.h>
#include <stdint.h>

typedef enum llawa_dtype {
    LLAWA_I8,
    LLAWA_I16,
    LLAWA_I32,
    LLAWA_F16,
    LLAWA_F32,
    LLAWA_COUNT
} llawa_dtype;

//extern size_t llawa_dtype_size[LLAWA_COUNT];

size_t llawa_sizeof_dtype(llawa_dtype dtype);

typedef struct llawa_tensor {
    uint32_t n_dim;
    uint32_t ne[LLAWA_MAX_DIM];

    llawa_op op_type;
    llawa_dtype dtype;
    struct llawa_tensor *src[LLAWA_MAX_OP_SRC];

    void *data;
} llawa_tensor;

struct llawa_object;

typedef struct llawa_context {
    size_t mem_size;
    void *mem;

    size_t end_offset;
} llawa_context;

int llawa_context_init(llawa_context *ctx, size_t mem_size);

llawa_tensor *llawa_new_tensor(
        llawa_context *ctx,
        llawa_dtype dtype,
        uint32_t n_dim,
        const uint32_t *ne,
        void *data
);

uint32_t llawa_tensor_bytes_size(llawa_tensor *tensor);

uint32_t llawa_tensor_elem_size(llawa_tensor *tensor);

#ifdef __cplusplus
};
#endif

#endif //LLAWA_LLAWA_H
