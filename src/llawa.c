//
// Created by Ellen7ions on 2023/8/9.
//
#include "llawa.h"

#define LLAWA_TENSOR_SIZE sizeof(struct llawa_tensor)
#define LLAWA_OBJECT_SIZE sizeof(struct llawa_object)

struct llawa_object {
    llawa_tensor *tensor;
    size_t offset;
    size_t size;
};

size_t llawa_dtype_size[LLAWA_COUNT] = {
        1, 2, 4, 2, 4
};

int llawa_context_init(llawa_context *ctx, size_t mem_size) {
    *ctx = (llawa_context) {
            .mem_size = mem_size,
            .mem = NULL,
            .end_offset = 0
    };
    ctx->mem = LLAWA_ALLOC_MEM(mem_size);
    if (ctx->mem == NULL) return 0;
    return 1;
}

/**
 * Memory Layout:
 * |obj | tensor | data |
 * @param ctx
 * @param dtype
 * @param n_dim
 * @param ne
 * @param data
 * @return
 */
llawa_tensor *llawa_new_tensor(
        llawa_context *ctx,
        llawa_dtype dtype,
        uint32_t n_dim,
        const uint32_t *ne,
        void *data
) {
    void *ctx_mem_end = ctx->mem + ctx->end_offset;

    struct llawa_object *new_obj = (struct llawa_object *) (ctx_mem_end);
    llawa_tensor *new_tensor = (struct llawa_tensor *) (ctx_mem_end + LLAWA_OBJECT_SIZE);
    void *new_data = (void *) (ctx_mem_end + LLAWA_OBJECT_SIZE + LLAWA_TENSOR_SIZE);

    size_t tensor_data_sz = 0;
    if (data == NULL) {
        tensor_data_sz += llawa_dtype_size[dtype];
        for (int i = 0; i < n_dim; i++) tensor_data_sz *= ne[i];
    }

    size_t total_sz = LLAWA_OBJECT_SIZE + LLAWA_TENSOR_SIZE + tensor_data_sz;
    *new_obj = (struct llawa_object) {
            .tensor = NULL,
            .offset = ctx->end_offset,
            .size = total_sz,
    };

    *new_tensor = (struct llawa_tensor) {
            .data = data == NULL ? new_data : data,
            .n_dim = n_dim,
            .ne = {0},
            .op_type = LLAWA_OP_NONE,
            .dtype = dtype,
            .src = {NULL},
    };

    for (int i = 0; i < n_dim; i++) {
        new_tensor->ne[i] = ne[i];
    }

    ctx->end_offset += total_sz;
    return new_tensor;
}

uint32_t llawa_tensor_bytes_size(llawa_tensor *tensor) {
    uint32_t dtype_size = llawa_dtype_size[tensor->dtype];
    return dtype_size * llawa_tensor_elem_size(tensor);
}

uint32_t llawa_tensor_elem_size(llawa_tensor *tensor) {
    uint32_t sz = 1;
    for (int i = 0; i < tensor->n_dim; i++) sz *= tensor->ne[i];
    return sz;
}

size_t llawa_sizeof_dtype(llawa_dtype dtype) {
    return llawa_dtype_size[dtype];
}
