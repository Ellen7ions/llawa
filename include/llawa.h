//
// Created by Ellen7ions on 2023/8/9.
//

#ifndef LLAWA_LLAWA_H
#define LLAWA_LLAWA_H


#define LLAWA_MAX_DIM       4
#define LLAWA_MAX_OP_SRC    4

// Memory management
#define LLAWA_ALLOC_MEM malloc
#define LLAWA_FREE_MEME free

#define LLAWA_INIT_STRIDE(x, ne)               \
{                                       \
*(x + 0) = ne[1] * ne[2] * ne[3];  \
*(x + 1) = ne[2] * ne[3];          \
*(x + 2) = ne[3];                  \
*(x + 3) = 1;                      \
}


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

#define LLAWA_MAX(x, y) ((x > y) ? x : y)
#define LLAWA_MIN(x, y) ((x < y) ? x : y)

//typedef enum llawa_op {
//    LLAWA_OP_NONE
//} llawa_op;

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

    uint32_t stride[LLAWA_MAX_DIM];

//    llawa_op op_type;
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
        const uint32_t *stride,
        void *data
);

llawa_tensor *llawa_new_tensor2d(
        llawa_context *ctx,
        llawa_dtype dtype,
        uint32_t d0,
        uint32_t d1,
        void *data
);

llawa_tensor *llawa_new_tensor3d(
        llawa_context *ctx,
        llawa_dtype dtype,
        uint32_t d0,
        uint32_t d1,
        uint32_t d2,
        void *data
);

llawa_tensor *llawa_new_tensor1d(
        llawa_context *ctx,
        llawa_dtype dtype,
        uint32_t d0,
        void *data
);

llawa_tensor *llawa_new_tensor4d(
        llawa_context *ctx,
        llawa_dtype dtype,
        uint32_t d0,
        uint32_t d1,
        uint32_t d2,
        uint32_t d3,
        void *data
);

llawa_tensor *llawa_zeros_like(
        llawa_context *ctx,
        llawa_tensor *tensor);

uint32_t llawa_tensor_bytes_size(llawa_tensor *tensor);

uint32_t llawa_tensor_elem_size(llawa_tensor *tensor);

// llawa ops

void llawa_tensor_set_val_f32(llawa_context *ctx, llawa_tensor *src, uint32_t d0, uint32_t d1, uint32_t d2, uint32_t d3,
                              float val);

float
llawa_tensor_get_val_f32(llawa_context *ctx, llawa_tensor *src, uint32_t d0, uint32_t d1, uint32_t d2, uint32_t d3);

void llawa_tensor_set_val_i32(llawa_context *ctx, llawa_tensor *src, uint32_t d0, uint32_t d1, uint32_t d2, uint32_t d3,
                              int32_t val);

llawa_tensor *llawa_get_rows(llawa_context *ctx, llawa_tensor *src, llawa_tensor *ids);


int llawa_mean(llawa_context *ctx, llawa_tensor *inp, int dim, llawa_tensor *dst);

int llawa_sum(llawa_context *ctx, llawa_tensor *src, int dim, llawa_tensor *dst);

int llawa_std(llawa_context *ctx, llawa_tensor *inp, llawa_tensor *mean, int dim, llawa_tensor *dst);

int llawa_mul_dot(llawa_context *ctx, llawa_tensor *src0, llawa_tensor *src1, llawa_tensor *dst);

int llawa_add(llawa_context *ctx, llawa_tensor *src0, llawa_tensor *src1, llawa_tensor *dst);

int llawa_sub(llawa_context *ctx, llawa_tensor *src0, llawa_tensor *src1, llawa_tensor *dst);

int llawa_div(llawa_context *ctx, llawa_tensor *src0, llawa_tensor *src1, llawa_tensor *dst);

int llawa_sqrt(llawa_context *ctx, llawa_tensor *src0, llawa_tensor *dst);

int llawa_exp(llawa_context *ctx, llawa_tensor *src0, llawa_tensor *dst);

llawa_tensor *llawa_scalar(llawa_context *ctx, llawa_dtype dtype, void *val);

int llawa_new_axis(llawa_context *ctx, llawa_tensor *src, int t0, llawa_tensor *dst);

int llawa_mat_mul(llawa_context *ctx, llawa_tensor *src0, llawa_tensor *src1, llawa_tensor *dst);

llawa_tensor **llawa_split(llawa_context *ctx, llawa_tensor *src, uint32_t sz, uint32_t dim, uint32_t *n);

llawa_tensor *
llawa_view(llawa_context *ctx, llawa_tensor *src, uint32_t new_n_dim, const uint32_t new_ne[LLAWA_MAX_DIM]);

llawa_tensor *llawa_permute(llawa_context *ctx, llawa_tensor *src, const uint32_t pm_ne[LLAWA_MAX_DIM]);

int llawa_softmax(llawa_context *ctx, llawa_tensor *src, int dim, llawa_tensor *dst);

llawa_tensor *llawa_contiguous(llawa_context *ctx, llawa_tensor *src);

#ifdef __cplusplus
};
#endif

#endif //LLAWA_LLAWA_H
