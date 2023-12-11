//
// Created by Ellen7ions on 2023/8/9.
//
#include "string.h"
#include "math.h"

#include "llawa.h"

#define LLAWA_TENSOR_SIZE sizeof(struct llawa_tensor)
#define LLAWA_OBJECT_SIZE sizeof(struct llawa_object)

struct llawa_object {
    llawa_tensor *tensor;
    size_t offset;
    size_t size;
};

typedef enum llawa_ops {
    LLAWA_ADD,
    LLAWA_SUB,
    LLAWA_MUL,
    LLAWA_DIV,
    LLAWA_SQRT,
    LLAWA_OPS_COUNT
} llawa_ops;

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
        const uint32_t *stride,
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
            .stride = {0},
//            .op_type = LLAWA_OP_NONE,
            .dtype = dtype,
            .src = {NULL},
    };

    for (int i = 0; i < LLAWA_MAX_DIM; i++) {
        new_tensor->ne[i] = ne[i];
        new_tensor->stride[i] = stride[i];
    }

    ctx->end_offset += total_sz;
    return new_tensor;
}

llawa_tensor *llawa_new_tensor2d(
        llawa_context *ctx,
        llawa_dtype dtype,
        uint32_t d0,
        uint32_t d1,
        void *data
) {
    uint32_t *ne = malloc(sizeof(uint32_t) * LLAWA_MAX_DIM);
    uint32_t *stride = malloc(sizeof(uint32_t) * LLAWA_MAX_DIM);
    memset(ne, 0, sizeof(uint32_t) * LLAWA_MAX_DIM);
    memset(stride, 0, sizeof(uint32_t) * LLAWA_MAX_DIM);
    for (int i = 0; i < LLAWA_MAX_DIM; i++) *(ne + i) = 1;
    *ne = d0;
    *(ne + 1) = d1;

    LLAWA_INIT_STRIDE(stride, ne);
    return llawa_new_tensor(ctx, dtype, 2, ne, stride, data);
}

llawa_tensor *llawa_new_tensor1d(
        llawa_context *ctx,
        llawa_dtype dtype,
        uint32_t d0,
        void *data
) {
    uint32_t *ne = malloc(sizeof(uint32_t) * LLAWA_MAX_DIM);
    uint32_t *stride = malloc(sizeof(uint32_t) * LLAWA_MAX_DIM);
    memset(ne, 0, sizeof(uint32_t) * LLAWA_MAX_DIM);
    memset(stride, 0, sizeof(uint32_t) * LLAWA_MAX_DIM);
    for (int i = 0; i < LLAWA_MAX_DIM; i++) *(ne + i) = 1;
    *ne = d0;

    LLAWA_INIT_STRIDE(stride, ne);
    return llawa_new_tensor(ctx, dtype, 1, ne, stride, data);
}

llawa_tensor *llawa_new_tensor4d(
        llawa_context *ctx,
        llawa_dtype dtype,
        uint32_t d0,
        uint32_t d1,
        uint32_t d2,
        uint32_t d3,
        void *data
) {

    uint32_t *ne = (uint32_t *) malloc(sizeof(uint32_t) * LLAWA_MAX_DIM);
    uint32_t *stride = (uint32_t *) malloc(sizeof(uint32_t) * LLAWA_MAX_DIM);
    memset(stride, 0, sizeof(uint32_t) * LLAWA_MAX_DIM);
    *(ne + 0) = d0;
    *(ne + 1) = d1;
    *(ne + 2) = d2;
    *(ne + 3) = d3;
    LLAWA_INIT_STRIDE(stride, ne);
    return llawa_new_tensor(ctx, dtype, 4, ne, stride, data);
}


llawa_tensor *llawa_zeros_like(
        llawa_context *ctx,
        llawa_tensor *tensor
) {
    llawa_tensor *res = llawa_new_tensor(ctx, tensor->dtype, tensor->n_dim, tensor->ne, tensor->stride, NULL);
    return res;
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

void llawa_tensor_set_val_f32(
        llawa_context *ctx, llawa_tensor *src0,
        uint32_t d0, uint32_t d1, uint32_t d2, uint32_t d3,
        float val
) {
    assert(src0->dtype == LLAWA_F32);
    *(float *)
            (src0->data +
             d0 * (src0->stride[0] * llawa_sizeof_dtype(src0->dtype)) +
             d1 * (src0->stride[1] * llawa_sizeof_dtype(src0->dtype)) +
             d2 * (src0->stride[2] * llawa_sizeof_dtype(src0->dtype)) +
             d3 * (src0->stride[3] * llawa_sizeof_dtype(src0->dtype))) = val;
}

float
llawa_tensor_get_val_f32(llawa_context *ctx, llawa_tensor *src0, uint32_t d0, uint32_t d1, uint32_t d2, uint32_t d3) {
    return *(float *)
            (src0->data +
             d0 * (src0->stride[0] * llawa_sizeof_dtype(src0->dtype)) +
             d1 * (src0->stride[1] * llawa_sizeof_dtype(src0->dtype)) +
             d2 * (src0->stride[2] * llawa_sizeof_dtype(src0->dtype)) +
             d3 * (src0->stride[3] * llawa_sizeof_dtype(src0->dtype)));
}

void llawa_tensor_set_val_i32(
        llawa_context *ctx, llawa_tensor *src0,
        uint32_t d0, uint32_t d1, uint32_t d2, uint32_t d3,
        int32_t val
) {
    assert(src0->dtype == LLAWA_I32);
    *(int32_t *)
            (src0->data +
             d0 * (src0->stride[0] * llawa_sizeof_dtype(src0->dtype)) +
             d1 * (src0->stride[1] * llawa_sizeof_dtype(src0->dtype)) +
             d2 * (src0->stride[2] * llawa_sizeof_dtype(src0->dtype)) +
             d3 * (src0->stride[3] * llawa_sizeof_dtype(src0->dtype))) = val;
}

size_t llawa_sizeof_dtype(llawa_dtype dtype) {
    return llawa_dtype_size[dtype];
}

llawa_tensor *llawa_get_rows(llawa_context *ctx, llawa_tensor *src, llawa_tensor *ids) {
    assert(src->dtype == LLAWA_F32);
    assert(ids->dtype == LLAWA_I32);
    llawa_tensor *res = llawa_new_tensor2d(ctx, src->dtype, ids->ne[0], src->ne[1], NULL);
//    llawa_tensor *res = dst;
    for (int i = 0; i < ids->ne[0]; i++) {
        for (int j = 0; j < src->ne[1]; j++) {
            // data[ids[i]][j][1][1] = src[ids[i]][j][1][1]
            // no strides ?
            uint32_t sub = *(int32_t *) (ids->data + i * llawa_sizeof_dtype(ids->dtype));
            float v = *(float *) (src->data + sub * (src->stride[0] * llawa_sizeof_dtype(src->dtype)) +
                                  j * llawa_sizeof_dtype(src->dtype));
            *(float *) (res->data + i * (src->stride[0] * llawa_sizeof_dtype(src->dtype)) +
                        j * llawa_sizeof_dtype(src->dtype)) = v;
        }
    }
    return res;
}

int llawa_exec_ops(llawa_context *ctx, llawa_tensor *src0, llawa_tensor *src1, llawa_tensor *dst, llawa_ops op) {
    uint32_t *ne = malloc(sizeof(uint32_t) * 4);
    *(ne + 0) = LLAWA_MAX(src0->ne[0], (src1 == NULL ? 0 : src1->ne[0]));
    *(ne + 1) = LLAWA_MAX(src0->ne[1], (src1 == NULL ? 0 : src1->ne[1]));
    *(ne + 2) = LLAWA_MAX(src0->ne[2], (src1 == NULL ? 0 : src1->ne[2]));
    *(ne + 3) = LLAWA_MAX(src0->ne[3], (src1 == NULL ? 0 : src1->ne[3]));

    assert(dst->ne[0] == ne[0] && dst->ne[1] == ne[1] && dst->ne[2] == ne[2] && dst->ne[3] == ne[3]);

    for (int i = 0; i < ne[0]; i++) {
        for (int j = 0; j < ne[1]; j++) {
            for (int k = 0; k < ne[2]; k++) {
                for (int q = 0; q < ne[3]; q++) {

                    float v0 = 0;
                    if (src0 != 0)
                        v0 = llawa_tensor_get_val_f32(ctx, src0,
                                                      (i % src0->ne[0]),
                                                      (j % src0->ne[1]),
                                                      (k % src0->ne[2]),
                                                      (q % src0->ne[3]));
                    float v1 = 0;
                    if (src1 != NULL)
                        v1 = llawa_tensor_get_val_f32(ctx, src1,
                                                      (i % src1->ne[0]),
                                                      (j % src1->ne[1]),
                                                      (k % src1->ne[2]),
                                                      (q % src1->ne[3]));
                    switch (op) {
                        case LLAWA_ADD:
                            llawa_tensor_set_val_f32(ctx, dst, i, j, k, q, v0 + v1);
                            break;
                        case LLAWA_SUB:
                            llawa_tensor_set_val_f32(ctx, dst, i, j, k, q, v0 - v1);
                            break;
                        case LLAWA_MUL:
                            llawa_tensor_set_val_f32(ctx, dst, i, j, k, q, v0 * v1);
                            break;
                        case LLAWA_DIV:
                            llawa_tensor_set_val_f32(ctx, dst, i, j, k, q, v0 / v1);
                            break;
                        case LLAWA_SQRT:
                            llawa_tensor_set_val_f32(ctx, dst, i, j, k, q, sqrtf(v0));
                            break;
                        default:
                            assert(0);
                    }

                }
            }
        }
    }
    return 0;
}

int llawa_add(llawa_context *ctx, llawa_tensor *src0, llawa_tensor *src1, llawa_tensor *dst) {
    return llawa_exec_ops(ctx, src0, src1, dst, LLAWA_ADD);
}

int llawa_sub(llawa_context *ctx, llawa_tensor *src0, llawa_tensor *src1, llawa_tensor *dst) {
    return llawa_exec_ops(ctx, src0, src1, dst, LLAWA_SUB);
}

int llawa_mul_dot(llawa_context *ctx, llawa_tensor *src0, llawa_tensor *src1, llawa_tensor *dst) {
    return llawa_exec_ops(ctx, src0, src1, dst, LLAWA_MUL);
}

int llawa_div(llawa_context *ctx, llawa_tensor *src0, llawa_tensor *src1, llawa_tensor *dst) {
    return llawa_exec_ops(ctx, src0, src1, dst, LLAWA_DIV);
}

int llawa_sqrt(llawa_context *ctx, llawa_tensor *src0, llawa_tensor *dst) {
    return llawa_exec_ops(ctx, src0, NULL, dst, LLAWA_SQRT);
}


int llawa_mean(llawa_context *ctx, llawa_tensor *inp, int dim, llawa_tensor *dst) {
    assert(inp->dtype == LLAWA_F32);

    llawa_tensor *res = dst;

    switch (dim) {
        case 0:
            for (int j = 0; j < inp->ne[1]; j++) {
                for (int k = 0; k < inp->ne[2]; k++) {
                    for (int q = 0; q < inp->ne[3]; q++) {
                        float c = 0;
                        for (int i = 0; i < inp->ne[0]; i++)
                            c += llawa_tensor_get_val_f32(ctx, inp, i, j, k, q);

                        llawa_tensor_set_val_f32(ctx, res, 0, j, k, q, c / ((float) inp->ne[0]));
                    }
                }
            }

            break;
        case 1:
            for (int i = 0; i < inp->ne[0]; i++)
                for (int k = 0; k < inp->ne[2]; k++) {
                    for (int q = 0; q < inp->ne[3]; q++) {
                        float c = 0;
                        for (int j = 0; j < inp->ne[1]; j++)
                            c += llawa_tensor_get_val_f32(ctx, inp, i, j, k, q);
                        llawa_tensor_set_val_f32(ctx, res, i, 0, k, q, c / ((float) inp->ne[1]));
                    }
                }
            break;
        case 2:
            for (int i = 0; i < inp->ne[0]; i++)
                for (int j = 0; j < inp->ne[1]; j++) {
                    for (int q = 0; q < inp->ne[3]; q++) {
                        float c = 0;
                        for (int k = 0; k < inp->ne[2]; k++)
                            c += llawa_tensor_get_val_f32(ctx, inp, i, j, k, q);
                        llawa_tensor_set_val_f32(ctx, res, i, j, 0, q, c / ((float) inp->ne[2]));
                    }
                }
            break;
        case 3:
            for (int i = 0; i < inp->ne[0]; i++)
                for (int j = 0; j < inp->ne[1]; j++) {
                    for (int k = 0; k < inp->ne[2]; k++) {
                        float c = 0;
                        for (int q = 0; q < inp->ne[3]; q++)
                            c += llawa_tensor_get_val_f32(ctx, inp, i, j, k, q);
                        llawa_tensor_set_val_f32(ctx, res, i, j, k, 0, c / ((float) inp->ne[3]));
                    }
                }
            break;
        default:
            assert(0);
    }

    return 0;
}

int llawa_std(llawa_context *ctx, llawa_tensor *inp, llawa_tensor *mean, int dim, llawa_tensor *dst) {
    assert(mean->ne[dim] == 1);

    llawa_tensor *sub = llawa_zeros_like(ctx, inp);
    llawa_sub(ctx, inp, mean, sub);
    llawa_mul_dot(ctx, sub, sub, sub);
    llawa_mean(ctx, sub, dim, dst);
    llawa_sqrt(ctx, dst, dst);
    return 0;
}

llawa_tensor *llawa_scalar(llawa_context *ctx, llawa_dtype dtype, void *val) {
    llawa_tensor *res = llawa_new_tensor1d(ctx, dtype, 1, NULL);

    switch (dtype) {
        case LLAWA_F32:
            llawa_tensor_set_val_f32(ctx, res, 0, 0, 0, 0, *(float *) val);
            break;
        default:
            assert(0);
    }
    return res;
}

int llawa_new_axis(llawa_context *ctx, llawa_tensor *src, int t0, llawa_tensor *dst) {
    if (t0 == 3) {
        assert(src->ne[3] == 1);
    }
    for (int i = 2; i >= 0; i--) dst->ne[i + 1] = dst->ne[i];
    dst->ne[0] = 1;
    dst->ne[t0] = 1;
    LLAWA_INIT_STRIDE(dst->stride, dst->ne);
    return 0;
}

int llawa_mat_mul(llawa_context *ctx, llawa_tensor *src0, llawa_tensor *src1, llawa_tensor *dst) {
//    assert(src0->ne[1] == src1->ne[0]);

    if (src0->ne[0] == src1->ne[0] && src0->ne[1] == src1->ne[1]) {
        for (int p = 0; p < src0->ne[0]; p++) {
            for (int q = 0; q < src0->ne[1]; q++) {

                for (int i = 0; i < src0->ne[2]; i++) {
                    for (int j = 0; j < src1->ne[3]; j++) {
                        float c = 0;
                        for (int k = 0; k < src0->ne[3]; k++) {
                            c += llawa_tensor_get_val_f32(ctx, src0, p, q, i, k) *
                                 llawa_tensor_get_val_f32(ctx, src1, p, q, k, j);
                        }
                        llawa_tensor_set_val_f32(ctx, dst, p, q, i, j, c);
                    }
                }

            }
        }
    } else if (src0->ne[0] == src1->ne[0]) {
        assert(src0->ne[3] == 1);
        for (int p = 0; p < src0->ne[0]; p++) {

            for (int i = 0; i < src0->ne[1]; i++) {
                for (int j = 0; j < src1->ne[2]; j++) {
                    float c = 0;
                    for (int k = 0; k < src0->ne[2]; k++) {
                        c += llawa_tensor_get_val_f32(ctx, src0, p, i, k, 0) *
                             llawa_tensor_get_val_f32(ctx, src1, p, k, j, 0);
                    }
                    llawa_tensor_set_val_f32(ctx, dst, p, i, j, 0, c);
                }
            }

        }
    } else {
        assert(src0->ne[3] == 1 && src0->ne[2] == 1);
        for (int i = 0; i < src0->ne[0]; i++) {
            for (int j = 0; j < src1->ne[1]; j++) {
                float c = 0;
                for (int k = 0; k < src0->ne[1]; k++) {
                    c += llawa_tensor_get_val_f32(ctx, src0, i, k, 0, 0) *
                         llawa_tensor_get_val_f32(ctx, src1, k, j, 0, 0);
                }
                llawa_tensor_set_val_f32(ctx, dst, i, j, 0, 0, c);
            }
        }
    }

    return 0;
}

llawa_tensor **llawa_split(llawa_context *ctx, llawa_tensor *src, uint32_t sz, uint32_t dim, uint32_t *n) {
    assert(src->ne[dim] % sz == 0);

    *n = src->ne[dim] / sz;
    llawa_tensor **dst = malloc(sizeof(llawa_tensor *) * (*n));

    for (int i = 0; i < *n; i++) {
        uint32_t *ne = malloc(sizeof(uint32_t) * LLAWA_MAX_DIM);
        uint32_t *stride = malloc(sizeof(uint32_t) * LLAWA_MAX_DIM);
        memcpy(ne, src->ne, sizeof(uint32_t) * LLAWA_MAX_DIM);
        memcpy(stride, src->stride, sizeof(uint32_t) * LLAWA_MAX_DIM);
        ne[dim] = sz;
//        stride[dim] += sz * i;
        llawa_tensor *new_tensor = llawa_new_tensor(
                ctx,
                src->dtype,
                src->n_dim,
                ne,
                stride,
                src->data + sz * i * llawa_sizeof_dtype(src->dtype)
        );
        dst[i] = new_tensor;
    }
    return dst;
}

llawa_tensor *llawa_view(llawa_context *ctx, llawa_tensor *src,
                         uint32_t new_n_dim,
                         const uint32_t new_ne[LLAWA_MAX_DIM]) {
    uint32_t *ne = malloc(sizeof(uint32_t) * LLAWA_MAX_DIM);
    uint32_t *stride = malloc(sizeof(uint32_t) * LLAWA_MAX_DIM);
    for (int i = 0; i < LLAWA_MAX_DIM; i++) { ne[i] = new_ne[i]; }
    LLAWA_INIT_STRIDE(stride, ne);
    for (int i = 0; i < LLAWA_MAX_DIM; i++)
        if (src->ne[i] == new_ne[i]) stride[i] = src->stride[i];

    return llawa_new_tensor(ctx, src->dtype, new_n_dim, ne, stride, src->data);
}

llawa_tensor *llawa_permute(llawa_context *ctx, llawa_tensor *src, const uint32_t *pm_ne) {
    uint32_t *ne = malloc(sizeof(uint32_t) * LLAWA_MAX_DIM);
    uint32_t *stride = malloc(sizeof(uint32_t) * LLAWA_MAX_DIM);
    for (int i = 0; i < LLAWA_MAX_DIM; i++) ne[i] = src->ne[pm_ne[i]], stride[i] = src->stride[pm_ne[i]];
//    LLAWA_INIT_STRIDE(stride, ne);
    return llawa_new_tensor(ctx, src->dtype, src->n_dim, ne, stride, src->data);
}
