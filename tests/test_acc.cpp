//#include <iostream>
#include <cstdio>
#include "llawa.h"
#include "omp.h"


int llawa_acc_f32(llawa_context *ctx, llawa_tensor *inp, int dim, float factor, llawa_tensor *dst) {
    LLAWA_ASSERT(inp->dtype == LLAWA_F32);

    llawa_tensor *res = dst;

    switch (dim) {
        case 0:
            for (int j = 0; j < inp->ne[1]; j++) {
                for (int k = 0; k < inp->ne[2]; k++) {
                    for (int q = 0; q < inp->ne[3]; q++) {
                        float c = 0;
                        for (int i = 0; i < inp->ne[0]; i++)
                            c += llawa_tensor_get_val_f32(ctx, inp, i, j, k, q);

                        llawa_tensor_set_val_f32(ctx, res, 0, j, k, q, c / factor);
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
                        llawa_tensor_set_val_f32(ctx, res, i, 0, k, q, c / factor);
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
                        llawa_tensor_set_val_f32(ctx, res, i, j, 0, q, c / factor);
                    }
                }
            break;
        case 3:
#pragma omp parallel for
            for (int i = 0; i < inp->ne[0]; i++)
                for (int j = 0; j < inp->ne[1]; j++) {
                    for (int k = 0; k < inp->ne[2]; k++) {
                        float c = 0;
                        for (int q = 0; q < inp->ne[3]; q++)
                            c += llawa_tensor_get_val_f32(ctx, inp, i, j, k, q);
                        llawa_tensor_set_val_f32(ctx, res, i, j, k, 0, c / factor);
                    }
                }
            break;
        default:
            LLAWA_ASSERT(0);
    }

    return 0;
}

int main() {
#define N 10000
//    mat_mul(N);

    auto *ctx = new llawa_context;
    llawa_context_init(ctx, sizeof(float) * (N * N) * 3);
    auto t1 = llawa_new_tensor4d(ctx, LLAWA_F32, 1, 1, N, N, nullptr);
    auto t2 = llawa_new_tensor4d(ctx, LLAWA_F32, 1, 1, N, N, nullptr);
    auto t3 = llawa_new_tensor4d(ctx, LLAWA_F32, 1, 1, N, N, nullptr);

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            llawa_tensor_set_val_f32(ctx, t1, 0, 0, i, j, i + j);
            llawa_tensor_set_val_f32(ctx, t2, 0, 0, i, j, i - j);
        }

//    llawa_acc_f32(ctx, t1, 0, 1, t3);
    llawa_exp(ctx, t1, t3);

//    for (int i = 0; i < N; i++) {
//        for (int j = 0; j < N; j++) {
//            auto val = llawa_tensor_get_val_f32(ctx, res, 0, 0, i, j);
//            std::cout << val << " ";
//        }
//        std::cout << std::endl;
//    }

    return 0;
}
