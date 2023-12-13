//#include <iostream>
#include <cstdio>
#include "llawa.h"
#include "omp.h"

float a[5000][5000];
float b[5000][5000];
float c[5000][5000];

void mat_mul(int n) {

#pragma omp parallel for

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
//            std::cout << omp_get_num_threads() << std::endl;
            float sum = 0;
            for (int k = 0; k < n; k++) {
                sum += a[i][k] * b[k][j];
            }
            c[i][j] = sum;
        }
    }


}

int llawa_mat_mul_local(llawa_context *ctx, llawa_tensor *src0, llawa_tensor *src1, llawa_tensor *dst) {
//    assert(src0->ne[1] == src1->ne[0]);

    if (src0->n_dim == 4 && src1->n_dim == 4 && src0->ne[0] == src1->ne[0] && src0->ne[1] == src1->ne[1]) {

        for (int p = 0; p < src0->ne[0]; p++) {
            for (int q = 0; q < src0->ne[1]; q++) {
#pragma omp parallel
#pragma parallel for
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
    } else if (src0->n_dim == 3 && src1->n_dim == 3 && src0->ne[0] == src1->ne[0]) {
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
    } else if (src0->n_dim == 2 && src1->n_dim == 2) {
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
    } else
        assert(0);

    return 0;
}

int main() {
#define N 2000
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
//
//    llawa_mat_mul_local(ctx, t1, t2, t3);
    llawa_mat_mul(ctx, t1, t2, t3);

//    for (int i = 0; i < N; i++) {
//        for (int j = 0; j < N; j++) {
//            auto val = llawa_tensor_get_val_f32(ctx, res, 0, 0, i, j);
//            std::cout << val << " ";
//        }
//        std::cout << std::endl;
//    }

    return 0;
}
