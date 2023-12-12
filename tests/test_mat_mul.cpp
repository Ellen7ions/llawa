#include <iostream>
#include "llawa.h"

int main() {
#define N 5
    auto *ctx = new llawa_context;
    llawa_context_init(ctx, sizeof(float) * (N + 200));
    auto t1 = llawa_new_tensor4d(ctx, LLAWA_F32, 1, 1, 2, 2, nullptr);
    auto t2 = llawa_new_tensor4d(ctx, LLAWA_F32, 1, 1, 2, 2, nullptr);

    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++) {
            llawa_tensor_set_val_f32(ctx, t1, 0, 0, i, j, i + j);
            llawa_tensor_set_val_f32(ctx, t2, 0, 0, i, j, i - j);
        }

    auto res = llawa_zeros_like(ctx, t1);
    llawa_mat_mul(ctx, t1, t2, res);

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            auto val = llawa_tensor_get_val_f32(ctx, res, 0, 0, i, j);
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
