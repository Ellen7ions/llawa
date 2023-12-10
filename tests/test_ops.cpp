//
// Created by lzz on 12/10/23.
//
#include "iostream"
#include "llawa.h"

int main() {
#define N 5
    auto *ctx = new llawa_context;
    llawa_context_init(ctx, sizeof(float) * (N + 200));
    llawa_tensor *x = llawa_new_tensor1d(ctx, LLAWA_F32, 5, nullptr);
    for (int i = 0; i < N; i++)
        llawa_tensor_set_val_f32(ctx, x, i, 0, 0, 0, (float) i);
    llawa_tensor *mean_dst = llawa_new_tensor4d(ctx, LLAWA_F32, 1, 1, 1, 1, nullptr);
    llawa_tensor *std_dst = llawa_new_tensor4d(ctx, LLAWA_F32, 1, 1, 1, 1, nullptr);
    llawa_mean(ctx, x, 0, mean_dst);
    llawa_std(ctx, x, mean_dst, 0, std_dst);
    std::cout << llawa_tensor_get_val_f32(ctx, std_dst, 0, 0, 0, 0);
    return 0;
}