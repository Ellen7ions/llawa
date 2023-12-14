//
// Created by Ellen7ions on 2023/8/9.
//
#include <string>
#include <fstream>
#include <cstring>
#include <iostream>
#include <map>
#include <vector>
#include <regex>
#include <cmath>
#include <random>

#include "llawa.h"

struct gpt2_hparams {
    int n_vocab;
    int n_ctx;
    int n_embd;
    int n_head;
    int n_layer;
    int ftype;
};

struct gpt2 {
    struct gpt2_hparams hparams;
    llawa_context context;

    std::map<std::string, int> token_to_id;
    std::map<int, std::string> id_to_token;

    std::map<std::string, llawa_tensor *> tensors;
};

bool gpt2_context_init(gpt2 &model) {
    size_t ctx_size = 0;
    {
        const auto &hparams = model.hparams;
        const int n_embd = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_ctx = hparams.n_ctx;
        const int n_vocab = hparams.n_vocab;
        const auto wtype = static_cast<const llawa_dtype>(hparams.ftype);

        ctx_size += n_embd * llawa_sizeof_dtype(LLAWA_F32); // ln_f_g
        ctx_size += n_embd * llawa_sizeof_dtype(LLAWA_F32); // ln_f_b

        ctx_size += n_vocab * n_embd * llawa_sizeof_dtype(wtype);         // wte
        ctx_size += n_ctx * n_embd * llawa_sizeof_dtype(LLAWA_F32); // wpe
        ctx_size += n_vocab * n_embd * llawa_sizeof_dtype(wtype);         // lm_head

        ctx_size += n_layer * (n_embd * llawa_sizeof_dtype(LLAWA_F32)); // ln_1_g
        ctx_size += n_layer * (n_embd * llawa_sizeof_dtype(LLAWA_F32)); // ln_1_b

        ctx_size += n_layer * (n_embd * llawa_sizeof_dtype(LLAWA_F32)); // ln_2_g
        ctx_size += n_layer * (n_embd * llawa_sizeof_dtype(LLAWA_F32)); // ln_2_b

        ctx_size += n_layer * (3 * n_embd * n_embd * llawa_sizeof_dtype(wtype));         // c_attn_attn_w
        ctx_size += n_layer * (3 * n_embd * llawa_sizeof_dtype(LLAWA_F32)); // c_attn_attn_b

        ctx_size += n_layer * (n_embd * n_embd * llawa_sizeof_dtype(wtype));           // c_attn_proj_w
        ctx_size += n_layer * (n_embd * llawa_sizeof_dtype(LLAWA_F32));   // c_attn_proj_b

        ctx_size += n_layer * (4 * n_embd * n_embd * llawa_sizeof_dtype(wtype));         // c_mlp_fc_w
        ctx_size += n_layer * (4 * n_embd * llawa_sizeof_dtype(LLAWA_F32)); // c_mlp_fc_b

        ctx_size += n_layer * (4 * n_embd * n_embd * llawa_sizeof_dtype(wtype));         // c_mlp_proj_w
        ctx_size += n_layer * (n_embd * llawa_sizeof_dtype(LLAWA_F32)); // c_mlp_proj_b

        ctx_size += n_ctx * n_layer * n_embd * llawa_sizeof_dtype(LLAWA_F32); // memory_k
        ctx_size += n_ctx * n_layer * n_embd * llawa_sizeof_dtype(LLAWA_F32); // memory_v

        ctx_size += (6 + 12 * n_layer) * 512; // object overhead

        printf("%s: llawa ctx size = %6.2f MB\n", __func__, ctx_size / (1024.0 * 1024.0));
    }
    return llawa_context_init(&model.context, ctx_size);
}

bool gpt2_load(gpt2 &model, const std::string &filename, bool verbose) {
    auto fs = std::ifstream(filename, std::ios::binary);

    if (!fs.is_open()) {
        std::cerr << "Can not find model from " << filename << std::endl;
        return false;
    }

    {
        char magic_str[6];
        fs.read(magic_str, 5);
        if (!strcmp(magic_str, "awall")) {
            std::cerr << "Magic number is wrong!" << std::endl;
            return false;
        }
    }

    {
        struct gpt2_hparams *hparams = &model.hparams;
        fs.read((char *) (&hparams->n_vocab), 4);
        fs.read((char *) (&hparams->n_ctx), 4);
        fs.read((char *) (&hparams->n_embd), 4);
        fs.read((char *) (&hparams->n_head), 4);
        fs.read((char *) (&hparams->n_layer), 4);
        fs.read((char *) (&hparams->ftype), 4);
#ifdef LLAWA_DEBUG
        if (verbose)
            std::cerr <<
                      "n_vocab  : " << hparams->n_vocab << std::endl <<
                      "n_ctx    : " << hparams->n_ctx << std::endl <<
                      "n_embd   : " << hparams->n_embd << std::endl <<
                      "n_head   : " << hparams->n_head << std::endl <<
                      "n_layer  : " << hparams->n_layer << std::endl <<
                      "ftype    : " << hparams->ftype << std::endl;
#endif

    }


    // load vocab
    {
        uint32_t dummy_n_vocab;
        fs.read((char *) (&dummy_n_vocab), 4);
        LLAWA_ASSERT(dummy_n_vocab == model.hparams.n_vocab);

        std::string word;
        std::vector<char> buf(128);

        for (int i = 0; i < dummy_n_vocab; i++) {
            uint32_t len;
            fs.read((char *) &len, sizeof(len));

            buf.clear();
            fs.read((char *) buf.data(), len);
            word.assign(buf.data(), len);

            model.token_to_id[std::string(word)] = i;
            model.id_to_token[i] = word;
        }

        LLAWA_ASSERT(model.token_to_id.size() == model.hparams.n_vocab);
    }

    if (!gpt2_context_init(model)) {
        std::cerr << "Context initialization error!" << std::endl;
        return false;
    }

    // load tensor
    while (true) {
        uint32_t n_dims, length, dtype;
        fs.read((char *) (&n_dims), sizeof(uint32_t));
        if (fs.eof()) break;
        fs.read((char *) (&length), sizeof(uint32_t));
        fs.read((char *) (&dtype), sizeof(uint32_t));
        uint32_t ne[LLAWA_MAX_DIM] = {1, 1, 1, 1};
        uint32_t stride[LLAWA_MAX_DIM] = {0, 0, 0, 0};
        for (int i = 0; i < n_dims; i++) fs.read((char *) (ne + i), sizeof(uint32_t));
        LLAWA_INIT_STRIDE(stride, ne);

        char buf[128];
        fs.read(buf, length);
        std::string name(buf, length);

#ifdef LLAWA_DEBUG
        if (verbose) {
            std::cerr << "load tensor: " << name << " -> [";
            for (int i = 0; i < n_dims; i++) {
                std::cerr << ne[i] << ", ";
            }
            std::cerr << "]" << std::endl;
        }
#endif
        auto tensor = llawa_new_tensor(&model.context, static_cast<llawa_dtype>(dtype),
                                       n_dims, ne, stride, nullptr);
        model.tensors[name] = tensor;
        uint32_t bytes_sz = llawa_tensor_bytes_size(tensor);
        fs.read((char *) (tensor->data), bytes_sz);
    }

    fs.close();
    return true;
}

llawa_tensor *gpt2_layer_norm(gpt2 &model, llawa_tensor *inp, llawa_tensor *w, llawa_tensor *bias) {
    llawa_tensor *res = llawa_zeros_like(&model.context, inp);

    llawa_tensor *mean_dst = llawa_new_tensor4d(&model.context, inp->dtype,
                                                inp->ne[0], 1, inp->ne[2], inp->ne[3],
                                                nullptr);
    llawa_tensor *std_dst = llawa_new_tensor4d(&model.context, inp->dtype,
                                               inp->ne[0], 1, inp->ne[2], inp->ne[3],
                                               nullptr);

    llawa_mean(&model.context, inp, 1, mean_dst);

    llawa_tensor *sub_dst = llawa_zeros_like(&model.context, inp);
    llawa_sub(&model.context, inp, mean_dst, sub_dst);
    llawa_std(&model.context, inp, mean_dst, 1, std_dst);

    float val = 1e-8;
    llawa_add(&model.context, std_dst, llawa_scalar(&model.context, LLAWA_F32, &val), std_dst);

    llawa_div(&model.context, sub_dst, std_dst, res);
    llawa_new_axis(&model.context, w, 0, w);
    llawa_new_axis(&model.context, bias, 0, bias);
    llawa_mul_dot(&model.context, res, w, res);
    llawa_add(&model.context, res, bias, res);
    return res;
}

llawa_tensor *gpt2_split_heads(gpt2 &model, llawa_tensor *tensor, bool is_key = false) {
    assert(tensor->n_dim == 2);
    uint32_t heads = model.hparams.n_head;
    uint32_t ne[LLAWA_MAX_DIM] = {tensor->ne[0], heads, tensor->ne[1] / heads, 1};
    auto res = llawa_view(&model.context, tensor, 3, ne);
    uint32_t pm_ne[LLAWA_MAX_DIM] = {1, 0, 2, 3};
    if (is_key) {
        pm_ne[1] = 2, pm_ne[2] = 0;
        return llawa_permute(&model.context, res, pm_ne);
    }
    return llawa_permute(&model.context, res, pm_ne);
}

llawa_tensor *gpt2_merge_heads(gpt2 &model, llawa_tensor *tensor) {
    assert(tensor->n_dim == 3);
    uint32_t pm_ne[LLAWA_MAX_DIM] = {1, 0, 2, 3};
    auto res = llawa_permute(&model.context, tensor, pm_ne);

//    for (int i = 0; i < res->ne[1]; i++)
//        for (int j = 0; j < res->ne[1]; j++)
//            printf("%f\n", llawa_tensor_get_val_f32(&model.context, res, 0, i, j, 0));

    res = llawa_contiguous(&model.context, res);

    uint32_t heads = model.hparams.n_head;
    uint32_t ne[LLAWA_MAX_DIM] = {tensor->ne[1], tensor->ne[0] * tensor->ne[2], 1, 1};
    res = llawa_view(&model.context, res, 2, ne);
    return res;
}

llawa_tensor *gpt2_mask(gpt2 &model, llawa_tensor *bias, int seq_n) {
    auto res = llawa_new_tensor2d(&model.context, bias->dtype, seq_n, seq_n, bias->data);
    res->stride[0] = bias->stride[2];
    return res;
}

llawa_tensor *gpt2_attention(
        gpt2 &model,
        llawa_tensor *inp,
        llawa_tensor *bias,
        llawa_tensor *attn_w,
        llawa_tensor *attn_bias,
        llawa_tensor *proj_w,
        llawa_tensor *proj_bias
) {
    llawa_tensor *qkv = llawa_new_tensor2d(&model.context, LLAWA_F32,
                                           inp->ne[0], attn_w->ne[1], nullptr);
    llawa_mat_mul(&model.context, inp, attn_w, qkv);
    llawa_new_axis(&model.context, attn_bias, 0, attn_bias);
    llawa_add(&model.context, qkv, attn_bias, qkv);

    uint32_t n3;
    llawa_tensor **qkv_splits = nullptr, *q, *k, *v;
    qkv_splits = llawa_split(&model.context, qkv, inp->ne[1], 1, &n3);

    q = qkv_splits[0];
    k = qkv_splits[1];
    v = qkv_splits[2];

    q = gpt2_split_heads(model, q);
    k = gpt2_split_heads(model, k, true);
    v = gpt2_split_heads(model, v);

    llawa_tensor *qk_dst = llawa_new_tensor4d(&model.context, LLAWA_F32, q->ne[0], q->ne[1], q->ne[1], 1, nullptr);
    qk_dst->n_dim = 3;

    llawa_mat_mul(&model.context, q, k, qk_dst);
    float scale = 1. / sqrt(v->ne[2]);
    llawa_mul_dot(&model.context, qk_dst, llawa_scalar(&model.context, LLAWA_F32, &scale), qk_dst);

    llawa_tensor *mask = gpt2_mask(model, bias, qk_dst->ne[1]);
    llawa_new_axis(&model.context, mask, 0, mask);
    llawa_tensor *neg_mask = llawa_zeros_like(&model.context, mask);

    {
        float k = -1;
        llawa_mul_dot(&model.context, mask, llawa_scalar(&model.context, LLAWA_F32, &k), neg_mask);
        k = 1;
        llawa_add(&model.context, neg_mask, llawa_scalar(&model.context, LLAWA_F32, &k), neg_mask);
        float inf = -1e10;
        llawa_mul_dot(&model.context, neg_mask, llawa_scalar(&model.context, LLAWA_F32, &inf), neg_mask);

        llawa_mul_dot(&model.context, qk_dst, mask, qk_dst);
        llawa_add(&model.context, qk_dst, neg_mask, qk_dst);
    }

    llawa_softmax(&model.context, qk_dst, -1, qk_dst);
    auto attn_v = llawa_new_tensor3d(&model.context, LLAWA_F32, qk_dst->ne[0], qk_dst->ne[1], v->ne[2], nullptr);

    llawa_mat_mul(&model.context, qk_dst, v, attn_v);
    attn_v = gpt2_merge_heads(model, attn_v);

    llawa_tensor *res = llawa_zeros_like(&model.context, inp);
    llawa_mat_mul(&model.context, attn_v, proj_w, res);
    llawa_new_axis(&model.context, proj_bias, 0, proj_bias);
    llawa_add(&model.context, res, proj_bias, res);
    return res;
}

llawa_tensor *gpt2_mlp(
        gpt2 &model,
        llawa_tensor *inp,
        llawa_tensor *mlp_w,
        llawa_tensor *mlp_bias,
        llawa_tensor *proj_w,
        llawa_tensor *proj_bias
) {
    llawa_tensor *mlp_dst = llawa_new_tensor2d(&model.context, LLAWA_F32,
                                               inp->ne[0], mlp_w->ne[1], nullptr);
    llawa_mat_mul(&model.context, inp, mlp_w, mlp_dst);
    llawa_new_axis(&model.context, mlp_bias, 0, mlp_bias);
    llawa_add(&model.context, mlp_dst, mlp_bias, mlp_dst);
    llawa_gelu(&model.context, mlp_dst, mlp_dst);

    llawa_tensor *proj_dst = llawa_new_tensor2d(&model.context, LLAWA_F32,
                                                mlp_dst->ne[0], proj_w->ne[1], nullptr);
    llawa_mat_mul(&model.context, mlp_dst, proj_w, proj_dst);
    llawa_new_axis(&model.context, proj_bias, 0, proj_bias);
    llawa_add(&model.context, proj_dst, proj_bias, proj_dst);
    return proj_dst;
}

llawa_tensor *gpt2_layer_forward(gpt2 &model, llawa_tensor *inp, int c_layer) {
    // attn
    {
        llawa_tensor *norm = gpt2_layer_norm(
                model, inp,
                model.tensors["h." + std::to_string(c_layer) + ".ln_1.weight"],
                model.tensors["h." + std::to_string(c_layer) + ".ln_1.bias"]
        );


        llawa_tensor *x_attn = gpt2_attention(
                model,
                norm,
                model.tensors["h." + std::to_string(c_layer) + ".attn.bias"],
                model.tensors["h." + std::to_string(c_layer) + ".attn.c_attn.weight"],
                model.tensors["h." + std::to_string(c_layer) + ".attn.c_attn.bias"],
                model.tensors["h." + std::to_string(c_layer) + ".attn.c_proj.weight"],
                model.tensors["h." + std::to_string(c_layer) + ".attn.c_proj.bias"]
        );

        llawa_add(&model.context, inp, x_attn, inp);

    }

    // mlp
    {
        llawa_tensor *norm = gpt2_layer_norm(
                model, inp,
                model.tensors["h." + std::to_string(c_layer) + ".ln_2.weight"],
                model.tensors["h." + std::to_string(c_layer) + ".ln_2.bias"]
        );

        llawa_tensor *x_mlp = gpt2_mlp(
                model,
                norm,
                model.tensors["h." + std::to_string(c_layer) + ".mlp.c_fc.weight"],
                model.tensors["h." + std::to_string(c_layer) + ".mlp.c_fc.bias"],
                model.tensors["h." + std::to_string(c_layer) + ".mlp.c_proj.weight"],
                model.tensors["h." + std::to_string(c_layer) + ".mlp.c_proj.bias"]
        );

        llawa_add(&model.context, inp, x_mlp, inp);
    }

    return inp;
}

int gpt2_forward(
        gpt2 &model,
        const std::vector<int> &token_ids,
        const std::vector<int> &pos_ids,
        int n_past,
        std::vector<int> *past_cache,
        std::vector<float> *ret_logits,
        std::vector<int> *ret_present_cache
) {
    auto wte = model.tensors["wte.weight"];
    auto wpe = model.tensors["wpe.weight"];
    auto token_ids_tensor = llawa_new_tensor1d(&model.context, LLAWA_I32, token_ids.size(), nullptr);
    auto pos_ids_tensor = llawa_new_tensor1d(&model.context, LLAWA_I32, pos_ids.size(), nullptr);
    for (auto i = 0; i < token_ids.size(); i++)
        llawa_tensor_set_val_i32(&model.context, token_ids_tensor, i, 0, 0, 0, token_ids[i]);

    for (auto i = 0; i < pos_ids.size(); i++)
        llawa_tensor_set_val_i32(&model.context, pos_ids_tensor, i, 0, 0, 0, pos_ids[i]);

    auto tokens_embd = llawa_get_rows(&model.context, wte, token_ids_tensor);
    auto pos_embd = llawa_get_rows(&model.context, wpe, pos_ids_tensor);

    auto inp_embd = llawa_zeros_like(&model.context, tokens_embd);
    llawa_add(&model.context, tokens_embd, pos_embd, inp_embd);


    // ?
//    if (past_cache == nullptr) {
//        past_cache = new std::vector<int>(12, -1);
//    }
//
//    auto *present_cache = new std::vector<int>();
//
    auto cur = inp_embd;
    for (int c_layer = 0; c_layer < model.hparams.n_layer; c_layer++) {
//        auto *kv_cache = llawa_new_tensor(&model.context, LLAWA_F32,);
        cur = gpt2_layer_forward(model, cur, c_layer);
//        break;
//        present_cache->push_back(kv_cache);
    }
//
    cur = gpt2_layer_norm(model, cur, model.tensors["ln_f.weight"], model.tensors["ln_f.bias"]);
    uint32_t pne[4] = {1, 0, 2, 3};
    llawa_tensor *logits_tensor = llawa_zeros_like(&model.context, cur);
    llawa_mat_mul(&model.context, cur, llawa_permute(&model.context, wte, pne), logits_tensor);
    cur = logits_tensor;
//    ret_present_cache = present_cache;
    for (int i = 0; i < cur->ne[1]; i++)
        ret_logits->push_back(llawa_tensor_get_val_f32(&model.context, cur, cur->ne[0] - 1, i, 0, 0));
    return 0;
}

void gpt_split_words(std::string str, std::vector<std::string> &words) {
    const std::string pattern = R"('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\s[:alpha:][:digit:]]+|\s+(?!\S)|\s+)";
    const std::regex re(pattern);
    std::smatch m;

    while (std::regex_search(str, m, re)) {
        for (auto x: m) {
            words.push_back(x);
        }
        str = m.suffix();
    }
}

std::vector<int> gpt_tokenize(const gpt2 &context, const std::string &text) {
    std::vector<std::string> words;

    // first split the text into words
    {
        std::string str = text;

        gpt_split_words(str, words);
    }

    // find the longest token that forms each word in words:
    std::vector<int> tokens;
    for (const auto &word: words) {
        for (int i = 0; i < (int) word.size();) {
            for (int j = word.size() - 1; j >= i; j--) {
                auto cand = word.substr(i, j - i + 1);
                auto it = context.token_to_id.find(cand);
                if (it != context.token_to_id.end()) { // word.substr(i, j-i+1) in vocab
                    tokens.push_back(it->second);
                    i = j + 1;
                    break;
                } else if (j == i) { // word.substr(i, 1) has no matching
                    fprintf(stderr, "%s: unknown token '%s'\n", __func__, word.substr(i, 1).data());
                    i++;
                }
            }
        }
    }

    return tokens;
}

std::string gpt2_sampling(
        gpt2 &model,
        std::vector<float> &logits,
        float temperature,
        int topk,
        std::mt19937 &rng
) {
    for (float &logit: logits) logit /= temperature;
    std::vector<std::pair<int, float>> logits_pos;
    for (int i = 0; i < logits.size(); i++)
        logits_pos.emplace_back(i, logits[i]);
    std::sort(
            logits_pos.begin(),
            logits_pos.end(),
            [](const std::pair<int, float> p1, const std::pair<int, float> p2) {
                return p1.second > p2.second;
            }
    );
    std::vector<std::pair<int, float>> topk_logits;
    for (int i = 0; i < topk; i++)
        topk_logits.push_back(logits_pos[i]);

    // softmax
    {
        float max1 = -INFINITY;
        for (auto i: topk_logits) max1 = std::max(max1, i.second);
        float sum = 0;
        for (std::pair<int, float> &topk_logit: topk_logits) {
            topk_logit.second = expf(topk_logit.second - max1);
            sum += topk_logit.second;
        }
        for (std::pair<int, float> &f: topk_logits)
            f.second /= sum;
    }

    std::vector<float> probs;
    for (std::pair<int, float> p: topk_logits) probs.push_back(p.second);

    std::discrete_distribution<> dist(probs.begin(), probs.end());
    int idx = dist(rng);

    int id = topk_logits[idx].first;
    return model.id_to_token[id];
}

int main(int argc, char *argv[]) {
    std::string model_path = "./llawa_gpt2.bin";
    std::mt19937 rng(0);
    gpt2 model;

    std::string prompt = "This is a story about ";
    int max_token = 5;

    for (int step = 0; step < max_token; step++) {
        std::cout << prompt << std::endl;
        gpt2_load(model, model_path, false);
        auto prompt_tokens = gpt_tokenize(model, prompt);
        auto pos = std::vector<int>();
        for (auto i = 0; i < prompt_tokens.size(); i++) pos.push_back(i);

        auto *ret_logits = new std::vector<float>;
        auto *ret_present_cache = new std::vector<int>;
        gpt2_forward(model, prompt_tokens, pos, 0,
                     nullptr,
                     ret_logits,
                     ret_present_cache);
        std::string token = gpt2_sampling(model, *ret_logits, 0.8, 5, rng);
        prompt += (token + " ");
        llawa_context_destroy(&model.context);
    }

    return 0;
}