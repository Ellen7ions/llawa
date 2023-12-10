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

bool gpt2_load(gpt2 &model, const std::string &filename) {
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
        for (int i = 0; i < n_dims; i++) fs.read((char *) (ne + i), sizeof(uint32_t));

        char buf[128];
        fs.read(buf, length);
        std::string name(buf, length);

#ifdef LLAWA_DEBUG
        std::cerr << "load tensor: " << name << " -> [";
        for (int i = 0; i < n_dims; i++) {
            std::cerr << ne[i] << ", ";
        }
        std::cerr << "]" << std::endl;
#endif
        auto tensor = llawa_new_tensor(&model.context, static_cast<llawa_dtype>(dtype), n_dims, ne, NULL);
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

llawa_tensor *gpt2_attention(
        gpt2 &model,
        llawa_tensor *inp,
        llawa_tensor *bias,
        llawa_tensor *attn_w,
        llawa_tensor *attn_bias,
        llawa_tensor *proj_w,
        llawa_tensor *proj_bias
) {
    llawa_tensor *res = llawa_zeros_like(&model.context, inp);
    llawa_tensor *qkv = llawa_new_tensor2d(&model.context, LLAWA_F32,
                                           inp->ne[0], attn_w->ne[1], nullptr);
    llawa_mat_mul(&model.context, inp, attn_w, qkv);
    llawa_new_axis(&model.context, attn_bias, 0, attn_bias);
    llawa_add(&model.context, qkv, attn_bias, qkv);

    
    return res;
}

llawa_tensor *gpt2_layer_forward(gpt2 &model, llawa_tensor *inp, int c_layer) {
    // atten
    {
        llawa_tensor *norm = gpt2_layer_norm(
                model, inp,
                model.tensors["h." + std::to_string(c_layer) + ".ln_1.weight"],
                model.tensors["h." + std::to_string(c_layer) + ".ln_1.bias"]
        );
        llawa_add(&model.context, inp, norm, inp);

        llawa_tensor *x_attn = gpt2_attention(
                model,
                inp,
                model.tensors["h." + std::to_string(c_layer) + ".attn.bias"],
                model.tensors["h." + std::to_string(c_layer) + ".attn.c_attn.weight"],
                model.tensors["h." + std::to_string(c_layer) + ".attn.c_attn.bias"],
                model.tensors["h." + std::to_string(c_layer) + ".attn.c_proj.weight"],
                model.tensors["h." + std::to_string(c_layer) + ".attn.c_proj.bias"]
        );

//        norm = gpt2_layer_norm(model, inp);
    }

    // mlp
    {

    }
}

int gpt2_forward(
        gpt2 &model,
        const std::vector<int> &token_ids,
        const std::vector<int> &pos_ids,
        int n_past,
        std::vector<int> *past_cache,
        std::vector<int> *ret_logits,
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
//    tokens = gpt2_layer_norm(cur, model.tensors["ln_f.weights"], model.tensors["ln_f.bias"]);
//    ret_logits = llawa_mat_mul(tokens, llawa_transpose(wte));
//    ret_present_cache = present_cache;
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

int main(int argc, char *argv[]) {
    std::string model_path = "./llawa_gpt2.bin";

    gpt2 model;

    gpt2_load(model, model_path);

//    gpt2_eval(model);

    auto res = gpt_tokenize(model, "hello world! what's your name ?");
    auto pos = std::vector<int>();
    for (auto i = 0; i < res.size(); i++) pos.push_back(i);

    std::vector<int> *ret_logits = new std::vector<int>;
    std::vector<int> *ret_present_cache = new std::vector<int>;
    gpt2_forward(model, res, pos, 0,
                 nullptr,
                 ret_logits,
                 ret_present_cache);

    return 0;
}