//
// Created by Ellen7ions on 2023/8/9.
//
#include <string>
#include <fstream>
#include <cstring>
#include <iostream>
#include <map>
#include <vector>

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
        uint32_t ne[LLAWA_MAX_DIM];
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

void gpt2_eval() {

}

int main(int argc, char *argv[]) {
    std::string model_path = "./llawa_gpt2.bin";

    gpt2 model;

    gpt2_load(model, model_path);
    return 0;
}