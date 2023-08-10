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
};

bool gpt2_load(gpt2 *model, const std::string &filename) {
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
        struct gpt2_hparams *hparams = &model->hparams;
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
        LLAWA_ASSERT(dummy_n_vocab == model->hparams.n_vocab);

        std::string word;
        std::vector<char> buf(128);

        for (int i = 0; i < dummy_n_vocab; i++) {
            uint32_t len;
            fs.read((char *) &len, sizeof(len));

            buf.clear();
            fs.read((char *) buf.data(), len);
            word.assign(buf.data(), len);

            model->token_to_id[std::string(word)] = i;
            model->id_to_token[i] = word;
        }

        LLAWA_ASSERT(model->token_to_id.size() == model->hparams.n_vocab);
    }

    // load tensor
    {

    }

    return true;
}

void gpt2_eval() {

}

int main(int argc, char *argv[]) {
    std::string model_path = "./llawa_gpt2.bin";
    llawa_context gpt2_w_ctx = llawa_context_init(4096);

    gpt2 model = (gpt2) {
            .hparams = gpt2_hparams{0},
            .context = gpt2_w_ctx,
    };

    gpt2_load(&model, model_path);
    return 0;
}