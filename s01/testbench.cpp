#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include "forward.h"
#include "config.h"

// === HLS C-simulation harness (S1: 110M config) =============================
// 토큰 주입 방식. 110M model.bin(버전2)을 로드해서 forward()를 돌리고
// greedy argmax 토큰 ID를 출력한다.
//
// GOLDEN 확보 방법 (보드 또는 PC에서):
//   - 새 host_llama_dbg가 프롬프트 인코딩 결과 토큰 ID를 출력하므로,
//     검증된 기존 improved 110M 보드 출력과 토큰 단위로 비교하면 된다.
//   - 아래 prompt_tokens 기본값은 llama2 tokenizer의 "I am happy"
//     (BOS=1 포함)이며, 실제 ID는 host의 [tokens] 출력으로 반드시 재확인할 것.

static bool load_model_v2(const char* path,
    Transformer<dim,hidden_dim,n_layers,n_heads,n_kv_heads,vocab_size,seq_len,GS>* t) {
    FILE* file = fopen(path, "rb");
    if (!file) { std::cerr << "Could not open " << path << "\n"; return false; }
    uint32_t magic; fread(&magic, sizeof(uint32_t), 1, file);
    if (magic != 0x616b3432) { std::cerr << "Bad magic\n"; return false; }
    int version; fread(&version, sizeof(int), 1, file);
    if (version != 2) { std::cerr << "Unsupported version\n"; return false; }
    Config fcfg; fread(&fcfg, sizeof(Config) - sizeof(int), 1, file);
    if (fcfg.dim != dim || fcfg.n_layers != n_layers || fcfg.vocab_size != vocab_size) {
        std::cerr << "config mismatch: file dim=" << fcfg.dim
                  << " layers=" << fcfg.n_layers
                  << " vocab=" << fcfg.vocab_size << "\n";
        return false;
    }
    uint8_t shared_classifier; fread(&shared_classifier, sizeof(uint8_t), 1, file);
    int group_size; fread(&group_size, sizeof(int), 1, file);
    fseek(file, 256, SEEK_SET);
    auto w = &t->weights;
    auto read_qt = [&](auto* tensor, int count, long size_each) {
        for (int i = 0; i < count; i++) {
            fread(tensor[i].q, sizeof(int8_t), size_each, file);
            fread(tensor[i].s, sizeof(float), size_each / GS, file);
        }
    };
    constexpr long kv_dim = (long)dim * n_kv_heads / n_heads;
    fread(w->rms_att_weight,   sizeof(float), n_layers*dim, file);
    fread(w->rms_ffn_weight,   sizeof(float), n_layers*dim, file);
    fread(w->rms_final_weight, sizeof(float), dim, file);
    read_qt(w->q_tokens, 1, (long)vocab_size*dim);
    read_qt(w->wq, n_layers, (long)dim*dim);
    read_qt(w->wk, n_layers, dim*kv_dim);
    read_qt(w->wv, n_layers, dim*kv_dim);
    read_qt(w->wo, n_layers, (long)dim*dim);
    read_qt(w->w1, n_layers, (long)dim*hidden_dim);
    read_qt(w->w2, n_layers, (long)dim*hidden_dim);
    read_qt(w->w3, n_layers, (long)dim*hidden_dim);
    if (shared_classifier)
        std::memcpy(w->wcls, w->q_tokens, sizeof(QuantizedTensor<vocab_size*dim>));
    else
        read_qt(w->wcls, 1, (long)dim*vocab_size);
    fclose(file);
    return true;
}

int main() {
    auto* transformer = new Transformer<dim,hidden_dim,n_layers,n_heads,n_kv_heads,vocab_size,seq_len,GS>();
    if (!load_model_v2("model.bin", transformer)) return EXIT_FAILURE;

    // <<< 110M llama2 tokenizer: BOS + " I"=306, " am"=626, " happy"=9796 (검증 완료) >>>
    int prompt_tokens[] = { 1, 306, 626, 9796 };
    int num_prompt_tokens = sizeof(prompt_tokens) / sizeof(prompt_tokens[0]);

    float *key_cache   = (float *)calloc((size_t)n_layers * seq_len * ((dim * n_kv_heads) / n_heads), sizeof(float));
    float *value_cache = (float *)calloc((size_t)n_layers * seq_len * ((dim * n_kv_heads) / n_heads), sizeof(float));
    float *logits      = (float *)calloc(vocab_size, sizeof(float));

    int pos = 0, next = 0;
    const int steps = 10;

    std::cout << "pos\ttoken_in\targmax_out\ttop_logit" << std::endl;
    int token = prompt_tokens[0];
    while (pos < steps) {
        forward(transformer, token, pos, key_cache, value_cache, logits);
        float max_val = logits[0]; int max_idx = 0;
        for (int j = 1; j < vocab_size; j++) {
            if (logits[j] > max_val) { max_val = logits[j]; max_idx = j; }
        }
        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];          // teacher-forcing through the prompt
        } else {
            next = max_idx;
        }
        std::cout << pos << "\t" << token << "\t" << max_idx << "\t" << max_val << std::endl;
        token = next;
        pos++;
    }

    free(key_cache); free(value_cache); free(logits);
    delete transformer;
    return 0;
}
