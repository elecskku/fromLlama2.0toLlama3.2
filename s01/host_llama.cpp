//========================================================================================
// XRT Native API Host Code for KV260 Llama Accelerator
//========================================================================================

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <utility>
#include <iomanip>
#include <limits>

#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_bo.h"

#include "forward.h"
#include "config.h"

//========================================================================================
// Tokenizer
//========================================================================================

struct TokenIndex { char *str; int id; };

struct Tokenizer {
    char **vocab;
    float *vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512];
};

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex *)a)->str, ((TokenIndex *)b)->str);
}

void build_tokenizer(Tokenizer *t, const std::string& tokenizer_path, int vocab_size) {
    t->vocab_size = vocab_size;
    t->vocab = (char **)malloc(vocab_size * sizeof(char *));
    t->vocab_scores = (float *)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL;
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    FILE *file = fopen(tokenizer_path.c_str(), "rb");
    if (!file) { std::cerr << "Failed to open tokenizer: " << tokenizer_path << "\n"; exit(1); }
    fread(&t->max_token_length, sizeof(int), 1, file);
    int len;
    for (int i = 0; i < vocab_size; i++) {
        fread(t->vocab_scores + i, sizeof(float), 1, file);
        fread(&len, sizeof(int), 1, file);
        t->vocab[i] = (char *)malloc(len + 1);
        fread(t->vocab[i], len, 1, file);
        t->vocab[i][len] = '\0';
    }
    fclose(file);
}

char *decode(Tokenizer *t, int prev_token, int token) {
    if (token < 0 || token >= t->vocab_size) return (char *)"[INVALID]";
    return t->vocab[token];
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    TokenIndex tok = {.str = str};
    TokenIndex *res = (TokenIndex *)bsearch(&tok, sorted_vocab, vocab_size,
                                            sizeof(TokenIndex), compare_tokens);
    return res ? res->id : -1;
}

void encode(Tokenizer *t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    if (text == NULL) { std::cerr << "NULL text\n"; exit(1); }
    if (t->sorted_vocab == NULL) {
        t->sorted_vocab = (TokenIndex *)malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }
    char *str_buffer = (char *)malloc((t->max_token_length * 2 + 3));
    size_t str_len = 0;
    *n_tokens = 0;
    if (bos) tokens[(*n_tokens)++] = 1;
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup((char *)" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }
    for (char *c = text; *c != '\0'; c++) {
        if ((*c & 0xC0) != 0x80) str_len = 0;
        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0';
        if ((*(c + 1) & 0xC0) == 0x80 && str_len < 4) continue;
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
        if (id != -1) {
            tokens[(*n_tokens)++] = id;
        } else {
            for (size_t i = 0; i < str_len; i++)
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
        }
        str_len = 0;
    }
    while (true) {
        float best_score = -1e10;
        int best_id = -1, best_idx = -1;
        for (int i = 0; i < (*n_tokens - 1); i++) {
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i + 1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }
        if (best_idx == -1) break;
        tokens[best_idx] = best_id;
        for (int i = best_idx + 1; i < (*n_tokens - 1); i++) tokens[i] = tokens[i + 1];
        (*n_tokens)--;
    }
    if (eos) tokens[(*n_tokens)++] = 2;
    free(str_buffer);
}

void softmax(float *x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < size; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }
    for (int i = 0; i < size; i++) x[i] /= sum;
}

//========================================================================================
// Main
//========================================================================================

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN> <Model> <Tokenizer> [pad_mb] [head|tail]\n";
        std::cout << "  pad_mb    : (S2) transformer 버퍼를 이 크기(MB)로 강제 할당. 0 또는 생략 = 패딩 없음\n";
        std::cout << "  head|tail : (S2) 가중치를 버퍼 앞(head, 기본) 또는 끝(tail)에 배치.\n";
        std::cout << "              tail은 커널이 높은 물리 주소를 읽게 만들어 1B와 같은 주소 조건을 재현.\n";
        return EXIT_FAILURE;
    }

    std::string xclbin_file(argv[1]), model_file(argv[2]), tokenizer_file(argv[3]);
    size_t pad_mb = (argc >= 5) ? (size_t)atoll(argv[4]) : 0;
    std::string placement = (argc >= 6) ? argv[5] : "head";

    std::cout << "=============================================================\n";
    std::cout << "  KV260 Llama Accelerator - XRT Native API\n";
    std::cout << "=============================================================\n\n";

    std::cout << "[1] Initializing FPGA device...\n";
    xrt::device device;
    xrt::uuid xclbin_uuid;
    try {
        device = xrt::device(0);
        xclbin_uuid = device.load_xclbin(xclbin_file);
        std::cout << "    ✓ Device initialized and XCLBIN loaded\n\n";
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n"; return EXIT_FAILURE;
    }

    std::cout << "[2] Loading model weights and tokenizer...\n";
    void *raw_ptr;
    if (posix_memalign(&raw_ptr, 4096,
        sizeof(Transformer<dim,hidden_dim,n_layers,n_heads,n_kv_heads,vocab_size,seq_len,GS>)) != 0) {
        std::cerr << "posix_memalign failed\n"; return EXIT_FAILURE;
    }
    std::unique_ptr<Transformer<dim,hidden_dim,n_layers,n_heads,n_kv_heads,vocab_size,seq_len,GS>,
                    decltype(&free)>
        transformer(reinterpret_cast<Transformer<dim,hidden_dim,n_layers,n_heads,
                    n_kv_heads,vocab_size,seq_len,GS>*>(raw_ptr), free);

    {
        FILE* file = fopen(model_file.c_str(), "rb");
        if (!file) { std::cerr << "Error: Could not open " << model_file << "\n"; return EXIT_FAILURE; }
        uint32_t magic; fread(&magic, sizeof(uint32_t), 1, file);
        if (magic != 0x616b3432) { std::cerr << "Bad magic\n"; return EXIT_FAILURE; }
        int version; fread(&version, sizeof(int), 1, file);
        if (version != 2) { std::cerr << "Unsupported version\n"; return EXIT_FAILURE; }
        Config config;
        fread(&config, sizeof(Config) - sizeof(int), 1, file);
        uint8_t shared_classifier; fread(&shared_classifier, sizeof(uint8_t), 1, file);
        int group_size; fread(&group_size, sizeof(int), 1, file);
        fseek(file, 256, SEEK_SET);
        auto w = &transformer->weights;
        auto read_qtensor = [&](auto* tensor, int count, int size_each) {
            for (int i = 0; i < count; i++) {
                fread(tensor[i].q, sizeof(int8_t), size_each, file);
                fread(tensor[i].s, sizeof(float), size_each / GS, file);
            }
        };
        fread(w->rms_att_weight,   sizeof(float), n_layers*dim, file);
        fread(w->rms_ffn_weight,   sizeof(float), n_layers*dim, file);
        fread(w->rms_final_weight, sizeof(float), dim, file);
        // kv_dim < dim under GQA (n_kv_heads < n_heads). For 110M they were equal.
        constexpr int kv_dim = (dim * n_kv_heads) / n_heads;
        read_qtensor(w->q_tokens, 1, vocab_size*dim);
        // (removed) full fp32 dequantize of embedding table — ~1GB at 128k vocab, unused now.
        read_qtensor(w->wq, n_layers, dim*dim);
        read_qtensor(w->wk, n_layers, dim*kv_dim);   // GQA: was dim*dim
        read_qtensor(w->wv, n_layers, dim*kv_dim);   // GQA: was dim*dim
        read_qtensor(w->wo, n_layers, dim*dim);
        read_qtensor(w->w1, n_layers, dim*hidden_dim);
        read_qtensor(w->w2, n_layers, hidden_dim*dim);
        read_qtensor(w->w3, n_layers, dim*hidden_dim);
        if (shared_classifier)
            std::memcpy(w->wcls, w->q_tokens, sizeof(QuantizedTensor<vocab_size*dim>));
        else
            read_qtensor(w->wcls, 1, dim*vocab_size);
        fclose(file);
    }

    Tokenizer tokenizer;
    {
        tokenizer.vocab_size = vocab_size;
        tokenizer.vocab = (char **)malloc(vocab_size * sizeof(char *));
        tokenizer.vocab_scores = (float *)malloc(vocab_size * sizeof(float));
        tokenizer.sorted_vocab = NULL;
        for (int i = 0; i < 256; i++) {
            tokenizer.byte_pieces[i * 2] = (unsigned char)i;
            tokenizer.byte_pieces[i * 2 + 1] = '\0';
        }
        FILE *file = fopen(tokenizer_file.c_str(), "rb");
        if (!file) { std::cerr << "Failed to open tokenizer\n"; return EXIT_FAILURE; }
        fread(&tokenizer.max_token_length, sizeof(int), 1, file);
        int len;
        for (int i = 0; i < vocab_size; i++) {
            fread(tokenizer.vocab_scores + i, sizeof(float), 1, file);
            fread(&len, sizeof(int), 1, file);
            tokenizer.vocab[i] = (char *)malloc(len + 1);
            fread(tokenizer.vocab[i], len, 1, file);
            tokenizer.vocab[i][len] = '\0';
        }
        fclose(file);
    }

    std::cout << "    ✓ Model loaded (dim=" << dim << ", layers=" << n_layers << ")\n";
    std::cout << "    ✓ Tokenizer initialized (vocab=" << vocab_size << ")\n\n";

    std::cout << "[3] Creating kernel and allocating buffers...\n";
    xrt::kernel krnl_forward;
    try {
        krnl_forward = xrt::kernel(device, xclbin_uuid, "forward");
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n"; return EXIT_FAILURE;
    }

    size_t transformer_size = sizeof(Transformer<dim,hidden_dim,n_layers,n_heads,n_kv_heads,vocab_size,seq_len,GS>);
    size_t kv_cache_size    = (size_t)n_layers * seq_len * ((dim * n_kv_heads) / n_heads);
    size_t out_size         = vocab_size;

    // ---- [S2] 버퍼 크기/물리 주소 축 실험용 패딩 할당 ----
    // pad_mb > 0 이면 transformer 버퍼를 pad_mb(MB)짜리 큰 CMA 버퍼로 잡고,
    // placement=head : 가중치를 버퍼 맨 앞에 (큰 "할당" 자체만 테스트)
    // placement=tail : 가중치를 버퍼 맨 뒤에 (커널 read가 base+~pad_mb 근처의
    //                  높은 물리 주소를 치게 함 = 1B 상황과 동일한 주소 조건)
    size_t alloc_size = transformer_size;
    if (pad_mb * 1024ull * 1024ull > alloc_size) alloc_size = pad_mb * 1024ull * 1024ull;

    xrt::bo transformer_parent = xrt::bo(device, alloc_size, krnl_forward.group_id(0));
    size_t sub_off = 0;
    if (placement == "tail" && alloc_size > transformer_size) {
        sub_off = ((alloc_size - transformer_size) / 4096) * 4096;  // 4K 정렬
    }
    xrt::bo transformer_buf = (sub_off > 0)
        ? xrt::bo(transformer_parent, alloc_size - sub_off, sub_off)
        : transformer_parent;
    xrt::bo key_cache_buf   = xrt::bo(device, kv_cache_size*sizeof(float), krnl_forward.group_id(3));
    xrt::bo value_cache_buf = xrt::bo(device, kv_cache_size*sizeof(float), krnl_forward.group_id(4));
    xrt::bo out_buf         = xrt::bo(device, out_size*sizeof(float),       krnl_forward.group_id(5));

    std::cout << "    ✓ Buffers allocated:\n";
    std::cout << "      - transformer: " << transformer_size/1024 << " KB"
              << " (alloc " << alloc_size/1024 << " KB, placement=" << placement
              << ", sub_off=" << sub_off << ")\n";
    std::cout << "      - key_cache:   " << kv_cache_size*sizeof(float)/1024 << " KB\n";
    std::cout << "      - value_cache: " << kv_cache_size*sizeof(float)/1024 << " KB\n";
    std::cout << "      - out:         " << out_size*sizeof(float)/1024 << " KB\n\n";

    // ---- 물리 주소 로그 (hang 디버깅의 1차 데이터) ----
    std::cout << std::hex;
    std::cout << "    [phys] transformer_parent base = 0x" << transformer_parent.address() << "\n";
    std::cout << "    [phys] transformer arg (kernel이 받는 주소) = 0x" << transformer_buf.address() << "\n";
    std::cout << "    [phys] transformer end = 0x" << (transformer_buf.address() + transformer_size) << "\n";
    std::cout << "    [phys] key_cache   = 0x" << key_cache_buf.address() << "\n";
    std::cout << "    [phys] value_cache = 0x" << value_cache_buf.address() << "\n";
    std::cout << "    [phys] out         = 0x" << out_buf.address() << "\n";
    std::cout << std::dec;
    if (transformer_buf.address() + transformer_size > 0x80000000ull) {
        std::cout << "    [!] WARNING: transformer 버퍼가 low-DDR 경계(0x8000_0000)를 넘음!\n";
        std::cout << "        HP 포트 address map에 DDR_HIGH가 없으면 여기서 hang이 발생함.\n";
    }
    std::cout << "\n";

    std::cout << "[4] Transferring model weights to FPGA (one-time)...\n";
    void* tbuf_map = transformer_buf.map<void*>();
    std::memcpy(tbuf_map, transformer.get(), transformer_size);
    transformer_buf.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    std::cout << "    ✓ Weights transferred\n\n";

    float* out_map = out_buf.map<float*>();

    xrt::run run(krnl_forward);
    run.set_arg(0, transformer_buf);
    run.set_arg(3, key_cache_buf);
    run.set_arg(4, value_cache_buf);
    run.set_arg(5, out_buf);

    std::cout << "=============================================================\n";
    std::cout << "  Ready for interactive generation\n";
    std::cout << "  (Type your prompt and press Enter. Type 'exit' to quit)\n";
    std::cout << "=============================================================\n\n";

    while (true) {
        std::cout << "You: ";
        std::string user_input;
        std::getline(std::cin, user_input);
        if (user_input == "exit" || user_input == "quit") { std::cout << "\nExiting...\n"; break; }
        if (user_input.empty()) continue;

        std::vector<int> prompt_tokens(seq_len);
        int num_prompt_tokens = 0;
        encode(&tokenizer, (char*)user_input.c_str(), 1, 0,
               prompt_tokens.data(), &num_prompt_tokens);
        if (num_prompt_tokens <= 0) { std::cerr << "Encode failed\n"; continue; }
        int effective_prompt_tokens = std::min(num_prompt_tokens, seq_len);

        // 프롬프트 토큰 ID 출력 (CSIM testbench golden 및 유효성 검증용)
        std::cout << "[tokens] n=" << effective_prompt_tokens << " ids=";
        for (int i = 0; i < effective_prompt_tokens; i++) {
            std::cout << prompt_tokens[i] << (i + 1 < effective_prompt_tokens ? "," : "\n");
            if (prompt_tokens[i] < 0 || prompt_tokens[i] >= vocab_size) {
                std::cout << "\n[!] ERROR: 잘못된 토큰 ID " << prompt_tokens[i]
                          << " (vocab=" << vocab_size << ") — 커널에 넣으면 안 됨\n";
            }
        }
        std::cout.flush();

        {
            float* kc = key_cache_buf.map<float*>();
            float* vc = value_cache_buf.map<float*>();
            std::memset(kc, 0, kv_cache_size * sizeof(float));
            std::memset(vc, 0, kv_cache_size * sizeof(float));
            key_cache_buf.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            value_cache_buf.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }

        const int max_tokens = 256;
        int generated_tokens = 0;
        bool eos_encountered = false;
        bool decode_timing_started = false;

        int token = prompt_tokens[0];
        int pos = 0;
        auto start_time = std::chrono::high_resolution_clock::now();
        auto end_time   = start_time;

        std::cout << "Assistant: ";
        std::cout.flush();

        while (true) {
            if (pos >= seq_len) { std::cout << "\n[WARN] max seq len\n"; break; }

            run.set_arg(1, token);
            run.set_arg(2, pos);

            // ---- per-call 계측: 이 두 줄 사이에서 멈추면 '진짜 hang' 확정 ----
            std::cout << "[pos " << pos << "] token=" << token << " kernel start..." << std::flush;
            auto k0 = std::chrono::high_resolution_clock::now();
            run.start();
            run.wait();
            auto k1 = std::chrono::high_resolution_clock::now();
            auto kms = std::chrono::duration_cast<std::chrono::milliseconds>(k1 - k0).count();
            std::cout << " done in " << kms << " ms\n" << std::flush;

            int next = 0;
            if (pos < effective_prompt_tokens - 1) {
                next = prompt_tokens[pos + 1];
            } else {
                if (!decode_timing_started) {
                    start_time = std::chrono::high_resolution_clock::now();
                    decode_timing_started = true;
                }

                out_buf.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                softmax(out_map, vocab_size);

                float max_val = out_map[0]; next = 0;
                for (int i = 1; i < vocab_size; i++)
                    if (out_map[i] > max_val) { max_val = out_map[i]; next = i; }

                if (next == 2) { eos_encountered = true; break; }

                std::cout << decode(&tokenizer, token, next);
                std::cout.flush();

                generated_tokens++;
                if (generated_tokens >= max_tokens) break;
            }

            token = next;
            pos++;
        }

        end_time = std::chrono::high_resolution_clock::now();
        if (!eos_encountered && generated_tokens >= max_tokens)
            std::cout << "\n[WARN] max_tokens reached\n";
        std::cout << std::endl;

        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>
                           (end_time - start_time).count();
        if (generated_tokens > 0) {
            std::cout << "\n[Performance]\n";
            std::cout << "  Prompt tokens:    " << effective_prompt_tokens << "\n";
            std::cout << "  Generated tokens: " << generated_tokens << "\n";
            std::cout << "  Total time:       " << duration_ms << " ms\n";
            std::cout << "  Throughput:       "
                      << generated_tokens / (duration_ms / 1000.0) << " tokens/sec\n\n";
        }
    }

    std::cout << "\n=============================================================\n";
    std::cout << "  Session ended.\n";
    std::cout << "=============================================================\n";
    return EXIT_SUCCESS;
}
