// Minimal llama2.c-style fp32 inference for ESP32 (no PSRAM).
// Weights: memory-mapped from the "model" flash partition.
// Tokenizer: embedded tok512.bin. KV cache + activations: heap.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_partition.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "esp_log.h"

#define CTX_LIMIT   128     // cap context so the KV cache fits in RAM
#define TEMPERATURE 0.8f    // fixed for comparable runs
#define SEED        1234ULL

static const char *TAG = "llm";
extern const uint8_t tok_start[] asm("_binary_tok512_bin_start");

typedef struct { int dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len; } Config;

typedef struct {
    const float *tok_emb, *rms_att, *wq, *wk, *wv, *wo, *rms_ffn, *w1, *w2, *w3, *rms_final, *wcls;
} Weights;

typedef struct {
    float *x, *xb, *xb2, *hb, *hb2, *q, *att, *logits, *kc, *vc;
} RunState;

static Config cfg;
static Weights w;
static RunState s;
static const char **vocab;
static int *vlen;

static void *alloc(size_t n) {
    void *p = heap_caps_calloc(n, 1, MALLOC_CAP_8BIT);
    if (!p) { ESP_LOGE(TAG, "alloc of %u bytes failed", (unsigned)n); abort(); }
    return p;
}

static void rmsnorm(float *o, const float *x, const float *wt, int n) {
    float ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / n + 1e-5f);
    for (int i = 0; i < n; i++) o[i] = wt[i] * (ss * x[i]);
}

static void softmax(float *x, int n) {
    float m = x[0];
    for (int i = 1; i < n; i++) if (x[i] > m) m = x[i];
    float sum = 0;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - m); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

// xout (d) = W (d,n) @ x (n)
static void matmul(float *xout, const float *x, const float *wt, int n, int d) {
    for (int i = 0; i < d; i++) {
        float v = 0;
        const float *row = wt + (size_t)i * n;
        for (int j = 0; j < n; j++) v += row[j] * x[j];
        xout[i] = v;
    }
}

static float *forward(int token, int pos) {
    int dim = cfg.dim, hid = cfg.hidden_dim;
    int kv_dim = (dim * cfg.n_kv_heads) / cfg.n_heads;
    int kv_mul = cfg.n_heads / cfg.n_kv_heads;
    int hs = dim / cfg.n_heads;
    int T = cfg.seq_len;

    memcpy(s.x, w.tok_emb + (size_t)token * dim, dim * sizeof(float));

    for (int l = 0; l < cfg.n_layers; l++) {
        rmsnorm(s.xb, s.x, w.rms_att + l * dim, dim);
        size_t loff = (size_t)l * T * kv_dim;
        float *k = s.kc + loff + (size_t)pos * kv_dim;
        float *v = s.vc + loff + (size_t)pos * kv_dim;
        matmul(s.q, s.xb, w.wq + (size_t)l * dim * dim, dim, dim);
        matmul(k,   s.xb, w.wk + (size_t)l * dim * kv_dim, dim, kv_dim);
        matmul(v,   s.xb, w.wv + (size_t)l * dim * kv_dim, dim, kv_dim);

        for (int i = 0; i < dim; i += 2) {
            int hd = i % hs;
            float freq = 1.0f / powf(10000.0f, hd / (float)hs);
            float val = pos * freq, fcr = cosf(val), fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1;
            for (int r = 0; r < rotn; r++) {
                float *vec = r == 0 ? s.q : k;
                float v0 = vec[i], v1 = vec[i + 1];
                vec[i] = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }

        for (int h = 0; h < cfg.n_heads; h++) {
            float *q = s.q + h * hs;
            float *att = s.att + h * T;
            for (int t = 0; t <= pos; t++) {
                const float *kk = s.kc + loff + (size_t)t * kv_dim + (h / kv_mul) * hs;
                float sc = 0;
                for (int i = 0; i < hs; i++) sc += q[i] * kk[i];
                att[t] = sc / sqrtf((float)hs);
            }
            softmax(att, pos + 1);
            float *xb = s.xb + h * hs;
            memset(xb, 0, hs * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                const float *vv = s.vc + loff + (size_t)t * kv_dim + (h / kv_mul) * hs;
                float a = att[t];
                for (int i = 0; i < hs; i++) xb[i] += a * vv[i];
            }
        }

        matmul(s.xb2, s.xb, w.wo + (size_t)l * dim * dim, dim, dim);
        for (int i = 0; i < dim; i++) s.x[i] += s.xb2[i];

        rmsnorm(s.xb, s.x, w.rms_ffn + l * dim, dim);
        matmul(s.hb,  s.xb, w.w1 + (size_t)l * dim * hid, dim, hid);
        matmul(s.hb2, s.xb, w.w3 + (size_t)l * dim * hid, dim, hid);
        for (int i = 0; i < hid; i++) {
            float val = s.hb[i];
            val *= 1.0f / (1.0f + expf(-val));
            s.hb[i] = val * s.hb2[i];
        }
        matmul(s.xb, s.hb, w.w2 + (size_t)l * hid * dim, hid, dim);
        for (int i = 0; i < dim; i++) s.x[i] += s.xb[i];
    }
    rmsnorm(s.x, s.x, w.rms_final, dim);
    matmul(s.logits, s.x, w.wcls, dim, cfg.vocab_size);
    return s.logits;
}

static uint64_t rng = SEED;
static float rand_f32(void) {
    rng ^= rng >> 12; rng ^= rng << 25; rng ^= rng >> 27;
    return ((rng * 0x2545F4914F6CDD1DULL) >> 32 & 0xFFFFFF) / 16777216.0f;
}

static int sample(float *logits, int n) {
    if (TEMPERATURE == 0.0f) {
        int b = 0;
        for (int i = 1; i < n; i++) if (logits[i] > logits[b]) b = i;
        return b;
    }
    for (int i = 0; i < n; i++) logits[i] /= TEMPERATURE;
    softmax(logits, n);
    float r = rand_f32(), c = 0;
    for (int i = 0; i < n; i++) { c += logits[i]; if (r < c) return i; }
    return n - 1;
}

static void print_piece(int prev, int tok) {
    const char *p = vocab[tok];
    int len = vlen[tok];
    if (len == 6 && p[0] == '<' && p[1] == '0' && p[2] == 'x') {
        char hex[3] = { p[3], p[4], 0 };
        putchar((int)strtol(hex, NULL, 16));
        return;
    }
    if (prev == 1 && p[0] == ' ') { p++; len--; }
    printf("%.*s", len, p);
}

void app_main(void) {
    ESP_LOGI(TAG, "free heap before load: %u", (unsigned)heap_caps_get_free_size(MALLOC_CAP_8BIT));

    // ---- map weights from flash ----
    const esp_partition_t *part = esp_partition_find_first(
        ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, "model");
    if (!part) { ESP_LOGE(TAG, "partition 'model' not found"); return; }
    const void *base; esp_partition_mmap_handle_t h;
    ESP_ERROR_CHECK(esp_partition_mmap(part, 0, 0x120000, ESP_PARTITION_MMAP_DATA, &base, &h));

    memcpy(&cfg, base, sizeof(Config));
    int shared = cfg.vocab_size > 0;
    if (cfg.vocab_size < 0) cfg.vocab_size = -cfg.vocab_size;
    ESP_LOGI(TAG, "dim=%d hidden=%d layers=%d heads=%d kv_heads=%d vocab=%d seq=%d",
             cfg.dim, cfg.hidden_dim, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads,
             cfg.vocab_size, cfg.seq_len);

    int hs = cfg.dim / cfg.n_heads;
    int kv_dim = (cfg.dim * cfg.n_kv_heads) / cfg.n_heads;
    int L = cfg.n_layers, D = cfg.dim, H = cfg.hidden_dim;
    const float *p = (const float *)((const char *)base + sizeof(Config));
    w.tok_emb = p;            p += (size_t)cfg.vocab_size * D;
    w.rms_att = p;            p += (size_t)L * D;
    w.wq = p;                 p += (size_t)L * D * D;
    w.wk = p;                 p += (size_t)L * D * kv_dim;
    w.wv = p;                 p += (size_t)L * D * kv_dim;
    w.wo = p;                 p += (size_t)L * D * D;
    w.rms_ffn = p;            p += (size_t)L * D;
    w.w1 = p;                 p += (size_t)L * D * H;
    w.w2 = p;                 p += (size_t)L * H * D;
    w.w3 = p;                 p += (size_t)L * D * H;
    w.rms_final = p;          p += D;
    p += (size_t)cfg.seq_len * hs;   // skip legacy freq_cis (uses ORIGINAL seq_len)
    w.wcls = shared ? w.tok_emb : p;

    cfg.seq_len = CTX_LIMIT;  // cap AFTER weight offsets are computed

    // ---- run state (heap) ----
    int T = cfg.seq_len;
    s.x = alloc(D * 4);  s.xb = alloc(D * 4);  s.xb2 = alloc(D * 4);
    s.hb = alloc(H * 4); s.hb2 = alloc(H * 4); s.q = alloc(D * 4);
    s.att = alloc((size_t)cfg.n_heads * T * 4);
    s.logits = alloc(cfg.vocab_size * 4);
    s.kc = alloc((size_t)L * T * kv_dim * 4);
    s.vc = alloc((size_t)L * T * kv_dim * 4);

    // ---- tokenizer ----
    const uint8_t *tp = tok_start + sizeof(int);   // skip max_token_length
    vocab = alloc(cfg.vocab_size * sizeof(char *));
    vlen = alloc(cfg.vocab_size * sizeof(int));
    for (int i = 0; i < cfg.vocab_size; i++) {
        tp += sizeof(float);                       // skip score
        memcpy(&vlen[i], tp, sizeof(int)); tp += sizeof(int);
        vocab[i] = (const char *)tp; tp += vlen[i];
    }

    size_t after = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    ESP_LOGI(TAG, "free heap after load: %u (KV cache = %u bytes)", (unsigned)after,
             (unsigned)((size_t)2 * L * T * kv_dim * 4));

    // ---- generate from BOS ----
    int token = 1, next, pos = 0;
    int64_t t0 = esp_timer_get_time();
    printf("\n--- output ---\n");
    while (pos < T - 1) {
        float *logits = forward(token, pos);
        next = sample(logits, cfg.vocab_size);
        pos++;
        if (next == 1) break;                      // BOS = end of story
        print_piece(token, next);
        fflush(stdout);
        token = next;
        vTaskDelay(1);                             // feed the idle-task watchdog
    }
    int64_t us = esp_timer_get_time() - t0;
    printf("\n--- end ---\n");
    ESP_LOGI(TAG, "tokens=%d time=%.2fs  %.2f tok/s  (fp32, ctx %d)",
             pos, us / 1e6, pos / (us / 1e6), T);
    ESP_LOGI(TAG, "min free heap ever: %u", (unsigned)heap_caps_get_minimum_free_size(MALLOC_CAP_8BIT));
}