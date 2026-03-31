/*
 * Run-Smol: Apple Mac Plus 31-Node Ring Edition
 * Toolchain: Retro68 (m68k-apple-macos-gcc)
 * Architecture: INT8 x Q15 Block Floating Point
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <Devices.h>
#include <Serial.h>
#include <Events.h>
#include <OSUtils.h>

/* --- CONFIGURATION (Matched to SmollerLM2-10M) --- */
#define Q_SCALE     65536
#define Q_HALF      32768
#define GROUP_SIZE  4
#define DIM         60
#define MAX_SEQ     256
#define CHUNK_ROWS  64
#define N_HEADS     9
#define HEAD_DIM    64

typedef struct {
    int32_t pos;
    int32_t token;
    int32_t is_prefill;
    int32_t x[DIM];
} Packet;

typedef struct {
    float temp;
    float top_p;
    int   top_k;
    float min_p;
    float rep_pen;
    int   rep_last_n;
} SamplerArgs;

/* ========================================================================== */
/* ERROR HANDLING                                                             */
/* ========================================================================== */

void fatal_error(const char* msg) {
    printf("\n[!] FATAL ERROR: %s\n", msg);
    printf("Press RETURN to exit...\n");
    fflush(stdout);
    int c;
    while ((c = getchar()) != '\n' && c != EOF) { SystemTask(); }
    getchar(); 
    exit(1);
}

/* ========================================================================== */
/* RING NETWORK - MAC OS SCC SERIAL MANAGER (SINGLE PORT VIRTUAL-MIDI)        */
/* ========================================================================== */

static char g_rx_buf[8192];
static short g_a_in = 0, g_a_out = 0;

void init_ring_serial(int is_master) {
    OSErr err;
    printf("Initializing Mac OS SCC Serial Ring (Modem Port)...\n");
    
    /* Open ONLY the Modem Port. This avoids AppleTalk conflicts on Port B. */
    err = OpenDriver("\p.AIn",  &g_a_in);   if (err) fatal_error("OpenDriver .AIn failed");
    err = OpenDriver("\p.AOut", &g_a_out);  if (err) fatal_error("OpenDriver .AOut failed");

    SerSetBuf(g_a_in, g_rx_buf, (short)sizeof(g_rx_buf));
    
    /* baud19200 is natively supported by Mac Plus ROM without an external clock */
    short cfg = baud19200 | data8 | noParity | stop10;
    
    SerReset(g_a_in, cfg);
    SerReset(g_a_out, cfg);

    /* Disable all hardware handshaking */
    SerShk shk;
    memset(&shk, 0, sizeof(SerShk));
    SerHShake(g_a_out, &shk);
}

void ring_read(void *buf, size_t size) {
    uint8_t *ptr = (uint8_t *)buf;
    long remaining = (long)size;
    while (remaining > 0) {
        long avail = 0;
        SerGetBuf(g_a_in, &avail);
        if (avail <= 0) {
            SystemTask(); // Let Mac OS service the SCC interrupts!
            continue; 
        }
        long chunk = (avail < remaining) ? avail : remaining;
        FSRead(g_a_in, &chunk, (Ptr)ptr);
        ptr += chunk;
        remaining -= chunk;
    }
}

void ring_write(void *buf, size_t size) {
    long count = (long)size;
    FSWrite(g_a_out, &count, (Ptr)buf);
}

/* ========================================================================== */
/* NODE AUTO-DETECTION                                                        */
/* ========================================================================== */

int get_node_id() {
    FILE* f;
    
    // Check for Master Vol 1
    f = fopen("MASTER_VOL1.BIN", "rb");
    if (f) {
        fclose(f);
        return 0; // Master Node
    }
    
    // Check for Layer Shards
    for (int i = 1; i <= 30; i++) {
        char filename[32];
        sprintf(filename, "LAYER_%02d.BIN", i);
        f = fopen(filename, "rb");
        if (f) {
            fclose(f);
            return i; // Slave Node
        }
    }
    
    fatal_error("Cannot auto-detect Node ID! Missing MASTER_VOL1.BIN or LAYER_XX.BIN in this folder.");
    return -1;
}

/* ========================================================================== */
/* Q16.16 MATH KERNELS                                                        */
/* ========================================================================== */

// Opus Fix: 64-bit precision square root for tiny Q32.32 values
uint32_t isqrt_u64(uint64_t val) {
    if (val == 0) return 0;
    uint64_t rx = val, ry = 1;
    while (rx > ry) {
        rx = (rx + ry) >> 1;
        ry = val / rx;
    }
    return (uint32_t)rx;
}

// Opus Fix: 64-bit precision RMSNorm accumulation
void rmsnorm_q16(int32_t* o, int32_t* x, int32_t* weight, int size) {
    int64_t ss = 0;
    for (int i = 0; i < size; i++) {
        ss += (int64_t)x[i] * x[i];          /* Q16.16 × Q16.16 = Q32.32 */
    }
    int64_t mean = ss / size + 42950;         /* +eps: 1e-5 in Q32.32 ≈ 42950 */

    uint32_t s = isqrt_u64((uint64_t)mean);   /* sqrt(Q32.32) = Q16.16 */
    if (s == 0) s = 1;

    int32_t inv_s = (int32_t)(4294967296ULL / (uint64_t)s);  /* 2^32 / Q16.16 = Q16.16 */

    for (int i = 0; i < size; i++) {
        int32_t norm = ((int64_t)x[i] * inv_s + Q_HALF) >> 16;
        o[i] = ((int64_t)norm * weight[i] + Q_HALF) >> 16;
    }
}

// Fast integer Taylor approx with 64-bit casts to prevent overflow!
int32_t exp_neg_q16(int32_t val) {
    if (val < 0) val = 0;
    if (val > 1048576) return 0; // > 16.0
    
    int64_t x_scaled = ((int64_t)val * 94548) >> 16; // 94548 = log2(e) in Q16
    int32_t int_part = x_scaled >> 16;
    int32_t frac_part = x_scaled & 65535;
    
    int32_t term1 = ((int64_t)frac_part * 45426) >> 16;
    int32_t frac_sq = ((int64_t)frac_part * frac_part) >> 16;
    int32_t term2 = ((int64_t)frac_sq * 15745) >> 16;
    int32_t approx = 65536 - term1 + term2;
    
    return approx >> int_part;
}

int32_t silu_q16(int32_t x) {
    if (x >= 0) {
        int32_t e_neg = exp_neg_q16(x);
        // Sigmoid = 1 / (1 + exp(-x)) -> scaled to Q16
        int32_t sig = 4294967296LL / (65536 + e_neg);
        return ((int64_t)x * sig) >> 16;
    } else {
        int32_t e_neg = exp_neg_q16(-x);
        // Sigmoid = exp(x) / (1 + exp(x))
        int32_t sig = ((int64_t)e_neg * 65536) / (65536 + e_neg);
        return ((int64_t)x * sig) >> 16;
    }
}

void matmul_int8_q15(int32_t* xout, int32_t* x, const int8_t* w, const int32_t* scales, int n, int d, int16_t* xq_buf) {
    int32_t amax = 0;
    for(int i = 0; i < n; i++) {
        int32_t a = (x[i] < 0) ? -x[i] : x[i];
        if (a > amax) amax = a;
    }
    if (amax == 0) { memset(xout, 0, d * sizeof(int32_t)); return; }

    int shift = 0; int32_t temp = amax;
    while (temp < 16384 && temp > 0) { temp <<= 1; shift++; }
    while (temp >= 32768) { temp >>= 1; shift--; }
    
    for(int i = 0; i < n; i++) {
        if (shift > 0) xq_buf[i] = (int16_t)(x[i] << shift);
        else           xq_buf[i] = (int16_t)(x[i] >> (-shift));
    }
    
    int groups = n / GROUP_SIZE;
    for(int i = 0; i < d; i++) {
        int32_t total_q16 = 0;
        int row_off = i * n;
        for(int g = 0; g < groups; g++) {
            register int32_t sum = 0;
            const int8_t* wg = &w[row_off + g * GROUP_SIZE];
            int16_t* xg = &xq_buf[g * GROUP_SIZE];
            // Safe: since GROUP_SIZE is exactly 4, this unrolled loop runs exactly once.
            for(int k = 0; k < GROUP_SIZE; k += 4) {
                sum += (int32_t)wg[k+0] * xg[k+0];
                sum += (int32_t)wg[k+1] * xg[k+1];
                sum += (int32_t)wg[k+2] * xg[k+2];
                sum += (int32_t)wg[k+3] * xg[k+3];
            }
            int64_t scaled = (int64_t)sum * scales[i * groups + g];
            total_q16 += (int32_t)(scaled >> 16);
        }
        if (shift > 0) xout[i] = total_q16 >> shift;
        else           xout[i] = total_q16 << (-shift);
    }
}

/* ========================================================================== */
/* MASTER LOGIC: DATA & TOKENIZER                                             */
/* ========================================================================== */

FILE *f_vol1, *f_vol2;
int32_t vocab_size, half_vocab;
int8_t  q_chunk[CHUNK_ROWS * DIM];
int32_t s_chunk[CHUNK_ROWS * (DIM / GROUP_SIZE)]; 
int16_t master_xq_buf[DIM];
int32_t rms_final[DIM];

uint8_t *tok_buffer;
uint32_t *tok_offsets;
uint32_t *tok_lengths;

void init_master() {
    printf("Mounting SCSI Volumes...\n");
    f_vol1 = fopen("MASTER_VOL1.BIN", "rb");
    f_vol2 = fopen("MASTER_VOL2.BIN", "rb");
    FILE *f_tok = fopen("TOKEN.BIN", "rb");
    
    if (!f_vol1 || !f_vol2 || !f_tok) fatal_error("Missing BIN files.");
    
    fseek(f_tok, 0, SEEK_END);
    long tok_sz = ftell(f_tok);
    fseek(f_tok, 0, SEEK_SET);
    
    tok_buffer = malloc(tok_sz);
    if (!tok_buffer) fatal_error("OOM allocating Tokenizer");
    fread(tok_buffer, 1, tok_sz, f_tok);
    fclose(f_tok);
    
    uint32_t *hdr = (uint32_t*)tok_buffer;
    vocab_size = hdr[1];
    half_vocab = vocab_size / 2;
    
    tok_offsets = malloc(vocab_size * 4);
    tok_lengths = malloc(vocab_size * 4);
    
    uint32_t *toc = (uint32_t*)(tok_buffer + 12);
    for(int i = 0; i < vocab_size; i++) {
        tok_offsets[i] = toc[i*2];
        tok_lengths[i] = toc[i*2 + 1];
    }
    
    fseek(f_vol1, 256, SEEK_SET);
    fread(rms_final, 4, DIM, f_vol1);
    printf("Master Init OK. Vocab: %d\n", (int)vocab_size);
}

void print_token(int32_t token) {
    if (token < 0 || token >= vocab_size || tok_lengths[token] == 0) return;
    uint32_t off = tok_offsets[token];
    uint32_t len = tok_lengths[token];
    for(uint32_t i = 0; i < len; i++) putchar(tok_buffer[off + i]);
    fflush(stdout);
}

int encode_string(const char* text, int* tokens) {
    int n_tok = 0;
    const char* c = text;
    while (*c != '\0') {
        int best_id = -1, best_len = 0;
        // Greedy longest-prefix match (Fallback BPE)
        for (int i = 0; i < vocab_size; i++) {
            int tlen = tok_lengths[i];
            if (tlen > best_len && strncmp(c, (char*)(tok_buffer + tok_offsets[i]), tlen) == 0) {
                best_id = i; best_len = tlen;
            }
        }
        if (best_id != -1) {
            tokens[n_tok++] = best_id;
            c += best_len;
        } else {
            tokens[n_tok++] = (unsigned char)(*c); // ASCII fallback
            c++;
        }
    }
    return n_tok;
}

void get_embedding(int32_t token, int32_t* out_x) {
    FILE* f;
    int32_t local_tok;
    long q_start, s_start;
    
    // Vol 2 offsets corrected (no 256 byte header, no DIM*4 byte rms_final)
    if (token < half_vocab) {
        f = f_vol1;
        local_tok = token;
        q_start = 256 + (DIM * 4);
        s_start = q_start + ((long)half_vocab * DIM);
    } else {
        f = f_vol2;
        local_tok = token - half_vocab;
        q_start = 0;
        s_start = ((long)half_vocab * DIM);
    }
    
    int8_t w_row[DIM];
    int32_t s_row[DIM / GROUP_SIZE];
    
    fseek(f, q_start + (local_tok * DIM), SEEK_SET);
    fread(w_row, 1, DIM, f);
    fseek(f, s_start + (local_tok * (DIM / GROUP_SIZE) * 4), SEEK_SET);
    fread(s_row, 4, DIM / GROUP_SIZE, f);
    
    // Removed the destructive >> 16
    for(int i = 0; i < DIM; i++) {
        out_x[i] = (int32_t)((int64_t)w_row[i] * s_row[i / GROUP_SIZE]);
    }
}

/* ========================================================================== */
/* STREAMING TOP-K SAMPLER                                                    */
/* ========================================================================== */

int32_t sample_streamed(int32_t* x, SamplerArgs* args, int* history, int hist_len) {
    int32_t top_logits[64];
    int32_t top_ids[64];
    for(int i=0; i<64; i++) top_logits[i] = -2147483647;
    
    int32_t amax = 0;
    for(int i = 0; i < DIM; i++) {
        int32_t a = (x[i] < 0) ? -x[i] : x[i];
        if (a > amax) amax = a;
    }
    int shift = 0; int32_t temp = amax;
    while (temp < 16384 && temp > 0) { temp <<= 1; shift++; }
    while (temp >= 32768) { temp >>= 1; shift--; }
    for(int i = 0; i < DIM; i++) {
        if (shift > 0) master_xq_buf[i] = (int16_t)(x[i] << shift);
        else           master_xq_buf[i] = (int16_t)(x[i] >> (-shift));
    }

    // Streaming Disk Grind -> Top 64
    for(int vol = 1; vol <= 2; vol++) {
        FILE* f;
        int32_t base_token;
        long q_start, s_start;
        
        if (vol == 1) {
            f = f_vol1;
            base_token = 0;
            q_start = 256 + (DIM * 4);
            s_start = q_start + ((long)half_vocab * DIM);
        } else {
            f = f_vol2;
            base_token = half_vocab;
            q_start = 0;
            s_start = ((long)half_vocab * DIM);
        }
        
        fseek(f, q_start, SEEK_SET); 
        
        int chunks = half_vocab / CHUNK_ROWS;
        for(int c = 0; c < chunks; c++) {
            fread(q_chunk, 1, CHUNK_ROWS * DIM, f);
            long q_pos = ftell(f);
            fseek(f, s_start + (c * CHUNK_ROWS * (DIM/GROUP_SIZE) * 4), SEEK_SET);
            fread(s_chunk, 4, CHUNK_ROWS * (DIM/GROUP_SIZE), f);
            fseek(f, q_pos, SEEK_SET); 
            
            for(int r = 0; r < CHUNK_ROWS; r++) {
                int32_t total_q16 = 0;
                int8_t* w_row = &q_chunk[r * DIM];
                int32_t* s_row = &s_chunk[r * (DIM/GROUP_SIZE)];
                
                for(int g = 0; g < (DIM/GROUP_SIZE); g++) {
                    int32_t sum = 0;
                    for(int k = 0; k < GROUP_SIZE; k++) {
                        sum += (int32_t)w_row[g*GROUP_SIZE + k] * master_xq_buf[g*GROUP_SIZE + k];
                    }
                    total_q16 += (int32_t)(((int64_t)sum * s_row[g]) >> 16);
                }
                
                int32_t logit = (shift > 0) ? (total_q16 >> shift) : (total_q16 << (-shift));
                int32_t tok_id = base_token + (c * CHUNK_ROWS) + r;
                
                // Apply Repetition Penalty on the fly
                if (args->rep_pen != 1.0f) {
                    for(int h = 0; h < hist_len; h++) {
                        if (history[h] == tok_id) {
                            if (logit > 0) logit = (int32_t)((float)logit / args->rep_pen);
                            else           logit = (int32_t)((float)logit * args->rep_pen);
                            break;
                        }
                    }
                }
                
                // Insertion sort into Top-K
                if (logit > top_logits[63]) {
                    int j = 62;
                    while (j >= 0 && top_logits[j] < logit) {
                        top_logits[j+1] = top_logits[j];
                        top_ids[j+1] = top_ids[j];
                        j--;
                    }
                    top_logits[j+1] = logit;
                    top_ids[j+1] = tok_id;
                }
            }
            SystemTask(); // Keep OS happy
        }
    }
    
    // 2. Float Sampling on Top 64
    if (args->temp <= 0.0f) return top_ids[0]; // Greedy
    
    int k_cap = (args->top_k < 64) ? args->top_k : 64;
    float probs[64];
    float max_l = (float)top_logits[0] / 65536.0f;
    float sum_p = 0.0f;
    
    for(int i = 0; i < k_cap; i++) {
        float f_log = (float)top_logits[i] / 65536.0f;
        probs[i] = expf((f_log - max_l) / args->temp);
        sum_p += probs[i];
    }
    for(int i = 0; i < k_cap; i++) probs[i] /= sum_p;
    
    // Min-P
    int eff_k = k_cap;
    float min_thresh = probs[0] * args->min_p;
    for(int i = 0; i < k_cap; i++) {
        if (probs[i] < min_thresh) { eff_k = i; break; }
    }
    if (eff_k == 0) eff_k = 1;
    
    // Top-P
    if (args->top_p > 0.0f && args->top_p < 1.0f) {
        float cdf = 0.0f;
        for(int i = 0; i < eff_k; i++) {
            cdf += probs[i];
            if (cdf >= args->top_p) { eff_k = i + 1; break; }
        }
    }
    
    float rsum = 0.0f;
    for(int i = 0; i < eff_k; i++) rsum += probs[i];
    float coin = ((float)rand() / (float)RAND_MAX) * rsum;
    
    float cdf = 0.0f;
    for(int i = 0; i < eff_k; i++) {
        cdf += probs[i];
        if (coin < cdf) return top_ids[i];
    }
    return top_ids[eff_k - 1];
}

/* ========================================================================== */
/* MASTER LOOP: TUI & CHATML                                                  */
/* ========================================================================== */

void test_ring_connection(int node_id) {
    uint8_t token = 0;
    
    if (node_id == 0) {
        printf("\n[Master] Press RETURN to test the 31-Node Ring...\n");
        fflush(stdout);
        
        int c; 
        while ((c = getchar()) != '\n' && c != EOF) { SystemTask(); }
        
        printf("[Master] Sending Ping (0xAA)... ");
        fflush(stdout);
        
        token = 0xAA;
        ring_write(&token, 1);
        ring_read(&token, 1); // Wait for it to traverse 30 Macs
        
        if (token == 0xAA) {
            printf("SUCCESS!\n[Master] Sending GO (0x55)...\n");
            token = 0x55;
            ring_write(&token, 1);
            ring_read(&token, 1); // Wait for GO to traverse
        } else {
            char err[64];
            sprintf(err, "Ring broken! Expected 0xAA, got 0x%02X", token);
            fatal_error(err);
        }
    } else {
        printf("Node %d: Waiting for Ring Ping...\n", node_id);
        while (1) {
            ring_read(&token, 1);
            if (token == 0xAA) {
                printf("  -> Received Ping! Forwarding...\n");
                ring_write(&token, 1);
            } else if (token == 0x55) {
                printf("  -> Received GO! Entering Main AI Loop.\n");
                ring_write(&token, 1);
                break;
            }
            SystemTask();
        }
    }
}

void master_loop() {
    init_master();
    
    SamplerArgs samp = { 0.8f, 0.9f, 40, 0.05f, 1.15f, 64 };
    int history[MAX_SEQ];
    int hist_len = 0;
    int pos = 0;
    
    int prompt_tokens[1024];
    char input_buf[512];
    
    printf("\n--- BOOTING AI ---\n");
    
    int n_tok = 0;
    n_tok += encode_string("<|im_start|>", &prompt_tokens[n_tok]); 
    n_tok += encode_string("system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n", &prompt_tokens[n_tok]);
    
    printf("Pre-filling System Prompt (%d tokens)...\n", n_tok);
    Packet pkt;
    for(int i = 0; i < n_tok; i++) {
        printf("  [%02d/%02d] Traversing ring with token %d... ", i+1, n_tok, prompt_tokens[i]);
        fflush(stdout);
        
        pkt.pos = pos; pkt.token = prompt_tokens[i]; pkt.is_prefill = 1;
        get_embedding(pkt.token, pkt.x);
        
        ring_write(&pkt, sizeof(Packet)); 
        ring_read(&pkt, sizeof(Packet));
        
        history[pos++] = pkt.token;
        printf("Done.\n");
    }
    
    printf("\nType /bye to quit, /clear to reset context.\n");
    
    while(1) {
        printf("\n>>> "); fflush(stdout);
        
        // Use native C fgets to preserve backspace/editing functionality!
        if(!fgets(input_buf, sizeof(input_buf), stdin)) break;
        input_buf[strcspn(input_buf, "\n")] = 0;
        
        if (strncmp(input_buf, "/bye", 4) == 0) break;
        if (strncmp(input_buf, "/clear", 6) == 0) { pos = 0; hist_len = 0; continue; }
        
        n_tok = 0;
        n_tok += encode_string("<|im_start|>user\n", &prompt_tokens[n_tok]);
        n_tok += encode_string(input_buf, &prompt_tokens[n_tok]);
        n_tok += encode_string("<|im_end|>\n<|im_start|>assistant\n", &prompt_tokens[n_tok]);
        
        if (pos + n_tok >= MAX_SEQ) { printf("\n[Context Full!]\n"); pos = 0; continue; }
        
        printf("Pre-filling User Prompt (%d tokens)...\n", n_tok);
        for(int i = 0; i < n_tok; i++) {
            printf("  [%02d/%02d] Traversing ring... ", i+1, n_tok);
            fflush(stdout);
            
            pkt.pos = pos; pkt.token = prompt_tokens[i]; pkt.is_prefill = 1;
            get_embedding(pkt.token, pkt.x);
            ring_write(&pkt, sizeof(Packet)); 
            ring_read(&pkt, sizeof(Packet));
            
            history[pos++] = pkt.token;
            printf("Done.\n");
        }
        
        int token = prompt_tokens[n_tok - 1];
        
        // --- GENERATION PHASE ---
        while (pos < MAX_SEQ) {
            pkt.pos = pos; pkt.token = token; pkt.is_prefill = 0;
            get_embedding(pkt.token, pkt.x);
            
            ring_write(&pkt, sizeof(Packet)); 
            ring_read(&pkt, sizeof(Packet));
            
            int32_t xb[DIM];
            rmsnorm_q16(xb, pkt.x, rms_final, DIM);
            
            int search_len = (pos < samp.rep_last_n) ? pos : samp.rep_last_n;
            int* hist_ptr = (pos < samp.rep_last_n) ? history : &history[pos - samp.rep_last_n];
            
            token = sample_streamed(xb, &samp, hist_ptr, search_len);
            history[pos++] = token;
            print_token(token);
            
            if (token == 2 || token == 0) break; // EOS
        }
        printf("\n");
    }
}

/* ========================================================================== */
/* SLAVE LOGIC (NODES 1-30)                                                   */
/* ========================================================================== */

int32_t rope_cos[MAX_SEQ * (HEAD_DIM / 2)];
int32_t rope_sin[MAX_SEQ * (HEAD_DIM / 2)];

void init_rope() {
    for (int pos = 0; pos < MAX_SEQ; pos++) {
        for (int i = 0; i < HEAD_DIM; i += 2) {
            float freq = 1.0f / powf(100000.0f, (float)i / (float)HEAD_DIM);
            float val = pos * freq;
            rope_cos[pos * (HEAD_DIM/2) + (i/2)] = (int32_t)(cosf(val) * 65536.0f);
            rope_sin[pos * (HEAD_DIM/2) + (i/2)] = (int32_t)(sinf(val) * 65536.0f);
        }
    }
}

void slave_loop(int layer_id) {
    char filename[32];
    sprintf(filename, "LAYER_%02d.BIN", layer_id);
    printf("Loading %s into RAM...\n", filename);
    
    FILE* f = fopen(filename, "rb");
    if (!f) { char err[64]; sprintf(err, "Missing %s", filename); fatal_error(err); }
    
    int32_t f_dim, f_hidden, f_att, f_kv;
    fread(&f_dim, 4, 1, f); fread(&f_hidden, 4, 1, f);
    fread(&f_att, 4, 1, f); fread(&f_kv, 4, 1, f);
    int n_kv_heads = f_kv / HEAD_DIM;
    
    fseek(f, 0, SEEK_END); long size = ftell(f) - 16; fseek(f, 16, SEEK_SET);
    uint8_t* layer_mem = malloc(size);
    if (!layer_mem) fatal_error("OOM loading layer");
    fread(layer_mem, 1, size, f);
    fclose(f);
    
    int32_t* rms_att = (int32_t*)layer_mem; 
    int32_t* rms_ffn = rms_att + DIM;
    uint8_t* ptr = (uint8_t*)(rms_ffn + DIM);
    
    #define ASSIGN_PTRS(q, s, d1, d2) \
        int8_t* q = (int8_t*)ptr; ptr += (d1 * d2); \
        int32_t* s = (int32_t*)ptr; ptr += ((d1 * d2)/GROUP_SIZE)*4;

    ASSIGN_PTRS(wq_q, wq_s, f_att, f_dim);
    ASSIGN_PTRS(wk_q, wk_s, f_kv, f_dim);
    ASSIGN_PTRS(wv_q, wv_s, f_kv, f_dim);
    ASSIGN_PTRS(wo_q, wo_s, f_dim, f_att);
    ASSIGN_PTRS(w1_q, w1_s, f_hidden, f_dim);
    ASSIGN_PTRS(w2_q, w2_s, f_dim, f_hidden);
    ASSIGN_PTRS(w3_q, w3_s, f_hidden, f_dim);
    
    int32_t* key_cache   = malloc(MAX_SEQ * f_kv * sizeof(int32_t));
    int32_t* value_cache = malloc(MAX_SEQ * f_kv * sizeof(int32_t));
    int32_t* q = malloc(f_att * sizeof(int32_t));
    int32_t* att_out = malloc(f_att * sizeof(int32_t));
    int32_t* hb = malloc(f_hidden * sizeof(int32_t));
    int32_t* hb2 = malloc(f_hidden * sizeof(int32_t));
    int max_buf = (f_hidden > DIM) ? f_hidden : DIM;
    int16_t* sq_buf = malloc(max_buf * sizeof(int16_t));
    
    if (!key_cache || !value_cache || !q || !att_out || !hb || !hb2 || !sq_buf) fatal_error("OOM on buffers");
    init_rope();
    int32_t attn_scaler = (int32_t)((1.0f / sqrtf((float)HEAD_DIM)) * 65536.0f);
    
    printf("Node %d Ready. Awaiting Packets...\n", layer_id);
    
    Packet pkt;
    int32_t xb[DIM], xb2[DIM];
    
    while(1) {
        ring_read(&pkt, sizeof(Packet));
        int pos = pkt.pos; if (pos >= MAX_SEQ) pos = MAX_SEQ - 1;
        
        rmsnorm_q16(xb, pkt.x, rms_att, DIM);
        int32_t* k = &key_cache[pos * f_kv];
        int32_t* v = &value_cache[pos * f_kv];
        
        matmul_int8_q15(q, xb, wq_q, wq_s, f_dim, f_att, sq_buf);
        matmul_int8_q15(k, xb, wk_q, wk_s, f_dim, f_kv,  sq_buf);
        matmul_int8_q15(v, xb, wv_q, wv_s, f_dim, f_kv,  sq_buf);
        
        for (int i = 0; i < f_att; i += 2) {
            int cidx = pos * (HEAD_DIM / 2) + (i % HEAD_DIM) / 2;
            int32_t fcr = rope_cos[cidx], fci = rope_sin[cidx];
            int32_t q0 = q[i], q1 = q[i+1];
            q[i]   = ((int64_t)q0 * fcr - (int64_t)q1 * fci + Q_HALF) >> 16;
            q[i+1] = ((int64_t)q0 * fci + (int64_t)q1 * fcr + Q_HALF) >> 16;
            if (i < f_kv) {
                int32_t k0 = k[i], k1 = k[i+1];
                k[i]   = ((int64_t)k0 * fcr - (int64_t)k1 * fci + Q_HALF) >> 16;
                k[i+1] = ((int64_t)k0 * fci + (int64_t)k1 * fcr + Q_HALF) >> 16;
            }
        }
        
        int kv_mul = N_HEADS / n_kv_heads;
        for (int h = 0; h < N_HEADS; h++) {
            int32_t* q_head = &q[h * HEAD_DIM];
            int kv_h = h / kv_mul;
            float att_scores[MAX_SEQ];
            float max_score = -1e20f;
            
            for (int t = 0; t <= pos; t++) {
                int32_t* k_head = &key_cache[t * f_kv + kv_h * HEAD_DIM];
                int32_t score = 0;
                for (int i = 0; i < HEAD_DIM; i++) score += ((int64_t)q_head[i] * k_head[i]) >> 16;
                score = ((int64_t)score * attn_scaler) >> 16;
                float f_score = (float)score / 65536.0f;
                att_scores[t] = f_score;
                if (f_score > max_score) max_score = f_score;
            }
            float sum_exp = 0.0f;
            for (int t = 0; t <= pos; t++) {
                att_scores[t] = expf(att_scores[t] - max_score);
                sum_exp += att_scores[t];
            }
            int32_t* out_head = &att_out[h * HEAD_DIM];
            memset(out_head, 0, HEAD_DIM * sizeof(int32_t));
            
            for (int t = 0; t <= pos; t++) {
                int32_t a_val = (int32_t)((att_scores[t] / sum_exp) * 65536.0f);
                int32_t* v_head = &value_cache[t * f_kv + kv_h * HEAD_DIM];
                for (int i = 0; i < HEAD_DIM; i++) out_head[i] += ((int64_t)a_val * v_head[i]) >> 16;
            }
        }
        
        matmul_int8_q15(xb2, att_out, wo_q, wo_s, f_att, f_dim, sq_buf);
        for(int i = 0; i < DIM; i++) pkt.x[i] += xb2[i];
        
        rmsnorm_q16(xb, pkt.x, rms_ffn, DIM);
        matmul_int8_q15(hb, xb, w1_q, w1_s, f_dim, f_hidden, sq_buf);
        matmul_int8_q15(hb2, xb, w3_q, w3_s, f_dim, f_hidden, sq_buf);
        
        // Final Fix: 64-bit casting to prevent silent integer overflow
        for(int i = 0; i < f_hidden; i++) {
            hb[i] = (int32_t)(((int64_t)silu_q16(hb[i]) * hb2[i] + Q_HALF) >> 16);
        }
        
        matmul_int8_q15(xb, hb, w2_q, w2_s, f_hidden, f_dim, sq_buf);
        for(int i = 0; i < DIM; i++) pkt.x[i] += xb[i];
        
        ring_write(&pkt, sizeof(Packet));
        SystemTask(); 
    }
}

/* ========================================================================== */
/* ENTRY POINT                                                                */
/* ========================================================================== */

int main(void) {
    srand(time(NULL));
    printf("\n==================================\n");
    printf("  SMOL-MAC: MARCHINTOSH RING 31   \n");
    printf("==================================\n\n");
    
    int node_id = get_node_id();
    printf("Auto-Detected Node ID: %d\n", node_id);
    
    init_ring_serial(node_id == 0);
    test_ring_connection(node_id);
    
    if (node_id == 0) master_loop();
    else slave_loop(node_id);

    return 0;
}
