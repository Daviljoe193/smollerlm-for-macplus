#!/usr/bin/env python3
"""
Run-Smol: Unified Mac Plus Ring Model Exporter
Collapses HF Loading, INT8 Quantization, Q16.16 Math, Tokenizer Export, and Sharding.
Output is 100% Big-Endian (m68k native) and Float-Free.
"""

import os
import struct
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def find_divisor(val, start_group):
    g = start_group
    while val % g != 0:
        g //= 2
    return g

def permute_reverse(w, n_heads, dim1, dim2):
    return w.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)

def quantize_q80_q16(w, group_size):
    w = w.float().reshape(-1, group_size)
    wmax = torch.abs(w).max(dim=1).values
    scale = wmax / 127.0
    quant = torch.round(w / scale[:, None]).to(torch.int8)
    
    q_bytes = quant.view(-1).numpy().tobytes()
    # Scale float to Q16.16, round, and pack as Big-Endian Int32 ('>i')
    s_q16 = (scale * 65536.0).round().to(torch.int32).numpy()
    s_bytes = struct.pack(f'>{len(s_q16)}i', *s_q16)
    
    return q_bytes, s_bytes

def serialize_q16(tensor):
    d = (tensor.detach().cpu().float() * 65536.0).round().to(torch.int32).numpy()
    return struct.pack(f'>{len(d)}i', *d)

def export_tokenizer_indexed(tokenizer, out_path):
    print(f"[*] Forging Indexed Tokenizer -> {out_path}")
    vocab_size = len(tokenizer) if hasattr(tokenizer, 'len') else tokenizer.vocab_size
    
    tokens = []
    for i in range(vocab_size):
        text = tokenizer.decode([i], clean_up_tokenization_spaces=False)
        if i in tokenizer.all_special_ids:
            text = tokenizer.convert_ids_to_tokens(i)
        tokens.append(text.encode('utf-8'))

    max_token_length = max(len(t) for t in tokens)
    
    with open(out_path, 'wb') as f:
        # Header: Magic, Vocab Size, Max Length (Big-Endian)
        f.write(struct.pack(">III", 0x544F4B4E, vocab_size, max_token_length))
        
        # Calculate TOC offset (12 bytes header + 8 bytes per TOC entry)
        current_offset = 12 + (vocab_size * 8)
        
        # Write TOC (Offset, Length)
        for b in tokens:
            f.write(struct.pack(">II", current_offset, len(b)))
            current_offset += len(b)
            
        # Write Strings Blob
        for b in tokens:
            f.write(b)

def export_mac_ring(hf_path, out_dir, target_group_size=64):
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"[*] Loading Hugging Face Model: {hf_path}")
    model = AutoModelForCausalLM.from_pretrained(hf_path)
    tok = AutoTokenizer.from_pretrained(hf_path)
    sd = model.state_dict()
    cfg = model.config
    
    export_tokenizer_indexed(tok, os.path.join(out_dir, "TOKEN.BIN"))
    
    dim = cfg.hidden_size
    n_layers = cfg.num_hidden_layers
    n_heads = cfg.num_attention_heads
    n_kv_heads = cfg.num_key_value_heads
    vocab_size = cfg.vocab_size
    hidden_dim = cfg.intermediate_size
    max_seq_len = getattr(cfg, 'max_position_embeddings', 256)
    head_dim = getattr(cfg, 'head_dim', dim // n_heads)
    
    att_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim
    
    group_size = find_divisor(dim, target_group_size)
    group_size = find_divisor(att_dim, group_size)
    print(f"[*] Active Architecture: Dim={dim}, Layers={n_layers}, Heads={n_heads}, GroupSize={group_size}")

    print("\n[*] Forging Mac Plus Slave Shards (1 Layer Per File)...")
    for i in range(n_layers):
        slave_path = os.path.join(out_dir, f"LAYER_{i+1:02d}.BIN")
        with open(slave_path, 'wb') as f:
            f.write(struct.pack('>iiii', dim, hidden_dim, att_dim, kv_dim))
            f.write(serialize_q16(sd[f'model.layers.{i}.input_layernorm.weight']))
            f.write(serialize_q16(sd[f'model.layers.{i}.post_attention_layernorm.weight']))
            
            tensors = [
                permute_reverse(sd[f'model.layers.{i}.self_attn.q_proj.weight'], n_heads, att_dim, dim),
                permute_reverse(sd[f'model.layers.{i}.self_attn.k_proj.weight'], n_kv_heads, kv_dim, dim),
                sd[f'model.layers.{i}.self_attn.v_proj.weight'],
                sd[f'model.layers.{i}.self_attn.o_proj.weight'],
                sd[f'model.layers.{i}.mlp.gate_proj.weight'],
                sd[f'model.layers.{i}.mlp.down_proj.weight'],
                sd[f'model.layers.{i}.mlp.up_proj.weight']
            ]
            for t in tensors:
                q_bytes, s_bytes = quantize_q80_q16(t, group_size)
                f.write(q_bytes)
                f.write(s_bytes)

    print("\n[*] Forging Mac Plus Master Shards (Split Vocab Volumes)...")
    w_emb = sd['model.embed_tokens.weight']
    w_lm = sd.get('lm_head.weight', w_emb)
    shared_classifier = torch.equal(w_emb, w_lm)
    
    q_lm, s_lm = quantize_q80_q16(w_lm, group_size)
    
    half_vocab = vocab_size // 2
    half_numel = half_vocab * dim
    s_half_size = (half_numel // group_size) * 4 
    
    for vol in [1, 2]:
        vol_path = os.path.join(out_dir, f"MASTER_VOL{vol}.BIN")
        with open(vol_path, 'wb') as f:
            if vol == 1:
                f.write(struct.pack('>Ii', 0x616b3432, 2)) 
                f.write(struct.pack('>iiiiiiii', dim, hidden_dim, n_layers, n_heads, 
                                     n_kv_heads, half_vocab, max_seq_len, head_dim))
                f.write(struct.pack('B', int(shared_classifier)))
                f.write(struct.pack('>i', group_size))
                pad = 256 - f.tell()
                f.write(b'\0' * pad)
                f.write(serialize_q16(sd['model.norm.weight']))
                f.write(q_lm[:half_numel])
                f.write(s_lm[:s_half_size])
            else:
                f.write(q_lm[half_numel:])
                f.write(s_lm[s_half_size:])
                
    print("\n[+] Success! Ready for Mac Plus ingestion.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf", type=str, required=True)
    parser.add_argument("-o", "--out", type=str, default="mac_ring_data")
    args = parser.parse_args()
    export_mac_ring(args.hf, args.out)
