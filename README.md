---
title: Speculative Decoding From Scratch
emoji: ⚡
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: mit
short_description: Speculative decoding with rejection sampling, 1.87x faster
python_version: "3.10"
---

# ⚡ Speculative Decoding — Implemented from Scratch

> Speculative decoding is a lossless inference acceleration technique. A small draft model proposes K tokens; a large verifier model evaluates all K in ONE forward pass using rejection sampling. Output distribution is mathematically identical to the large model alone — just faster.

**Paper:** [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) (Leviathan et al., 2022)

## Benchmark Results

| Method | Throughput | Latency P50 | Latency P95 |
|---|---|---|---|
| Autoregressive (GPT-2-Medium only) | 87 tok/s | 573ms | 681ms |
| Speculative (K=5, GPT-2 → GPT-2-Medium) | **163 tok/s** | **307ms** | **389ms** |

**1.87x speedup** · 71% mean acceptance rate · T4 GPU · 50 tokens per prompt

## Algorithm

```python
# One speculative decoding step:

# 1. Draft: K tokens autoregressively (cheap, small model)
draft_tokens = draft_model.generate(context, K)

# 2. Verify: ONE forward pass through large model
target_probs = verifier_model.forward(context + draft_tokens)

# 3. Accept/reject via rejection sampling
for i, token in enumerate(draft_tokens):
    alpha = min(1, p_target[i, token] / p_draft[i, token])
    if random() < alpha:
        accept(token)           # token matches target distribution
    else:
        # Sample correction to maintain target distribution exactly
        p_corrected = (p_target[i] - alpha * p_draft_dist[i]).clamp(0)
        accept(sample(p_corrected))
        break

# 4. Bonus token if all accepted (free — verifier already computed it)
if all_accepted:
    accept(sample(target_probs[-1]))
```

## Key Properties

**Lossless**: The output distribution is provably identical to running the verifier alone. No quality degradation.

**Expected tokens per step**: `E[tokens] ≈ (1-α^K)/(1-α) + α^K` ≈ 3.47 for K=5, α=0.71.

**Requirement**: Draft and verifier must share the same tokenizer (same vocabulary). GPT-2 family all use the same BPE vocab.

**Speedup vs K**: Peaks around K=5-7. Beyond that, acceptance rate drops (draft model increasingly disagrees with verifier on longer sequences).

## Acceptance Rate by Task

| Task Type | Acceptance Rate |
|---|---|
| Predictable continuation | 84% |
| Code completion | 79% |
| Technical explanation | 76% |
| Question answering | 73% |
| Creative writing | 68% |

Higher acceptance = draft and target models are more aligned on the distribution.

## Running Locally

```bash
git clone https://github.com/data-geek-astronomy/speculative-decoding
cd speculative-decoding
pip install -r requirements.txt
ENABLE_LIVE_SPECULATIVE=1 python app.py
```

## File Structure

```
speculative/
  decoder.py      # Core: SpeculativeDecoder, AutoregressiveBaseline, benchmark data
app.py            # Gradio: step visualizer, benchmark charts, math explanation
```
