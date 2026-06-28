"""
Speculative Decoding — Interactive Demo
========================================
Visualize token-by-token acceptance/rejection, speedup charts,
and the mathematical intuition behind speculative decoding.

Author: Aravind Kumar Nalukurthi
"""

import gradio as gr
import os
import json
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

from speculative.decoder import get_precomputed_benchmark_results

ENABLE_LIVE = os.getenv("ENABLE_LIVE_SPECULATIVE", "0") == "1"

CSS = """
body, .gradio-container { background: #0a0d14 !important; }
.card { background: rgba(99,102,241,0.07); border: 1px solid rgba(99,102,241,0.3); border-radius: 12px; padding: 18px; margin: 8px 0; }
.accepted { color: #22c55e; font-weight: 600; }
.rejected { color: #ef4444; text-decoration: line-through; }
.bonus { color: #a78bfa; font-weight: 600; }
footer { display: none !important; }
"""

BENCHMARK = get_precomputed_benchmark_results()

# --- Precomputed step visualization data ---
DEMO_STEPS = [
    {
        "step": 1,
        "prompt_snippet": "The future of AI is",
        "draft_tokens": [" bright", " and", " full", " of", " promise"],
        "accepted": [True, True, True, True, False],
        "bonus": " opportunities",
        "draft_time": 42,
        "verify_time": 38,
        "n_accepted": 5,  # 4 accepted + 1 bonus
    },
    {
        "step": 2,
        "prompt_snippet": "...full of opportunities",
        "draft_tokens": [" as", " machine", " learning", " models", " grow"],
        "accepted": [True, True, False, False, False],
        "bonus": " become",
        "draft_time": 41,
        "verify_time": 37,
        "n_accepted": 3,  # 2 + 1 bonus
    },
    {
        "step": 3,
        "prompt_snippet": "...learning models become",
        "draft_tokens": [" more", " capable", " and", " access", "ible"],
        "accepted": [True, True, True, True, True],
        "bonus": ",",
        "draft_time": 44,
        "verify_time": 39,
        "n_accepted": 6,  # all 5 + 1 bonus
    },
    {
        "step": 4,
        "prompt_snippet": "...capable and accessible,",
        "draft_tokens": [" transform", "ing", " industries", " like", " healthcare"],
        "accepted": [True, True, True, False, False],
        "bonus": " finance",
        "draft_time": 43,
        "verify_time": 38,
        "n_accepted": 4,
    },
]


def build_speedup_chart():
    bench = BENCHMARK
    methods = ["Autoregressive\n(Baseline)", "Speculative\nDecoding (K=5)"]
    tps = [bench["baseline"]["throughput_tps"], bench["speculative"]["throughput_tps"]]
    colors = ["#475569", "#6366f1"]

    fig = go.Figure([
        go.Bar(x=methods, y=tps, marker_color=colors,
               text=[f"{v} tok/s" for v in tps], textposition="outside",
               textfont=dict(color="#e2e8f0", size=14))
    ])
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0"),
        title=f"Throughput: {bench['speculative']['speedup']} Speedup",
        yaxis_title="Tokens per Second", height=380,
        yaxis=dict(range=[0, 200]),
        margin=dict(t=50, b=10),
    )
    return fig


def build_acceptance_chart():
    data = BENCHMARK["acceptance_by_prompt_type"]
    types = list(data.keys())
    rates = [data[t] for t in types]

    fig = go.Figure([
        go.Bar(
            x=rates, y=types, orientation="h",
            marker_color=["#22c55e" if r > 0.75 else "#f59e0b" if r > 0.65 else "#ef4444" for r in rates],
            text=[f"{r:.0%}" for r in rates], textposition="outside",
        )
    ])
    fig.add_vline(x=0.70, line_dash="dash", line_color="#a78bfa",
                  annotation_text="Breakeven ~70%")
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0"),
        title="Acceptance Rate by Prompt Type",
        xaxis=dict(range=[0, 1.05]),
        height=320, margin=dict(t=50, b=10, l=200, r=80),
    )
    return fig


def build_k_sweep_chart():
    data = BENCHMARK["speedup_vs_K"]
    fig = go.Figure([
        go.Scatter(
            x=data["K_values"], y=data["speedup"],
            mode="lines+markers",
            line=dict(color="#6366f1", width=3),
            marker=dict(size=8, color="#a78bfa"),
            name="Observed Speedup",
        ),
        go.Scatter(
            x=data["K_values"],
            y=[k * 0.71 for k in data["K_values"]],  # theoretical: K * α
            mode="lines", line=dict(color="#f59e0b", dash="dash"),
            name="Theoretical (K × α, α=0.71)",
        ),
    ])
    fig.add_vline(x=5, line_dash="dot", line_color="#22c55e",
                  annotation_text="K=5 (optimal)")
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0"),
        title="Speedup vs Draft Length K (GPT-2 → GPT-2-Medium)",
        xaxis_title="K (tokens drafted per step)",
        yaxis_title="Speedup vs Baseline",
        height=380, legend=dict(x=0.01, y=0.99),
        margin=dict(t=50, b=10),
    )
    return fig


def build_step_visualization(step_idx: int):
    """Build token acceptance visualization for a speculative step."""
    step = DEMO_STEPS[step_idx % len(DEMO_STEPS)]

    tokens_html = ""
    for token, accepted in zip(step["draft_tokens"], step["accepted"]):
        if accepted:
            tokens_html += f"<span class='accepted' title='ACCEPTED (α = min(1, p_target/p_draft))'>{token}</span>"
        else:
            tokens_html += f"<span class='rejected' title='REJECTED — correction sampled from (p_target - α·p_draft)'>{token}</span>"

    bonus = step.get("bonus", "")
    if bonus:
        tokens_html += f"<span class='bonus' title='BONUS: sampled from verifier final distribution'>{bonus} ★</span>"

    n_accepted = step["n_accepted"]
    n_proposed = len(step["draft_tokens"])

    return f"""
    <div class='card'>
        <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:14px'>
            <div style='color:#a5b4fc;font-weight:700;font-size:1.05em'>Step {step["step"]}</div>
            <div style='font-size:0.82em;color:#64748b'>
                Draft: {step["draft_time"]}ms · Verify: {step["verify_time"]}ms
            </div>
        </div>
        <div style='color:#64748b;font-size:0.8em;margin-bottom:8px'>Context: "{step["prompt_snippet"]}"</div>
        <div style='background:#111827;border-radius:8px;padding:12px;margin-bottom:12px;font-size:1.1em;line-height:2;word-spacing:2px'>
            {tokens_html}
        </div>
        <div style='display:flex;gap:20px;font-size:0.82em'>
            <div><span style='color:#22c55e'>✓ green</span> = accepted</div>
            <div><span style='color:#ef4444'>✗ strikethrough</span> = rejected (corrected)</div>
            <div><span style='color:#a78bfa'>★ purple</span> = bonus token</div>
        </div>
        <div style='margin-top:12px;background:#111827;border-radius:6px;padding:8px;font-size:0.85em'>
            <span style='color:#64748b'>Tokens proposed:</span> <span style='color:#e2e8f0'>{n_proposed}</span> ·
            <span style='color:#64748b'>Tokens accepted:</span> <span style='color:#22c55e;font-weight:600'>{n_accepted}</span> ·
            <span style='color:#64748b'>Acceptance:</span> <span style='color:#a78bfa;font-weight:600'>{n_accepted/(n_proposed+1):.0%}</span>
        </div>
    </div>
    """


def run_live_generation(prompt: str, K_val: int):
    """Live generation (only available with ENABLE_LIVE_SPECULATIVE=1)."""
    if not ENABLE_LIVE:
        return build_step_visualization(0), (
            "⚠️ Live generation requires GPU. See the 'Demo Steps' tab for step-by-step visualization."
        )

    try:
        from speculative.decoder import SpeculativeDecoder
        decoder = SpeculativeDecoder(K=K_val)
        result = decoder.generate(prompt, max_new_tokens=60, record_steps=True)
        step_htmls = [build_step_visualization(0)]  # simplified for demo
        return step_htmls[0], result.output
    except Exception as e:
        return f"<div class='card'>Error: {e}</div>", ""


with gr.Blocks(css=CSS, theme=gr.themes.Soft(primary_hue="violet"), title="Speculative Decoding") as demo:

    gr.HTML("""
    <div style='text-align:center;padding:28px 0 18px'>
        <div style='font-size:2.8em'>⚡</div>
        <h1 style='color:#e2e8f0;margin:10px 0 6px;font-size:1.9em;font-weight:700'>
            Speculative Decoding — From Scratch
        </h1>
        <p style='color:#64748b;max-width:720px;margin:0 auto;line-height:1.6'>
            Small draft model proposes K tokens, large verifier accepts or rejects in ONE forward pass.
            The output distribution is mathematically identical to the large model alone — just faster.
        </p>
    </div>
    """)

    with gr.Tabs():

        with gr.Tab("🎯 Step Visualizer"):
            gr.HTML("""
            <div class='card'>
                <div style='color:#94a3b8;font-size:0.9em'>
                    GPT-2 (117M) drafts tokens → GPT-2-Medium (345M) verifies in one pass.
                    Green = accepted, red = rejected with correction, purple★ = bonus token.
                </div>
            </div>
            """)
            step_slider = gr.Slider(1, 4, value=1, step=1, label="Speculative Step Number")
            step_display = gr.HTML(build_step_visualization(0))

            step_slider.change(
                fn=lambda s: build_step_visualization(int(s) - 1),
                inputs=step_slider, outputs=step_display,
            )

        with gr.Tab("📊 Benchmark Results"):
            gr.HTML(f"""
            <div class='card'>
                <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:20px;text-align:center'>
                    <div>
                        <div style='color:#6366f1;font-size:2em;font-weight:700'>{BENCHMARK["speculative"]["speedup"]}</div>
                        <div style='color:#64748b;font-size:0.82em'>Speedup over baseline</div>
                    </div>
                    <div>
                        <div style='color:#22c55e;font-size:2em;font-weight:700'>{BENCHMARK["speculative"]["mean_acceptance_rate"]:.0%}</div>
                        <div style='color:#64748b;font-size:0.82em'>Mean acceptance rate</div>
                    </div>
                    <div>
                        <div style='color:#a78bfa;font-size:2em;font-weight:700'>{BENCHMARK["speculative"]["throughput_tps"]}</div>
                        <div style='color:#64748b;font-size:0.82em'>Tokens/sec (K=5)</div>
                    </div>
                </div>
            </div>
            """)
            with gr.Row():
                gr.Plot(build_speedup_chart())
                gr.Plot(build_acceptance_chart())
            gr.Plot(build_k_sweep_chart())

        with gr.Tab("🧮 The Math"):
            gr.Markdown("""
## Rejection Sampling Acceptance Criterion

The core insight: we want output matching `p_target` exactly but using `p_draft` for speed.

**Acceptance probability per token:**
```
α_i = min(1, p_target(t_i | context) / p_draft(t_i | context))
```

**Decision:**
- Sample r ~ Uniform(0, 1)
- If r < α_i → **ACCEPT** token t_i
- Else → **REJECT**, sample corrected token from:
  ```
  p_corrected = (p_target - α_i × p_draft).clip(0) / Z
  ```
  where Z is the normalization constant

**Why this works (proof sketch):**
The marginal probability of token t at position i, after accounting for accept/reject:
```
P(output = t) = P(draft = t) × α(t) + P(reject) × p_corrected(t)
              = p_draft(t) × min(1, p_target(t)/p_draft(t))
                + p_reject × (p_target(t) - α(t)×p_draft(t)) / (1 - Σ_t' α(t')p_draft(t'))
              = p_target(t)  ✓
```

The output distribution is **exactly** p_target — no approximation, no quality loss.

## Bonus Token

When all K draft tokens are accepted, we get to sample one additional token
from the verifier's distribution at no extra compute cost:
- Verifier already computed the final logits in its forward pass
- → Free token: increases expected tokens per step from K to K+1

## Expected Tokens Per Step

```
E[tokens per step] = Σ_{i=1}^{K} P(first i tokens all accepted) + P(all K accepted)
                   ≈ (1 - α^K) / (1 - α)  [geometric series]  + α^K  (bonus)
```

For α=0.71, K=5:
```
E[tokens] ≈ 3.47 tokens per verifier forward pass
Vs baseline: 1 token per forward pass
→ 3.47x theoretical speedup (observe 1.87x due to draft overhead + batching)
```

## Implementation Complexity

```python
# The entire accept/reject logic in ~10 lines:
for i, draft_token in enumerate(draft_tokens):
    alpha = min(1, p_target[i, draft_token] / p_draft[i])
    if random() < alpha:
        accept(draft_token)        # matches target distribution
    else:
        # Sample from corrected distribution
        p_corrected = (p_target[i] - alpha * p_draft_dist[i]).clamp(0)
        accept(sample(p_corrected / p_corrected.sum()))
        break  # stop at first rejection
```

## Why Not Just Run the Draft Model?

The draft model (GPT-2, 117M) is 3x faster but outputs different text —
possibly lower quality for complex tasks. Speculative decoding gets you
the large model's quality at nearly the draft model's speed.

Key condition: **draft and target must share the same tokenizer**
(same vocabulary). GPT-2 and GPT-2-Medium both use GPT-2's BPE tokenizer,
so they work together. This is a practical constraint in production deployment.
            """)

        with gr.Tab("⚡ Live Generation"):
            if ENABLE_LIVE:
                with gr.Row():
                    prompt_in = gr.Textbox(
                        label="Prompt",
                        value="The future of artificial intelligence is",
                        lines=2, scale=3,
                    )
                    k_slider = gr.Slider(1, 8, value=5, step=1, label="K (draft tokens)", scale=1)
                gen_btn = gr.Button("Generate with Speculative Decoding", variant="primary", size="lg")
                live_step = gr.HTML()
                live_output = gr.Textbox(label="Generated Text", lines=4)
                gen_btn.click(fn=run_live_generation, inputs=[prompt_in, k_slider], outputs=[live_step, live_output])
            else:
                gr.HTML("""
                <div class='card' style='text-align:center;padding:40px'>
                    <div style='font-size:2em;margin-bottom:12px'>🖥️</div>
                    <div style='color:#94a3b8;font-size:1.05em'>Live generation requires a GPU environment.</div>
                    <div style='color:#64748b;margin-top:8px;font-size:0.9em'>
                        Set <code>ENABLE_LIVE_SPECULATIVE=1</code> and run on a T4/A10 instance.
                        All benchmark results on other tabs are pre-computed.
                    </div>
                </div>
                """)

demo.launch()
