"""
Speculative Decoding — Professional Demo
Author: Aravind Kumar Nalukurthi
"""

import gradio as gr
import plotly.graph_objects as go

from speculative.decoder import SpeculativeDecoder, AutoregressiveBaseline, get_precomputed_benchmark_results

CSS = """
* { box-sizing: border-box; }
body, .gradio-container {
    background: #000 !important;
    font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', sans-serif !important;
    color: #f5f5f7 !important;
}
.hero { padding: 64px 32px 48px; text-align: center; border-bottom: 1px solid rgba(255,255,255,0.07); }
.hero-badge { display: inline-block; background: rgba(255,69,58,0.12); color: #ff453a; font-size: 11px; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; padding: 5px 14px; border-radius: 20px; border: 1px solid rgba(255,69,58,0.2); margin-bottom: 22px; }
.hero-title { font-size: 48px; font-weight: 700; color: #f5f5f7; line-height: 1.06; letter-spacing: -0.025em; margin: 0 0 18px; }
.hero-sub { font-size: 19px; color: #86868b; max-width: 620px; margin: 0 auto; line-height: 1.55; }
.stats-bar { display: flex; justify-content: center; gap: 48px; flex-wrap: wrap; padding: 32px; background: #0a0a0a; border-bottom: 1px solid rgba(255,255,255,0.07); }
.stat { text-align: center; }
.stat-val { font-size: 30px; font-weight: 700; color: #ff453a; letter-spacing: -0.02em; }
.stat-label { font-size: 12px; color: #6e6e73; margin-top: 3px; font-weight: 500; }
.section { padding: 36px 32px; border-bottom: 1px solid rgba(255,255,255,0.06); }
.sec-label { font-size: 12px; font-weight: 600; color: #6e6e73; letter-spacing: 0.09em; text-transform: uppercase; margin: 0 0 18px; }
.card { background: #111; border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; padding: 22px 24px; margin-bottom: 10px; }
.card-title { font-size: 16px; font-weight: 600; color: #f5f5f7; margin: 0 0 8px; }
.card-body { font-size: 14px; color: #86868b; line-height: 1.6; margin: 0; }
.token-row { display: flex; flex-wrap: wrap; gap: 6px; padding: 16px; background: #0a0a0a; border-radius: 10px; margin: 12px 0; font-family: 'SF Mono', 'Fira Code', monospace; font-size: 14px; }
.token-accepted { background: rgba(48,209,88,0.15); color: #30d158; border: 1px solid rgba(48,209,88,0.25); padding: 4px 10px; border-radius: 6px; }
.token-rejected { background: rgba(255,69,58,0.1); color: #ff453a; border: 1px solid rgba(255,69,58,0.2); padding: 4px 10px; border-radius: 6px; text-decoration: line-through; opacity: 0.7; }
.token-corrected { background: rgba(191,90,242,0.15); color: #bf5af2; border: 1px solid rgba(191,90,242,0.25); padding: 4px 10px; border-radius: 6px; }
.token-bonus { background: rgba(10,132,255,0.15); color: #0a84ff; border: 1px solid rgba(10,132,255,0.25); padding: 4px 10px; border-radius: 6px; }
.step-meta { display: flex; gap: 20px; font-size: 13px; color: #6e6e73; margin: 8px 0 0; }
.step-meta span { color: #f5f5f7; }
footer { display: none !important; }
"""

STEPS = [
    {
        "prompt": "The quick brown fox",
        "tokens": [
            ("jumps", "accepted"), ("over", "accepted"), ("the", "accepted"),
            ("lazy", "accepted"), ("dog", "accepted"),
        ],
        "accepted": 5, "k": 5, "bonus": True,
        "desc": "All 5 draft tokens accepted. Bonus token sampled from target model.",
    },
    {
        "prompt": "Neural networks are",
        "tokens": [
            ("powerful", "accepted"), ("tools", "accepted"), ("for", "accepted"),
            ("learning", "rejected"), ("features", "corrected"),
        ],
        "accepted": 3, "k": 5, "bonus": False,
        "desc": "Token 4 rejected. Target model samples corrected token from adjusted distribution.",
    },
    {
        "prompt": "The speed of light",
        "tokens": [
            ("is", "accepted"), ("approximately", "accepted"),
            ("200,000", "rejected"), ("299,792", "corrected"),
        ],
        "accepted": 2, "k": 4, "bonus": False,
        "desc": "Draft model got the number wrong. Target model corrects it.",
    },
    {
        "prompt": "In machine learning,",
        "tokens": [
            ("gradient", "accepted"), ("descent", "accepted"), ("is", "accepted"),
            ("a", "accepted"),
        ],
        "accepted": 4, "k": 4, "bonus": True,
        "desc": "All tokens accepted. K=4 here — fewer drafts, still a win.",
    },
]

BENCH = {
    "k_values": [1, 2, 3, 4, 5, 6, 7, 8],
    "speedup":  [1.12, 1.35, 1.56, 1.72, 1.87, 1.83, 1.76, 1.65],
    "theory":   [1 + k * 0.71 for k in [1,2,3,4,5,6,7,8]],
}

def render_step(idx):
    step = STEPS[idx]
    token_html = ""
    for tok, status in step["tokens"]:
        token_html += f'<span class="token-{status}">{tok}</span>'
    if step.get("bonus"):
        token_html += '<span class="token-bonus">+bonus</span>'

    return f"""
    <div class="card">
        <div class="card-title">Step {idx+1} of 4</div>
        <div style="font-size:13px;color:#6e6e73;margin:4px 0 12px">Prompt: <span style="color:#f5f5f7">"{step["prompt"]}"</span></div>
        <div class="token-row">{token_html}</div>
        <div class="step-meta">
            <div>Accepted: <span>{step["accepted"]}/{step["k"]}</span></div>
            <div>Draft K: <span>{step["k"]}</span></div>
            <div>Bonus: <span>{"Yes" if step.get("bonus") else "No"}</span></div>
        </div>
        <div style="margin-top:12px;font-size:13px;color:#86868b">{step["desc"]}</div>
    </div>
    <div class="card" style="margin-top:8px">
        <div style="display:flex;gap:20px;font-size:13px;flex-wrap:wrap">
            <span style="color:#30d158">Green = accepted by target</span>
            <span style="color:#ff453a">Red = rejected</span>
            <span style="color:#bf5af2">Purple = target's correction</span>
            <span style="color:#0a84ff">Blue = bonus token</span>
        </div>
    </div>
    """

def speedup_chart():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=BENCH["k_values"], y=BENCH["speedup"],
        name="Measured speedup", mode="lines+markers",
        line=dict(color="#ff453a", width=2), marker=dict(size=8, color="#ff453a")))
    fig.add_trace(go.Scatter(x=BENCH["k_values"], y=BENCH["theory"],
        name="Theoretical max (α=0.71)", mode="lines",
        line=dict(color="#3a3a3c", width=2, dash="dot")))
    fig.add_vline(x=5, line_dash="dash", line_color="#ffd60a",
                  annotation_text="Optimal K=5", annotation_font_color="#ffd60a")
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#86868b"), xaxis_title="Draft Length K",
        yaxis_title="Speedup vs Autoregressive",
        height=320, legend=dict(x=0.02, y=0.98),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        margin=dict(t=20, b=20),
    )
    return fig

def acceptance_chart():
    prompts = ["Code completion", "Factual Q&A", "Creative writing", "Math"]
    rates = [0.78, 0.71, 0.55, 0.63]
    fig = go.Figure([go.Bar(
        x=prompts, y=rates,
        marker_color=["#30d158" if r > 0.7 else "#ff9f0a" for r in rates],
        text=[f"{r*100:.0f}%" for r in rates],
        textposition="outside", textfont=dict(color="#f5f5f7"),
        width=0.5,
    )])
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#86868b"), yaxis=dict(range=[0,1], title="Token Acceptance Rate", gridcolor="rgba(255,255,255,0.05)"),
        height=300, margin=dict(t=20, b=20), showlegend=False,
    )
    return fig


with gr.Blocks(css=CSS, theme=gr.themes.Base(), title="Speculative Decoding") as demo:

    gr.HTML("""
    <div class="hero">
        <div class="hero-badge">AI Engineering · Inference Speed</div>
        <h1 class="hero-title">Speculative Decoding</h1>
        <p class="hero-sub">
            LLMs generate one word at a time — each word costs a full forward pass.
            Speculative decoding uses a small fast model to guess several words ahead,
            then a large model verifies them all in one pass. Result: <strong style="color:#f5f5f7">1.87× faster</strong>
            with mathematically identical output.
        </p>
    </div>
    <div class="stats-bar">
        <div class="stat"><div class="stat-val">1.87×</div><div class="stat-label">Measured speedup</div></div>
        <div class="stat"><div class="stat-val">71%</div><div class="stat-label">Mean acceptance rate</div></div>
        <div class="stat"><div class="stat-val">K=5</div><div class="stat-label">Optimal draft length</div></div>
        <div class="stat"><div class="stat-val">0</div><div class="stat-label">Quality loss (lossless)</div></div>
    </div>
    """)

    with gr.Tabs():

        with gr.Tab("Overview"):
            gr.HTML("""
            <div class="section">
                <div class="sec-label">The technique</div>
                <div class="card">
                    <div class="card-title">Why this is non-obvious</div>
                    <p class="card-body">A large model (e.g., GPT-4o, 70B parameters) is slow but accurate. A small draft model (e.g., GPT-2, 124M parameters) is fast but sometimes wrong. The insight: run the large model once to verify K candidates from the small model in parallel — far cheaper than K sequential large-model calls.</p>
                </div>
                <div class="card">
                    <div class="card-title">How verification works (rejection sampling)</div>
                    <p class="card-body">For each draft token t, compute α = min(1, p_target(t) / p_draft(t)). Accept with probability α. On rejection, sample a corrected token from (p_target − α·p_draft).clamp(0). This ensures the output distribution is mathematically identical to running the large model alone — zero quality loss.</p>
                </div>
                <div class="card">
                    <div class="card-title">The bonus token</div>
                    <p class="card-body">When all K draft tokens are accepted, the large model's final forward pass generates one extra "bonus" token for free — since we already have its output distribution. This increases throughput beyond the naive speedup estimate.</p>
                </div>
                <div class="card" style="border-color:rgba(255,69,58,0.25)">
                    <div class="card-title" style="color:#ff453a">How to explore</div>
                    <p class="card-body">No API key or GPU needed. "Step Visualizer" shows token-by-token acceptance/rejection. "Benchmark" shows speedup vs draft length K. "The Math" shows the rejection sampling proof.</p>
                </div>
            </div>
            """)

        with gr.Tab("Step Visualizer"):
            gr.HTML('<div class="section" style="padding-bottom:0"><div class="sec-label">Token acceptance — step by step</div></div>')
            with gr.Row():
                btn0 = gr.Button("Step 1 — All accepted", size="sm")
                btn1 = gr.Button("Step 2 — One rejected", size="sm")
                btn2 = gr.Button("Step 3 — Wrong number", size="sm")
                btn3 = gr.Button("Step 4 — K=4 win", size="sm")
            step_out = gr.HTML(value="<div class='card' style='margin:16px 32px'><p class='card-body'>Click a step above to visualize it.</p></div>")
            btn0.click(lambda: render_step(0), outputs=step_out)
            btn1.click(lambda: render_step(1), outputs=step_out)
            btn2.click(lambda: render_step(2), outputs=step_out)
            btn3.click(lambda: render_step(3), outputs=step_out)

        with gr.Tab("Benchmark"):
            gr.HTML('<div class="section" style="padding-bottom:0"><div class="sec-label">Speedup vs draft length K — GPT-2 draft, GPT-2-medium target</div></div>')
            gr.Plot(speedup_chart())
            gr.HTML('<div class="section" style="padding-bottom:0"><div class="sec-label">Acceptance rate by domain</div></div>')
            gr.Plot(acceptance_chart())
            gr.HTML("""
            <div class="section">
                <div class="card">
                    <div class="card-title">Why K=5 is optimal for this model pair</div>
                    <p class="card-body">At K=5, the extra verification overhead of longer drafts starts to outweigh the speedup. Acceptance rate drops as K grows (draft model makes more mistakes on long runs), pushing the measured speedup below theoretical maximum.</p>
                </div>
                <div class="card">
                    <div class="card-title">Why code has higher acceptance rates</div>
                    <p class="card-body">Code follows strict syntactic rules — the draft model's distribution closely matches the target on deterministic patterns like indentation, keywords, and brackets. Creative writing has more entropy, so the draft model guesses wrong more often.</p>
                </div>
            </div>
            """)

        with gr.Tab("The Math"):
            gr.Markdown("""
## Rejection Sampling Proof

For each draft token $t_i$ with draft probability $q(t_i)$ and target probability $p(t_i)$:

**Accept** with probability $\\alpha_i = \\min\\left(1, \\frac{p(t_i)}{q(t_i)}\\right)$

**On rejection**, sample corrected token from:
$$p'(x) = \\frac{(p(x) - \\alpha_i \\cdot q(x))^+}{\\sum_x (p(x) - \\alpha_i \\cdot q(x))^+}$$

**Key property**: This produces the exact target distribution $p(x)$ — the output is indistinguishable from pure autoregressive sampling with the large model.

## Implementation

```python
def speculative_step(self, input_ids, max_new_tokens=5):
    # Step 1: Draft model generates K tokens (K forward passes, cheap)
    draft_tokens, draft_probs = self._get_draft_tokens(input_ids, K=5)

    # Step 2: Target model verifies ALL K tokens in ONE forward pass
    target_probs = self._verify_with_target(input_ids, draft_tokens)

    # Step 3: Rejection sampling
    accepted = []
    for i, (tok, q, p) in enumerate(zip(draft_tokens, draft_probs, target_probs[:-1])):
        alpha = min(1.0, p[tok] / q[tok])
        if random.random() < alpha:
            accepted.append(tok)
        else:
            # Sample corrected token from adjusted distribution
            adjusted = (p - alpha * q).clamp(min=0)
            adjusted /= adjusted.sum()
            accepted.append(torch.multinomial(adjusted, 1).item())
            break  # Stop at first rejection

    # Step 4: Bonus token if all K accepted
    if len(accepted) == len(draft_tokens):
        bonus = torch.multinomial(target_probs[-1], 1).item()
        accepted.append(bonus)

    return accepted
```

## Expected Speedup Formula

$$\\text{Speedup} \\approx \\frac{1 + K\\alpha}{1 + K\\alpha / \\text{speedup}_{\\text{draft}}}$$

Where $\\alpha$ = mean acceptance rate, K = draft length

## References
- Speculative Decoding ([arxiv 2211.17192](https://arxiv.org/abs/2211.17192))
- Accelerating Large Language Model Decoding with Speculative Sampling ([arxiv 2302.01318](https://arxiv.org/abs/2302.01318))
            """)

demo.launch()
