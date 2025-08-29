"""Streamlit app showcasing LLM watermarking and spoof replication.

Tabs cover: a primer on KGW watermarking, a high-level view of watermark
stealing, a small replication demo, and reflections/policy notes.
"""

import math
import random
import hashlib
import json
import os
import pandas as pd
import altair as alt
import streamlit as st

st.title("Watermark Stealing Replication")
st.caption("Juan Belieni -- [AISB](https://aisb.dev) Capstone Project")


watermark_tab, watermark_stealing_tab, replication_tab, conclusion_tab = st.tabs(
    ["Watermark", "Watermark Stealing", "Spoof Attack Replication", "Conclusion"]
)

with watermark_tab:
    st.subheader("Why Watermark LLM Outputs?")
    st.markdown(
        """
        Large Language Model (LLM) watermarking encodes a hidden statistical signal in the
        model's generated text so that an authorized verifier can later test whether a passage
        likely came from a specific model. Goals:

        - Provenance: attribute text to a model or deployment.
        - Integrity: detect large-scale automated generation (e.g., spam, misuse).
        - Low overhead: preserve fluency and utility; no fine-tuning required.

        Watermarks are not cryptographic signatures embedded in the text itself; they are subtle
        shifts in token sampling probabilities that are only detectable with the secret key.
        """
    )

    st.divider()

    st.subheader("KGW Scheme (Greenlist Sampling)")
    st.markdown(
        """
        A widely used LLM watermark works
        by biasing the model to prefer a secret, pseudorandom subset of the vocabulary — the
        greenlist — at each decoding step. The subset changes per position using a secret key and a
        pseudo‑random function (PRF), so it looks random to anyone without the key.
        """
    )

    with st.expander("Algorithm intuition", expanded=True):
        st.markdown(
            """
            - Let $V$ be the vocabulary. Choose a **greenlist rate** $\\gamma \\in (0,1)$ and a **bias**
                $\\delta > 0$.
            - With a secret key $K$, a PRF maps the current context to a seed that selects a set
                $G_t \\subset V$ with $|G_t| \\approx \\gamma \\cdot |V|$.
            - At time step $t$, take the model logits $\\ell_t(i)$ and add $\\delta$ only to tokens in
                the greenlist, then sample as usual.
            """
        )

        st.latex(
            r"p_t(i) = \frac{\exp(\ell_t(i) + \delta\,\mathbf{1}[i\in G_t])}{\sum_{j\in V} \exp(\ell_t(j) + \delta\,\mathbf{1}[j\in G_t])}"
        )

    st.markdown(
        """
        The text still looks natural because the model could plausibly choose many tokens, but
        across a long passage it chooses green tokens a little more often than chance.
        """
    )

    st.divider()

    st.subheader("Verification (One‑Sided Hypothesis Test)")
    st.markdown(
        """
        Given a generated text `x`, the verifier recomputes each position's greenlist using the key
        and counts how many emitted tokens fall in the corresponding greenlist. Under the null
        hypothesis (unwatermarked text), the count is approximately Binomial with success
        probability $\\gamma$. A simple z‑score detects the watermark:
        """
    )

    st.latex(
        r"z = \frac{\sum_{t=1}^{n} \mathbf{1}[x_t \in G_t] - \gamma n}{\sqrt{n\,\gamma(1-\gamma)}}"
    )

    st.markdown(
        """
        Hypotheses and what is tested:
        - H0 (unwatermarked): at each position, the event "$\\text{token} \\in \\text{greenlist}$" is $\\text{Bernoulli}(\\gamma)$. The total $S$ of greens is $\\text{Binomial}(n, \\gamma)$, so $z \\sim N(0,1)$.
        - H1 (watermarked): watermark bias increases the green rate ($E[S] > \\gamma n$), shifting z positive.
        - Test: one‑sided; we reject H0 for large positive z (an excess of green tokens over chance).
        """
    )

    st.markdown(
        """
        Choose a threshold $\\tau$ to control the false positive rate (FPR). If $z \\ge \\tau$ we
        declare the text watermarked. Larger bias $\\delta$ and longer text $n$ increase power.
        Typical choices: $\\gamma \\approx 0.5$, $\\delta \\in [1.5, 4]$.
        """
    )

    # Interactive: visualize z under H0 and simulate colored tokens
    st.subheader("Simulation")
    st.caption("Assume fixed length n = 100 tokens (word‑tokenized lorem ipsum).")


    n_fixed = 100
    gamma = st.slider("Green rate $\\gamma$", 0.00, 1.0, 0.5, step=0.05)
    s_green = st.slider("Observed green tokens (s)", 0, n_fixed, 50, step=2)

    mu = gamma * n_fixed
    sigma = math.sqrt(n_fixed * gamma * (1.0 - gamma))
    # Guard against σ=0 at γ in {0,1} to avoid infs/NaNs in charts
    z_obs = (s_green - mu) / sigma if sigma > 0 else 0.0

    def normal_pdf(z: float) -> float:
        return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * z * z)

    st.metric("Observed z-score", f"{z_obs:.2f}")

    if "green_seed" not in st.session_state:
        st.session_state.green_seed = 42

    base_words = (
        "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor "
        "incididunt ut labore et dolore magna aliqua ut enim ad minim veniam quis nostrud "
        "exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat duis aute irure "
        "dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur "
        "excepteur sint occaecat cupidatat non proident sunt in culpa qui officia deserunt mollit anim id est laborum"
    ).split()
    tokens = (base_words * (n_fixed // len(base_words) + 1))[:n_fixed]

    rng = random.Random(st.session_state.green_seed)
    green_idx = set(rng.sample(range(n_fixed), s_green)) if s_green > 0 else set()

    # Build colored HTML
    green_style = "color:#2e7d32;font-weight:600;"
    red_style = "color:#c62828;"
    html_tokens = []
    for i, tok in enumerate(tokens):
        if i in green_idx:
            html_tokens.append(f'<span style="{green_style}">{tok}</span>')
        else:
            html_tokens.append(f'<span style="{red_style}">{tok}</span>')
    legend = (
        '<span style="color:#2e7d32;font-weight:700;">green</span> = in greenlist · '
        '<span style="color:#c62828;font-weight:700;">red</span> = out of greenlist'
    )
    html_block = (
        '<div style="border:1px solid #eee;padding:12px;border-radius:8px;'
        'line-height:1.7;font-size:0.95rem;">' + " ".join(html_tokens) + "</div>"
    )
    st.markdown(
        f"Selected s = {s_green} greens out of n = {n_fixed}. {legend}",
        unsafe_allow_html=True,
    )
    st.markdown(html_block, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Plot standard normal density and overlay a vertical rule at z_obs
    z_min, z_max, step = -5.0, 5.0, 0.02
    x_left = min(z_min, z_obs - 0.1)
    x_right = max(z_max, z_obs + 0.1)
    z_vals = [x_left + i * step for i in range(int((x_right - x_left) / step) + 1)]
    pdf_vals = [normal_pdf(z) for z in z_vals]
    norm_df = pd.DataFrame({"z": z_vals, "pdf": pdf_vals})

    line = (
        alt.Chart(norm_df)
        .mark_line(color="#1f77b4")
        .encode(
            x=alt.X("z:Q", title="z", scale=alt.Scale(domain=[x_left, x_right])),
            y=alt.Y("pdf:Q", title="Standard normal density"),
        )
    )
    # Use same dataset for the rule layer to avoid unnamed dataset refs in Vega
    rule = alt.Chart(norm_df).mark_rule(color="#ff7f0e", strokeWidth=2).encode(
        x=alt.datum(z_obs)
    )
    chart = (line + rule).properties(height=220)
    st.altair_chart(chart, use_container_width=True)
    st.caption("Standard normal under H0 with your z (orange) as a vertical line.")

    st.divider()

    st.subheader("How is it implemented?")

    st.markdown(
        """
        The watermarking algorithm is surprisingly simple. It works at sampling time, so it can be
        applied to any pre-trained model without retraining. The only overhead is a small logit bias
        per step and the need to compute the greenlist using a PRF (e.g., HMAC‑SHA256).
        """
    )

    st.code(
        """
# Generation (with secret key K)
for t in 1..n:
    seed = PRF(K, context[:t])                # derive per-step seed
    G_t  = sample_greenlist(V, seed, rate=gamma)
    logits = model(context[:t])
    logits[i in G_t] += delta                   # bias green tokens
    x_t = sample_softmax(logits)
    append x_t to context
        """,
        language="python",
    )

    st.markdown(
        """
        The verifier, who knows the key, recomputes the greenlist at each step and counts how many
        emitted tokens fall in the corresponding greenlist. The z‑score is easy to compute.
        """
    )

    st.code(
        """
# Verification (given text x_1..x_n and key K)
z = 0; s = 0
for t in 1..n:
    seed = PRF(K, context[:t])
    G_t  = sample_greenlist(V, seed, rate=gamma)
    s   += indicator(x_t in G_t)
z = (s - gamma*n) / sqrt(n*gamma*(1-gamma))
watermarked = (z >= tau)
        """,
        language="python",
    )

    st.divider()

    st.subheader("Strengths and Caveats")
    col1, col2 = st.columns(2)
    with col1:
        st.success(
            """
            - Model‑agnostic: works at sampling time; no retraining.
            - Efficient: a small logit bias per step.
            - Tunable: trade off visibility (delta) vs. utility.
            """
        )
    with col2:
        st.warning(
            """
            - Edits/paraphrases dilute the signal.
            - Deterministic decoding (e.g., greedy) weakens embedding.
            - Access to the key enables detection but must be kept secret.
            - Not a proof of authorship; statistical evidence only.
            """
        )

with watermark_stealing_tab:
    st.subheader("What Is Watermark Stealing? 😈")
    st.markdown(
        """
        Watermark stealing is an attack where an adversary, with black‑box access to a
        watermarked model, learns a surrogate predictor that approximates the model’s hidden
        watermark partition (the per‑step greenlist vs. redlist). Once learned, the surrogate can
        be used to:

        - Spoof: cause unwatermarked text to be detected as watermarked.
        - Scrub: transform genuine watermarked text so the detector misses it.

        This undermines watermarking’s goals of provenance and attribution.
        """
    )

    st.info("Original paper available in [arXiv:2402.19361](https://arxiv.org/pdf/2402.19361).")

    st.divider()

    st.subheader("Threat Model")
    st.markdown(
        """
        - Model owner deploys a proprietary instruction‑tuned model $\\text{LM}_{mo}$ with a statistical
            watermark keyed by secret $\\xi$.
        - Attacker has black‑box access to full generations of $\\text{LM}_{mo}$ via an API and knows a
            watermark is present behind the API.
        - Goal: with a minimal number of queries, learn an approximate model of the watermarking
            rules determined by $\\xi$ (i.e., steal the watermark), decoupled from downstream uses
            like spoofing/scrubbing.
        - Kerckhoffs’ principle: attacker knows the scheme and its parameters (e.g., $\\gamma,\\delta$),
            but not the secret key $\\xi$.
        - Auxiliary model $\\text{LM}_{att}$ is available (open‑source base model) to provide features
            or a proxy for the non‑watermarked distribution.
        """
    )

    # Dimensions table
    with st.expander("Threat model dimensions", expanded=True):
        st.markdown(
            """
            - **Detector access**: D0 (no access) vs. D1 (binary API access; detector returns only a flag,
            not the exact z‑score).
            - **Availability of base responses**: B0 (unavailable) vs. B1 (available, e.g., pre‑watermark corpus or
            unwatermarked mode).
            """
        )

        dims_df = pd.DataFrame(
            {
                "": ["B0 · Unavailable base", "B1 · Available base"],
                "D0 · No detector access": [
                    "Most restrictive.",
                    "Leverage base corpus only",
                ],
                "D1 · API access (binary)": [
                    "Validate/evaluate via detector API",
                    "Use both base corpus and detector",
                ],
            }
        )
        st.table(dims_df)
        st.caption(
            "Paper reports that the algorithm applies to all four settings; experiments focus on (D0, B0) with additional results in the appendix."
        )

    st.divider()

    st.subheader("Core Idea: Learn the Hidden Partition")
    st.markdown(
        """
        KGW watermarking secretly boosts a pseudorandom subset of tokens (greenlist) at each step.
        Without the key, we can’t recover the exact set, but we can learn how the watermarked
        distribution reshapes token probabilities relative to an unwatermarked base. Concretely,
        estimate two conditional pmfs:

        - Base pmf: $p_b(T\mid\\text{ctx})$ from an unwatermarked model.
        - Watermarked pmf: $p_w(T\mid\\text{ctx})$ fitted from observed watermarked outputs.

        Then define a clipped likelihood‑ratio scorer that favors tokens made more likely by the
        watermark:
        """
    )

    st.latex(
        r"s(T, \text{ctx}) = \begin{cases} \tfrac{1}{c}\,\min\!\left( \tfrac{p_w(T\mid\text{ctx})}{p_b(T\mid\text{ctx})},\ c \right), & \text{if } \tfrac{p_w}{p_b} \ge 1 \\ 0, & \text{otherwise} \end{cases}"
    )
    st.markdown("$c > 1$ is a clipping constant to bound the bias.")

    with st.expander("Steps to perform the attack", expanded=True):
        st.markdown(
            """
            - Obtain $p_b(\cdot\mid\\text{ctx})$ and $p_w(\cdot\mid\\text{ctx})$ by counting and computing the probabilities from the model outputs.
            - Compute the ratio $r(T,\\text{ctx})=p_w/p_b$ and apply clipping.
            - Use $s(T,\\text{ctx})$ to bias decoding: add for spoofing, subtract for scrubbing.
            """
        )

    st.divider()

    st.subheader("Two Attacks")
    with st.expander("1) Spoof (forge watermarked look)", expanded=False):
        st.markdown(
            """
            Generate with an unwatermarked model, but add a small bias toward tokens the surrogate
            judges as green. This increases the green‑token rate, raising the detector’s z‑score so
            texts are falsely flagged as watermarked.
            """
        )
        st.code(
            """
# Spoofing with surrogate s
for t in 1..n:
    logits = base_model(context[:t])
    for token in top_k(vocab):
        logits[token] += delta * s(context[:t], token)
    x_t = sample_softmax(logits)
    append x_t to context
            """,
            language="python",
        )

    with st.expander("2) Scrub (evade detection)", expanded=False):
        st.markdown(
            """
            Take genuine watermarked text and paraphrase it while steering away from tokens the
            surrogate predicts as green. This lowers the green rate and z‑score, pushing texts below
            the detection threshold while preserving meaning.
            """
        )
        st.code(
            """
# Scrubbing guided by surrogate s
for t in 1..n:
    logits = paraphraser(context[:t])
    for token in top_k(vocab):
        logits[token] -= delta * s(context[:t], token)
    y_t = sample_softmax(logits)
    append y_t to context
            """,
            language="python",
        )

    # Interactive: Spoof Attack Simulation
    st.subheader("Simulation of Spoof Attack")
    st.caption(
        "Move δ to see tokens shift toward detector‑green. Only $\delta$ is adjustable."
    )

    # --- Controls (only δ)
    delta_bias = st.slider("Attack bias $\delta$", 0.0, 4.0, 0.0, step=0.1)

    # Fixed settings for clarity
    sim_len = 100
    gamma_sim = 0.5
    clip_c = 4.0
    base_seed = 1234
    key_seed = 4242

    # --- Toy vocabulary and simple bigram‑ish base model
    base_text = (
        "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor "
        "incididunt ut labore et dolore magna aliqua ut enim ad minim veniam quis nostrud "
        "exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat duis aute irure "
        "dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur "
        "excepteur sint occaecat cupidatat non proident sunt in culpa qui officia deserunt mollit anim id est laborum"
    )
    vocab = list(dict.fromkeys(base_text.split()))

    def build_bigrams(words):
        counts = {}
        for a, b in zip(words, words[1:]):
            d = counts.setdefault(a, {})
            d[b] = d.get(b, 0) + 1
        return counts

    words_seq = base_text.split()
    bigrams = build_bigrams(words_seq)
    unigrams = {w: words_seq.count(w) for w in vocab}

    def base_pmf(ctx_last: str | None, alpha: float = 0.1) -> dict[str, float]:
        weights = {}
        bg_total = max(sum(unigrams.values()), 1)
        for w in vocab:
            c_bg = unigrams.get(w, 0) / bg_total
            c_bi = bigrams.get(ctx_last, {}).get(w, 0) if ctx_last in bigrams else 0.0
            weights[w] = alpha * c_bg + (1.0 - alpha) * c_bi
        # strictly positive + normalize
        min_pos = 1e-6
        total = 0.0
        for w in vocab:
            weights[w] = max(weights[w], min_pos)
            total += weights[w]
        return {w: weights[w] / total for w in vocab}

    def greenlist(ctx_last: str | None, t: int) -> set[str]:
        h = hashlib.sha256(f"{key_seed}|{t}|{ctx_last or ''}".encode()).digest()
        rnd = random.Random(h)
        k = max(1, int(round(gamma_sim * len(vocab))))
        return set(rnd.sample(vocab, k))

    def watermarked_pmf(
        pb: dict[str, float], G_t: set[str], delta_w: float = 2.0
    ) -> dict[str, float]:
        logits = {}
        for w, p in pb.items():
            logit = math.log(p)
            if w in G_t:
                logit += delta_w
            logits[w] = logit
        m = max(logits.values())
        exps = {w: math.exp(v - m) for w, v in logits.items()}
        Z = sum(exps.values())
        return {w: exps[w] / Z for w in vocab}

    def scorer_s(
        pb: dict[str, float], pw: dict[str, float], c: float
    ) -> dict[str, float]:
        s = {}
        eps = 1e-12
        for w in vocab:
            ratio = pw[w] / max(pb[w], eps)
            s[w] = (1.0 / c) * min(ratio, c) if ratio >= 1.0 else 0.0
        return s

    def decode(delta_bias: float, n: int, seed: int, prompt_tokens: list[str]):
        rng = random.Random(seed)
        ctx = list(prompt_tokens)
        emitted = []
        greens = []
        s_count = 0
        for t in range(1, n + 1):
            last = ctx[-1] if ctx else None
            pb = base_pmf(last)
            Gt = greenlist(last, t)
            pw = watermarked_pmf(pb, Gt)
            s_map = scorer_s(pb, pw, clip_c)
            logits = {w: math.log(pb[w]) + delta_bias * s_map[w] for w in vocab}
            m = max(logits.values())
            exps = {w: math.exp(v - m) for w, v in logits.items()}
            Z = sum(exps.values())
            r = rng.random()
            cdf = 0.0
            chosen = None
            for w in vocab:
                cdf += exps[w] / Z
                if r <= cdf:
                    chosen = w
                    break
            chosen = chosen or vocab[-1]
            emitted.append(chosen)
            ctx.append(chosen)
            is_green = chosen in Gt
            greens.append(is_green)
            s_count += 1 if is_green else 0
        return emitted, greens, s_count

    # Deterministic small prompt (internal)
    rnd = random.Random(base_seed)
    prompt_tokens = [rnd.choice(vocab) for _ in range(3)]

    tokens, greens, s_count = decode(delta_bias, sim_len, base_seed, prompt_tokens)

    # Show observed z-score (one-sided test baseline γ = 0.5)
    n = sim_len
    mu = gamma_sim * n
    sigma = math.sqrt(max(n * gamma_sim * (1.0 - gamma_sim), 1e-9))
    z_val = (s_count - mu) / sigma
    st.metric("Observed z-score", f"{z_val:.2f}")

    # Match styles to earlier visualization
    green_style = "color:#2e7d32;font-weight:600;"
    red_style = "color:#c62828;"
    html_tokens = []
    for tok, is_green in zip(tokens, greens):
        if is_green:
            html_tokens.append(f'<span style="{green_style}">{tok}</span>')
        else:
            html_tokens.append(f'<span style="{red_style}">{tok}</span>')
    legend = (
        '<span style="color:#2e7d32;font-weight:700;">green</span> = in greenlist · '
        '<span style="color:#c62828;font-weight:700;">red</span> = out of greenlist'
    )
    st.markdown(legend, unsafe_allow_html=True)
    html_block = (
        '<div style="border:1px solid #eee;padding:12px;border-radius:8px;line-height:1.7;font-size:0.95rem;">'
        + " ".join(html_tokens)
        + "</div>"
    )
    st.markdown(html_block, unsafe_allow_html=True)

    st.write("")

    st.warning("Caveat: increasing attack bias $\delta$ can harm output quality.")

with replication_tab:
    st.subheader("KGW2‑SelfHash")
    st.markdown(
        """
        For the replication, I reimplemented KGW2‑SelfHash watermark with window size
        $w = 2$. I intentionally chose this value instead the standard $w = 3$ to simplify
        experimentation and speed up iteration for this replication.

        However, it is important to note some trade-offs about window sizes:
        - Larger windows: spoofing becomes harder (stronger context coupling), but scrubbing becomes easier (more structure to paraphrase against).
        - Smaller windows: spoofing becomes easier (weaker coupling), but scrubbing becomes harder (less exploitable structure).

        Samples were generated with [Llama 3.2 3B](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) and [Qwen 2.5 3B](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) using prompts from the [UltraFeedback prompts dataset](https://huggingface.co/datasets/trl-lib/ultrafeedback-prompt).
        The code is available in [juanbelieni/watermark-stealing-replication](https://github.com/juanbelieni/watermark-stealing-replication).
        """
    )

    # KGW2‑SelfHash greenlist selection pseudocode
    with st.expander("KGW2‑SelfHash: greenlist (pseudocode)", expanded=False):
        st.code(
            """
# Inputs at step t
#   ctx         : previous token ids
#   logits      : model logits for all tokens at step t
#   V           : vocabulary token ids
#   gamma       : ratio of green tokens between
#   w           : window size
#   F_K         : PRF with key K
# Returns
#   G_t         : green tokens at step t

sorted_logits = argsort_desc(scores)
G_t = []

for x, logit in sorted_logits[:1000]: # Limit computation
    seed = min(F_K(x, ctx[t - i]) for i range(1, w + 1))
    u = random(seed)
    if u < gamma:
        G_t.append(v)

return set(G_t)
            """,
            language="python",
        )

    st.divider()

    st.subheader("Watermark Examples")


    data_path = os.path.join("web", "examples.json")

    def load_examples(path: str):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            st.warning(f"No examples file found at `{data_path}`. Using an empty list.")
            return []
        except Exception as e:
            st.error(f"Failed to load examples: {e}")
            return []

    examples = load_examples(data_path)

    if not examples:
        st.info(
            "Add your dataset as a JSON list with fields: `prompt`, `base` {`tokens`, `green`} and `watermarked` {`tokens`, `green`}."
        )

    if examples:
        idx = st.selectbox(
            "Choose example by prompt",
            options=list(range(len(examples))),
            index=0,
            format_func=lambda i: examples[i].get("prompt", f"Example {i}"),
        )

        ex = examples[int(idx)]

        st.markdown(f"**Prompt:** {ex.get('prompt', '')}")

        def render_series(label: str, series: dict):
            tokens: list[str] = series.get("tokens", [])
            greens: list[bool] = series.get("green", [])
            if len(tokens) != len(greens):
                st.error(
                    f"Token/label length mismatch in `{label}`: {len(tokens)} vs {len(greens)}"
                )
                return
            n = len(tokens)
            s = sum(1 for g in greens if g)
            gamma_fixed = 0.5
            mu = gamma_fixed * n
            sigma = math.sqrt(max(n * gamma_fixed * (1.0 - gamma_fixed), 1e-12))
            z = (s - mu) / sigma if sigma > 0 else float("inf")
            st.metric("Observed z-score", f"{z:.2f}")

            green_style = "color:#2e7d32;font-weight:600;"
            red_style = "color:#c62828;"
            html_tokens = []
            for tok, is_green in zip(tokens, greens):
                if is_green:
                    html_tokens.append(f'<span style="{green_style}">{tok}</span>')
                else:
                    html_tokens.append(f'<span style="{red_style}">{tok}</span>')
            html_block = (
                '<div style="border:1px solid #eee;padding:12px;border-radius:8px;line-height:1.7;font-size:0.95rem;">'
                + "".join(html_tokens)
                + "</div>"
            )
            st.markdown(html_block, unsafe_allow_html=True)

        legend = (
            '<span style="color:#2e7d32;font-weight:700;">green</span> = in greenlist · '
            '<span style="color:#c62828;font-weight:700;">red</span> = out of greenlist'
        )
        st.markdown(legend, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Base")
            render_series("Base", ex.get("base", {}))
        with c2:
            st.subheader("Watermarked")
            render_series("Watermarked", ex.get("watermarked", {}))

    st.divider()

    st.subheader("Spoof Attack Replication (B1, D0)")
    st.markdown(
        """
        Threat model and setup:
        - Dimensions used: **B1** (base responses available), **D0** (no detector access). This contrasts
          base vs. watermarked behavior without relying on a detector API.
        - Alternate setting: **B0, D0** is also feasible by swapping in a different auxiliary/base model
          to approximate the unwatermarked distribution when true base outputs are unavailable.

        Data and procedure:
        - Collected ~30,000 generations each from the base and watermarked models (≈350 tokens on average).
        - Computed a per-(context, target) "boost" that captures how much the watermark increases the relative odds of a token
          compared to the base.
        - During decoding, added a bias proportional to the boost to steer outputs toward tokens likely to be green under the watermark,
          yielding spoofed samples.

        Results below visualize the FPR for multiple runs, varying the number of samples used and the attack delta.
        """
    )



    # Build the chart from CSV data and style for dark background
    csv_path = os.path.join(os.path.dirname(__file__), "graph_data_b1.csv")
    try:
        df = pd.read_csv(csv_path)

        base = alt.Chart(df).encode(
            x=alt.X("limit_samples:Q", title="Number of samples", scale=alt.Scale(type="log")),
            y=alt.Y("success_rate:Q", title="FPR", axis=alt.Axis(format="%")),
            color=alt.Color("delta_att:N", title="δ"),
            tooltip=[
                alt.Tooltip("limit_samples:Q", title="Samples"),
                alt.Tooltip("success_rate:Q", title="Success", format=".1%"),
                alt.Tooltip("z_avg:Q", title="Avg. z", format=".2f"),
                alt.Tooltip("delta_att:N", title="δ"),
            ],
        )

        line = base.mark_line(interpolate="monotone")

        # Add end-of-line labels colored per series
        labels = (
            base.transform_joinaggregate(max_limit="max(limit_samples)", groupby=["delta_att"])  # type: ignore
            .transform_filter("datum.limit_samples == datum.max_limit")
            .transform_calculate(label='"δ = " + toString(datum.delta_att)')
            .mark_text(align="left", dx=6, dy=0, fontSize=12)
            .encode(text=alt.Text("label:N"), color=alt.Color("delta_att:N", legend=None))
        )

        chart = (
            (line + labels)
            .configure_view(strokeWidth=0)
            .configure_axis(
                grid=True,
                gridColor="#333333",
                domainColor="#666666",
                tickColor="#666666",
                labelColor="#FFFFFF",
                titleColor="#FFFFFF",
            )
            .configure_legend(labelColor="#FFFFFF", titleColor="#FFFFFF")
            .configure_title(color="#FFFFFF")
        )

        st.altair_chart(chart, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to render graph: {e}")

    st.divider()

    st.subheader("Spoof Attack Replication Alternative Setting (B0, D0)")
    st.markdown(
        """
        Setup notes:
        - Dimensions: **B0** (no true base responses), **D0** (no detector access).
        - Approach: approximate the base distribution with an auxiliary model to learn the surrogate scorer, then decode with a small spoofing bias.

        This experiment is very similar to the previous one, only changing the auxiliary model.
        """
    )

    # Build the chart from B0 CSV data (same schema)
    csv_path_b0 = os.path.join(os.path.dirname(__file__), "graph_data_b0.csv")
    try:
        df_b0 = pd.read_csv(csv_path_b0)

        base_b0 = alt.Chart(df_b0).encode(
            x=alt.X("limit_samples:Q", title="Number of samples", scale=alt.Scale(type="log")),
            y=alt.Y("success_rate:Q", title="FPR", axis=alt.Axis(format="%")),
            color=alt.Color("delta_att:N", title="δ"),
            tooltip=[
                alt.Tooltip("limit_samples:Q", title="Samples"),
                alt.Tooltip("success_rate:Q", title="Success", format=".1%"),
                alt.Tooltip("z_avg:Q", title="Avg. z", format=".2f"),
                alt.Tooltip("delta_att:N", title="δ"),
            ],
        )

        line_b0 = base_b0.mark_line(interpolate="monotone")

        labels_b0 = (
            base_b0.transform_joinaggregate(max_limit="max(limit_samples)", groupby=["delta_att"])  # type: ignore
            .transform_filter("datum.limit_samples == datum.max_limit")
            .transform_calculate(label='"δ = " + toString(datum.delta_att)')
            .mark_text(align="left", dx=6, dy=0, fontSize=12)
            .encode(text=alt.Text("label:N"), color=alt.Color("delta_att:N", legend=None))
        )

        chart_b0 = (
            (line_b0 + labels_b0)
            .configure_view(strokeWidth=0)
            .configure_axis(
                grid=True,
                gridColor="#333333",
                domainColor="#666666",
                tickColor="#666666",
                labelColor="#FFFFFF",
                titleColor="#FFFFFF",
            )
            .configure_legend(labelColor="#FFFFFF", titleColor="#FFFFFF")
            .configure_title(color="#FFFFFF")
        )

        st.altair_chart(chart_b0, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to render B0 graph: {e}")

with conclusion_tab:
    st.subheader("Conclusion")
    st.markdown(
        """
        **What this replication demonstrates**
        - **Surrogates are inexpensive and useful.** A clipped likelihood-ratio scorer trained from base vs. watermarked samples
          can approximate the hidden green/red partition well enough to **spoof** (push $z$ above threshold).
        - **Defensive tuning has costs.** Small bias $\\delta$ can improve spoof attack defense but makes detection harder.

        **Why the attack works (intuition)**
        - KGW-style schemes bias a rotating subset of tokens; learning *relative changes* $p_w/p_b$ is enough to steer sampling
          towards likely-green tokens without knowing the key.
        """
    )

    st.divider()

    st.subheader("What worked vs. what broke")
    c1, c2 = st.columns(2)
    with c1:
        st.success(
            """
            **Worked**
            - Lightweight surrogate $s(\\text{ctx}, t)$ lifted detector $z$ reliably in spoofing.
            - Results transferred across prompts reasonably well.
            - Using a **different auxiliary model** in the B0/D0 setting still produced effective surrogates,
            showing that attackers don’t need the true base model to succeed.
            """
        )
    with c2:
        st.warning(
            """
            **Broke / Fragile**
            - For larger windows (e.g. $w=3$), the surrogate required a much larger dataset of samples; this
            made the attack infeasible to replicate fully within this project’s scope.
            - Higher $\\delta$ makes spoofing more successful, but can degrade quality.
            - Low-entropy sequences reduces the signal available for both watermark detection and surrogate-based attacks,
            limiting effectiveness on both sides.
            """
        )

    st.info("Bottom line: statistical text watermarking is fragile under data-driven extraction. Treat it as auxiliary, not primary, provenance.")

    st.divider()

    st.subheader("Open technical questions")

    st.markdown(
        """
        - Can we design text-native signals that survive paraphrase without major utility loss?
        - What are realistic attacker query budgets in deployed systems?
        - How well do surrogates transfer across model scales and instruction-tuning variants?
        - What detection thresholds control false positives at internet scale?
        """
    )
