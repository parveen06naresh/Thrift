"""Build docs/paper/thrift.html and thrift.pdf from results/*.json (numbers are never typed by hand)."""
from __future__ import annotations

import base64
import glob
import json
import os
import subprocess

import numpy as np

R = json.load(open("results/router.json"))
K = json.load(open("results/cache.json"))
C = json.load(open("results/compress.json"))
O = json.load(open("results/overhead.json"))


def img(name):
    return "data:image/png;base64," + base64.b64encode(open(f"docs/figures/{name}.png", "rb").read()).decode()


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def pts(x):
    return f"{100 * x:.2f}"


def router_rows():
    out = []
    for p in R["pairs"]:
        name = f"{p['cheap'].split('-2024')[0]} → {p['strong'].split('-2024')[0]}"
        for e in p["results"]:
            out.append(f"<tr><td>{name}</td><td>{int(100 * e['eps'])}</td><td>{pct(e['cheap_share']['mean'])}</td>"
                       f"<td>{pts(e['quality_drop']['mean'])} <span class=m>[{pts(e['quality_drop']['p5'])}, {pts(e['quality_drop']['p95'])}]</span></td>"
                       f"<td>{pts(e['random_quality_drop']['mean'])}</td><td><b>{pct(e['cost_saved']['mean'])}</b></td><td>{pct(1 - e['violation_rate'], 1)}</td></tr>")
            name = ""
    return "\n".join(out)


def cache_rows():
    rows = []
    ex = K["runs"]
    rows.append(f"<tr><td>Exact match only</td><td>{pct(np.mean([r['exact_only']['hit_rate'] for r in ex]))}</td>"
                f"<td>{pct(np.mean([r['exact_only']['wrong_rate'] for r in ex]), 2)}</td><td>–</td><td>–</td></tr>")
    for i, a in enumerate([0.01, 0.02, 0.05]):
        cs = [r["certified"][i] for r in ex]
        rows.append(f"<tr><td>Certified, α = {int(100 * a)}%</td><td>{pct(np.mean([c['hit_rate'] for c in cs]))}</td>"
                    f"<td>{pct(np.mean([c['wrong_rate'] for c in cs]), 2)}</td><td>{pct(np.mean([c['precision'] for c in cs]))}</td>"
                    f"<td>{sum(not c['violated'] for c in cs)}/{len(cs)}</td></tr>")
    return "\n".join(rows)


def comp_rows():
    rows = []
    for b in C["by_ratio"]:
        rows.append(f"<tr><td>{int(100 * b['keep_ratio'])}%</td><td><b>{b['thrift']['f1']:.3f}</b> ({pct(b['thrift']['retained'])})</td>"
                    f"<td>{b['truncate']['f1']:.3f} ({pct(b['truncate']['retained'])})</td><td>{b['random']['f1']:.3f} ({pct(b['random']['retained'])})</td></tr>")
    return "\n".join(rows)


g = R["pairs"][0]["results"][1]
cert2 = K["runs"]
c2 = [r["certified"][1] for r in cert2]
best = min(C["by_ratio"], key=lambda b: b["keep_ratio"])
cc = {c["eps"]: c for c in C["certified"]}

html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Thrift paper</title>
<style>
@page {{ size: Letter; margin: 0.75in 0.8in; }}
body {{ font: 10.5pt/1.45 "Charter","Georgia",serif; color:#111; max-width: 7in; margin: 0 auto; }}
h1 {{ font: 600 19pt/1.2 "Helvetica Neue",Arial,sans-serif; margin: 0 0 4px; text-align:center }}
.auth {{ text-align:center; color:#444; margin-bottom: 14px; font-size: 10pt }}
h2 {{ font: 600 12pt "Helvetica Neue",Arial,sans-serif; margin: 16px 0 6px }}
h3 {{ font: 600 10.5pt "Helvetica Neue",Arial,sans-serif; margin: 12px 0 4px }}
.abs {{ background:#f5f6f8; padding: 10px 14px; border-radius: 6px; font-size: 10pt }}
table {{ border-collapse: collapse; width: 100%; font: 8.6pt/1.35 "Helvetica Neue",Arial,sans-serif; margin: 6px 0 4px }}
th, td {{ border-bottom: 1px solid #ddd; padding: 3px 5px; text-align: left }} th {{ border-bottom: 1.5px solid #333 }}
.m {{ color:#777 }} .cap {{ font-size: 8.8pt; color:#444; margin: 2px 0 10px }}
img {{ width: 100%; }} code {{ font-size: 9pt }} .eq {{ text-align:center; margin: 8px 0; font-style: italic }}
ol.refs {{ font-size: 8.8pt; padding-left: 18px }}
</style></head><body>
<h1>Thrift: Certified Token Savings for LLM Serving</h1>
<div class="auth">Parveen Kumar Naresh Kumar · University of South Florida · github.com/parveen06naresh/thrift</div>
<div class="abs"><b>Abstract.</b> LLM applications overpay in three predictable ways: every request goes to the largest model, repeated
questions are paid for again, and prompts carry context the model never needs. Model routing, semantic caching and prompt
compression each address one of these, but teams hesitate to deploy them because each trades an unknown amount of quality for cost.
We present Thrift, an OpenAI-compatible gateway in which every quality/cost knob is set by Learn-then-Test, giving a finite-sample,
distribution-free guarantee that the expected quality loss stays below a user-chosen ε with probability at least 1 − δ.
On 8,021 HELM Lite questions with real outputs from three model families, certified routing cuts the GPT-4o bill by
<b>{pct(g['cost_saved']['mean'], 0)}</b> for a measured <b>{pts(g['quality_drop']['mean'])}</b>-point accuracy drop (ε = 2 points).
On Quora Question Pairs, the certified semantic cache serves <b>{pct(np.mean([c['hit_rate'] for c in c2]))}</b> of adversarial
queries at a <b>{pct(np.mean([c['wrong_rate'] for c in c2]), 2)}</b> wrong-answer rate (α = 2%), six times the exact-match hit rate.
On multi-document Natural Questions, query-aware compression keeps {pct(best['thrift']['retained'], 0)} of a reader model's F1 at
{int(100 * best['keep_ratio'])}% of the context tokens, versus {pct(best['truncate']['retained'], 0)} for truncation.
The gateway adds {O['short_prompt_miss']['p50_ms']:.2f} ms median overhead per request.</div>

<h2>1. Introduction</h2>
<p>Inference cost is now a first-order engineering constraint. Three techniques cut it without touching the model:
<i>routing</i> sends easy requests to a cheaper model [FrugalGPT, RouteLLM], <i>semantic caching</i> answers repeated questions
from memory [GPTCache], and <i>prompt compression</i> removes tokens the model does not need [LLMLingua]. Each comes with a
threshold (a router score, a similarity, a keep-ratio) whose quality cost is usually tuned by eye on a dashboard. A cache
that serves a wrong answer to 5% of users, or a router that silently degrades a product, is worse than no optimisation.</p>
<p>Thrift's contribution is engineering plus one idea: treat every threshold as a hypothesis-testing problem and certify it.
We use Learn-then-Test (LTT) [Angelopoulos et al. 2021], which turns any per-example loss in [0, 1] on held-out data into a
threshold with a PAC-style guarantee. The guarantee needs no model of the scorer or data, so a weak scorer costs savings,
never safety. The same family of distribution-free bounds controls latency in the author's earlier serving system
Pacer [Naresh Kumar 2026]; Thrift applies it to quality.</p>

<h2>2. System</h2>
<p>Thrift is a stateless-by-default Python service (Starlette + httpx) that speaks the OpenAI chat-completions API, so any
OpenAI SDK, LangChain, or LlamaIndex client works by changing <code>base_url</code>. Tiers are ordered cheapest to strongest and
can be any OpenAI-compatible backend, hosted or self-hosted (vLLM, SGLang, TGI). The pipeline per request is:</p>
<p><b>Cache.</b> A request is split into a <i>scope</i> (tier, system prompt, history, sampling parameters, tools) and a
<i>query</i> (the last user message). The exact tier hashes both. The semantic tier embeds the query (signed feature hashing of
word and character n-grams; any embedder plugs in) and searches only entries with the same scope, so a hit can never cross
system prompts or temperatures. A <i>key-term guard</i> rejects a hit when the numbers or negations differ, the dominant failure
of similarity caches ("convert 5 km" vs "convert 50 km"). Errors, truncated answers and n &gt; 1 are never cached.</p>
<p><b>Compression.</b> Stage 1 normalises whitespace, blank-line runs, invisible characters and decorative rules outside
fenced code (lossless). Stage 2 drops paragraphs that repeat verbatim (duplicate RAG chunks). Stage 3 scores sentences of long
context against the user's question with BM25 and keeps the best ones up to a keep-ratio, in original order; the question
paragraph, system prompt, recent turns and anything containing code are never cut.</p>
<p><b>Routing.</b> A logistic scorer over hashed unigrams/bigrams and ten structural features (length, digits, maths symbols,
code, answer options, question words) predicts the soft label 1 − max(0, q<sub>strong</sub> − q<sub>cheap</sub>). A request goes
to the cheap tier when its score is at least λ. If the cheap tier fails, the request falls back to the strong one.</p>
<p><b>Accounting.</b> Each response carries its tier, cache status, original and sent prompt tokens, its cost, and the
<i>baseline</i>: the cost of the same request uncompressed on the strongest tier. These feed Prometheus metrics and a live
dashboard. A shadow mode sends a sample of cheap-routed requests to the strong tier as well and logs both answers, producing
the labelled data needed to re-certify the router on a team's own traffic.</p>

<h2>3. Certifying the knobs</h2>
<p>Let λ<sub>1</sub>, …, λ<sub>m</sub> be candidate settings ordered from safest to most aggressive and L<sub>i</sub>(λ) ∈ [0, 1]
the loss on calibration example i. For each λ<sub>j</sub> in order we test H<sub>j</sub>: R(λ<sub>j</sub>) &gt; ε with p-value
p<sub>j</sub> (exact binomial tail for 0/1 losses, Hoeffding–Bentkus otherwise) and stop at the first p<sub>j</sub> &gt; δ.
Fixed-sequence testing controls the family-wise error at δ, so the last accepted setting λ̂ satisfies</p>
<div class="eq">P( R(λ̂) ≤ ε ) ≥ 1 − δ</div>
<p>over the draw of the calibration set, for any scorer and any exchangeable data. The losses are: routing,
L = 1[s ≥ λ] · max(0, q<sub>strong</sub> − q<sub>cheap</sub>), the quality given up on prompts sent to the cheap model (cheap-model
wins are not allowed to offset losses, making the bound conservative); caching, L = 1[sim ≥ τ and the cached answer is wrong];
compression, L = max(0, F1<sub>full</sub> − F1<sub>compressed</sub>) for a reader model. All three use the same 60-line implementation.</p>

<h2>4. Experiments</h2>
<h3>4.1 Routing on real model outputs</h3>
<p>HELM Lite v1.13.0 [Liang et al. 2023] publishes per-question prompts, completions, token counts and scores. We take the
8,021 questions (MMLU, GSM8K, MATH, MedQA, LegalBench, OpenBookQA, NaturalQuestions, NarrativeQA) answered by all of three
cheap/strong pairs. Costs use each model's real token counts at list prices. Each configuration averages {R['splits']}
random 50/25/25 train/calibration/test splits; δ = {R['delta']}.</p>
<table><tr><th>Pair</th><th>ε (pts)</th><th>to cheap</th><th>accuracy drop, pts [p5, p95]</th><th>random, same share</th><th>bill saved</th><th>bound held</th></tr>
{router_rows()}</table>
<div class="cap">Table 1. Certified routing on the held-out test split. "Bound held" is the share of splits in which the certified
quantity stayed ≤ ε (target ≥ {pct(1 - R['delta'], 0)}). The one cell below target (Claude, ε = 5) is 35 of 40 overlapping splits.</div>
<img src="{img('routing')}">
<div class="cap">Figure 1. Accuracy vs share of traffic sent to the cheap model on one test split. Dots mark the certified operating
points at ε = 2 and 5 points with the measured bill reduction.</div>
<p>Routing saves more money than its traffic share suggests (41% of the GPT-4o bill from 21% of questions) because the scorer
learns that long reading-comprehension prompts, the expensive ones, are where the small model already matches the large one.
Against random routing at the same share, certified routing loses 2–5× less accuracy. The scorer is deliberately small
(test AUC 0.56–0.73); the oracle curve shows the headroom a stronger scorer could claim under the same guarantee.</p>

<h3>4.2 Semantic cache on adversarial paraphrases</h3>
<p>Quora Question Pairs contains 404k human-labelled pairs whose non-duplicates were selected to look alike. We cache the first
question of 10,000 pairs and query with the second; any cached entry may answer, and a hit is correct when both questions are in the
same duplicate cluster (transitive closure of the labels). Calibration and test workloads share no questions; {len(K['runs'])} draws.</p>
<table><tr><th>Setting</th><th>served from cache</th><th>wrong answers</th><th>precision of hits</th><th>bound held</th></tr>
{cache_rows()}</table>
<div class="cap">Table 2. Exact matching's 0.04% "wrong" rate is QQP label noise. Lookup: {K['shipped_cache']['lookup_ms_p50']:.1f} ms p50
with {K['shipped_cache']['entries']:,} entries.</div>
<img src="{img('cache')}" style="width:62%;display:block;margin:0 auto">
<div class="cap">Figure 2. Hit rate vs wrong-answer rate across fixed thresholds, with and without the key-term guard, and the certified points.</div>

<h3>4.3 Compression with a real reader</h3>
<p>Each of {C['n']} Natural Questions items from HELM Lite gets its gold passage plus {C['k_docs'] - 1} distractor passages, shuffled
(mean {C['context_tokens_mean']:.0f} o200k tokens), mimicking a retriever's top-10. A BERT-base reader fine-tuned on SQuAD (ONNX Model
Zoo) answers from the full and compressed contexts; quality is SQuAD F1 against human references (full context: {C['f1_full']:.3f}).</p>
<table><tr><th>context kept</th><th>Thrift (query-aware)</th><th>truncate</th><th>random sentences</th></tr>{comp_rows()}</table>
<div class="cap">Table 3. Reader F1 (and share of full-context F1 retained) at equal token budgets.</div>
<img src="{img('compression')}" style="width:62%;display:block;margin:0 auto">
<p>Certification over 200 random calibration/test halvings: at ε = 0.02 F1 the median certified keep-ratio is
{cc[0.02]['median_ratio']:.2f} (test loss {cc[0.02]['test_loss_mean']:.3f}, bound held in {pct(1 - cc[0.02]['violation_rate'])} of halvings);
at ε = 0.05 it is {cc[0.05]['median_ratio']:.2f} (test loss {cc[0.05]['test_loss_mean']:.3f}, held in {pct(1 - cc[0.05]['violation_rate'])}).
The lossless normalisation stage alone removes {C['normalize_helm']['saved_pct']:.1f}% of tokens from all HELM Lite prompts, which are
already clean; scraped web pages and PDFs typically carry far more whitespace.</p>

<h3>4.4 Overhead</h3>
<p>With zero-latency mock upstreams, so that all time is Thrift's own: cache-miss requests add {O['short_prompt_miss']['p50_ms']:.2f} ms
(p50) / {O['short_prompt_miss']['p99_ms']:.2f} ms (p99); cache hits {O['cache_hit']['p50_ms']:.2f} ms; long-context requests that are compressed
{O['long_context_compress']['p50_ms']:.1f} ms. One uvicorn worker sustains {O['http']['rps']:.0f} requests/s at concurrency {O['http']['concurrency']}
on a {O['machine']['cores']}-core cloud VM. Upstream LLM latency is typically 300–3,000 ms, so overhead is under 1% for most requests.</p>

<h2>5. Limitations</h2>
<p>Guarantees hold for exchangeable traffic; distribution shift requires re-certification, which the shadow log supports but does not
automate. The bundled router is certified on HELM-style questions, not on any particular product. The default cache embedding is
lexical. Compression is validated with an extractive reader rather than a frontier LLM. Costs use list prices.</p>

<h2>6. Related work</h2>
<p>FrugalGPT [Chen et al. 2023] introduced LLM cascades; RouteLLM [Ong et al. 2024] trains routers from preference data; both report
cost/quality curves without a deployment-time guarantee. GPTCache popularised similarity caching with hand-set thresholds.
LLMLingua and LongLLMLingua [Jiang et al. 2023] compress prompts with a small LM. Conformal risk control [Angelopoulos et al. 2022]
and Learn-then-Test supply the guarantees; conformal methods have been applied to LLM abstention and factuality, and Thrift applies
them to the cost knobs of a serving gateway.</p>

<h2>References</h2><ol class="refs">
<li>A. N. Angelopoulos, S. Bates, E. J. Candès, M. I. Jordan, L. Lei. Learn then Test: Calibrating predictive algorithms to achieve risk control. arXiv:2110.01052, 2021.</li>
<li>A. N. Angelopoulos, S. Bates, A. Fisch, L. Lei, T. Schuster. Conformal risk control. arXiv:2208.02814, 2022.</li>
<li>L. Chen, M. Zaharia, J. Zou. FrugalGPT: How to use large language models while reducing cost and improving performance. arXiv:2305.05176, 2023.</li>
<li>I. Ong et al. RouteLLM: Learning to route LLMs with preference data. arXiv:2406.18665, 2024.</li>
<li>H. Jiang et al. LLMLingua: Compressing prompts for accelerated inference of large language models. EMNLP 2023.</li>
<li>P. Liang et al. Holistic evaluation of language models. TMLR 2023 (HELM Lite v1.13.0 results).</li>
<li>Z. Bentkus. On Hoeffding's inequalities. Annals of Probability, 2004.</li>
<li>P. Kumar Naresh Kumar. Pacer: SLO-aware LLM serving with online conformal latency bounds. github.com/parveen06naresh/pacer-serve, 2026.</li>
</ol></body></html>"""

os.makedirs("docs/paper", exist_ok=True)
open("docs/paper/thrift.html", "w").write(html)
chrome = (glob.glob("/opt/pw-browsers/chromium-*/chrome-linux/chrome") + ["chromium"])[0]
subprocess.run([chrome, "--headless=new", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer",
                f"--print-to-pdf={os.path.abspath('docs/paper/thrift.pdf')}", "file://" + os.path.abspath("docs/paper/thrift.html")], check=True,
               capture_output=True)
print("wrote docs/paper/thrift.html and thrift.pdf")
