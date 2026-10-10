# d1 experiment harness — spec (for Codex)

Root: /workspace/substack-posts/liquid-d1  (harness/ code, data/ datasets, results/ outputs, images/ charts)
Python 3.13, deps installed: datasets, scikit-learn, pandas, numpy, requests, matplotlib. No torch.

## Models / endpoints (already running locally, CPU only, 8 threads, one request at a time)
1. `d1-local`  : LiquidAI d1-3B (Q4_K_M GGUF) via llama.cpp llama-server at http://127.0.0.1:8081
   POST /v1/systemone  body: {"state": str, "questions": {name: {"type":"noul"|"choice"|"score","instructions":str,"criteria":...}}}
   - noul: criteria optional. answer: {"type":"noul","noul":P(yes)}
   - choice: criteria = {label: description}. answer: {"choice":label,"probabilities":{label:p},"confidence":c}
   - score: criteria = [level descriptions lowest first]. answer: {"score":expected level (0-indexed float),"probabilities":{"0":p,...},"confidence":c}
   - response also has "usage":{"input_tokens":n,"output_tokens":0}
2. `qwen-local`: Qwen2.5-3B-Instruct (Q4_K_M GGUF, same size & quant as d1) via llama-server at http://127.0.0.1:8082
   OpenAI-compatible POST /v1/chat/completions with temperature 0, max_tokens 4, "logprobs": true, "top_logprobs": 20.
   Response: choices[0].message.content, choices[0].logprobs.content[0].top_logprobs = [{"token","logprob"}...], usage.prompt_tokens/completion_tokens.
   Turn this into a probability distribution over options from the FIRST generated token's top_logprobs:
   map tokens case-insensitively after stripping whitespace (e.g. "yes"/"Yes"/" yes" -> yes), sum probs per option, renormalise over the options.
   If an option is absent from top_logprobs give it 0 mass (renormalise; if all absent → uniform, and flag `invalid=True`).
   Also record the parsed text answer and whether it was a valid option.
3. `d1-hosted` (PENDING, must be implemented but will not run now): Liquid API https://api.liquid.ai/decisions/v1/systemone,
   Authorization: Bearer $LIQUID_API_KEY, body same as local plus "model":"d1" (or "d1:free"). Skip cleanly with a clear message if the env var is missing.
4. `openrouter:<model_id>` chat baseline (PENDING, implement, won't run now): https://openrouter.ai/api/v1/chat/completions with $OPENROUTER_API_KEY,
   same prompts/logprob handling as qwen-local (logprobs may be unsupported → fall back to text-only answer with prob 1.0 on the parsed option, flag it).
   Record usage tokens and, if returned, `usage.cost`.
   Also `openrouter-d1`: OpenRouter's Decisions API, POST https://openrouter.ai/api/v1/systemone with model "liquid/d1" (same body as Liquid API). Implement, won't run now.
Model registry should make adding endpoints a one-liner. Base URLs configurable via env vars.

## Tasks (data/ built by harness/prepare_data.py, seed 42, fully deterministic; write data/<task>.jsonl with id,text,label and data/<task>_train.jsonl for the TF-IDF baseline from DISJOINT data)
- `spam` (noul): HF `ucirvine/sms_spam` (split train, fields sms,label 0 ham/1 spam). Eval 200 items stratified: 60 spam, 140 ham. Train = all remaining rows.
  Question: "Is this SMS message spam (unsolicited advertising, scams, prize or premium-rate number promotions)?"
- `toxic` (noul): HF `cardiffnlp/tweet_eval` config `offensive`. Eval: 200 from test split stratified with natural ratio rounded (≈ 58 offensive / 142 not; compute from test-set ratio). Train = train split.
  Question: "Is this tweet offensive (insults, profanity, threats, or targeted abuse)?"
- `routing` (choice): HF `bitext/Bitext-customer-support-llm-chatbot-training-dataset` field `instruction`, label `category`. All 11 categories,
  eval 200 items (as balanced as possible, 18-19 per category), drop texts containing '{{' placeholders? NO — keep them, they are part of the data, but replace `{{Order Number}}`-style placeholders with a fixed realistic value via a small dict (e.g. {{Order Number}} -> #48213); unknown placeholders -> remove braces. Train = 3000 other rows.
  Write a one-line plain-English description for each category label (ACCOUNT, CANCEL, CONTACT, DELIVERY, FEEDBACK, INVOICE, NEWSLETTER, ORDER, PAYMENT, REFUND, SHIPPING — check actual label set) based on the label name only, e.g. "REFUND: asking for money back, refund status or refund policy". Store them in harness/tasks.py and use the SAME descriptions for d1 and the chat baseline.
  Question: "Which support team should handle this customer message?"
- `rating` (score): HF `Yelp/yelp_review_full` test split. Eval 200 = 40 per star. Truncate text to 1,000 characters (cut at a word boundary, add "…"). Labels 0-4 (1-5 stars). Train = 5000 random rows from train split.
  Question: "How many stars did this reviewer give?" criteria = ["1 star: very negative", "2 stars: mostly negative", "3 stars: mixed or average", "4 stars: mostly positive", "5 stars: very positive"].
Task definitions (question text, type, options) live in harness/tasks.py and are used by every model so prompts are identical in content.

Chat baseline prompt (system + user), for each task: system = "You are a careful classifier. Reply with only the answer token, nothing else." user = instructions + options list + the text, ending with "Answer:".
 - noul → options "yes"/"no"; choice → single letters A..K mapped to labels (show "A) REFUND — description"); score → digits 1..5.

## Baseline 3: `tfidf-lr`
scikit-learn TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True) + LogisticRegression(max_iter=2000) trained on data/<task>_train.jsonl; predict_proba on the eval set. For rating treat as 5-class. Record per-item latency (time per single predict on one item) too.

## Runner: harness/run.py
`python harness/run.py --model d1-local --task spam [--limit N]`  and `--all`.
- One request at a time (sequential), 1 warm-up call (not logged in metrics, but logged with warmup=true).
- Log EVERY call append-only to results/raw/<model>__<task>.jsonl: {ts_utc, model, task, id, label, request (full JSON body), response (full JSON), latency_ms (wall clock, time.perf_counter around the HTTP call), input_tokens, output_tokens, pred, probs{option:p}, valid, error}.
- Resumable: skip ids already logged without error. Retry transient HTTP errors up to 3x with backoff, log failures.
- Hard safety for paid endpoints: a --max-calls cap (default 1000) and a running cost estimate printed; abort if estimated spend > $2 (env MAX_USD).

## Multi-question experiment: harness/multiq.py (d1-local only)
30 routing eval items; for each item call d1 with k = 1, 2, 3, 5 questions in one request (questions: team (choice), is_complaint noul, urgency score 3 levels, wants_refund noul, is_angry noul). Log to results/raw/d1-local__multiq.jsonl with k, latency_ms, input_tokens. Shuffle order of k per item to avoid warm-cache bias.

## Analysis: harness/analyze.py → results/summary.csv, results/summary.json, results/per_item.csv
Per (model, task): n, n_errors, n_invalid, accuracy, macro_f1; binary tasks: precision/recall/F1 for positive class, ROC-AUC, Brier, ECE (10 equal-width bins on P(positive)); multiclass: ECE on top-1 confidence; rating: exact accuracy, within-1 accuracy, MAE in stars (d1 use argmax AND expected score rounded — report both, primary = argmax), Spearman. Latency p50/p95/mean (ms). Mean input tokens and output tokens.
Projected hosted cost per 1,000 decisions:
 - d1 at list price $0.04 / 1M input tokens using the logged d1 input_tokens (label clearly as projection; same tokenizer family as hosted is an assumption).
 - chat baselines: configurable price table in harness/prices.py (leave empty dict by default with a TODO; only compute when present).
Bootstrap 95% CI (1000 resamples, seed 0) for accuracy.
Also print a nice plain-text table to stdout and save it to results/summary.txt.

## Charts: harness/charts.py → images/*.png (dpi 200, 1600px wide-ish)
Clean light style: white background, no top/right spines, light grey y-grid, font DejaVu Sans, title left-aligned, small grey subtitle with "n=200 per task · CPU, 8 threads · <date>". Consistent palette:
 d1-local #5B3FD9 (purple), qwen-local #F28C28 (orange), tfidf-lr #9AA0A6 (grey), d1-hosted #2E86DE, openrouter:* #2BB673.
Charts:
 1. accuracy_by_task.png — grouped bars, accuracy with 95% CI error bars, value labels on bars.
 2. reliability_spam.png, reliability_toxic.png — reliability diagram (diagonal, per-model line with markers, bin counts as marker size) + ECE in legend.
 3. latency.png — p50 and p95 per model per task (horizontal bars or dot plot), ms, log x if needed.
 4. multiq.png — latency p50 and input tokens vs number of questions k (two panels).
 5. routing_confusion_d1.png — confusion matrix heatmap (row-normalised) for d1-local on routing; same for qwen-local.
 6. rating_scatter.png — true stars vs d1 expected score (jittered) with per-star mean line; plus same for qwen.
 7. confidence_hist.png — histogram of d1 P(yes) on spam split by true label.
Only plot data that exists in results; skip models without data.

## Terminal screenshots: harness/render_terminal.py
Render a text file (or stdin) to a PNG that looks like a terminal window (dark #1e1e2e bg, monospace DejaVu Sans Mono 15px, macOS-style 3 dots title bar, title text). Pure PIL (install pillow if missing via pip --break-system-packages). Usage: `python harness/render_terminal.py in.txt out.png --title "..."`. Wrap long lines at 110 chars.

## README.md at root/harness: how to reproduce (download llama.cpp b11541, GGUF links, commands), how to run hosted once a key exists.
Do NOT call any paid API. Do NOT run the full experiments yourself — just build, then do a smoke test with --limit 3 on d1-local and qwen-local and tfidf-lr for each task, then delete the smoke-test raw logs (results/raw/*) so the real run starts clean, and run charts/analyze on nothing gracefully.
