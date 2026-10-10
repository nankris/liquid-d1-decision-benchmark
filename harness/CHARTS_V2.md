# Chart polish brief (v2) — edit harness/charts.py (keep it reading ONLY logged data in results/)
Friendly model names everywhere: d1-local -> "d1-3B (open weights, local)", qwen-local -> "Qwen2.5-3B-Instruct (chat, local)",
tfidf-lr -> "TF-IDF + logistic regression (trained)", qwen-local-names -> "Qwen2.5-3B, label-name prompt".
Keep palette (d1 #5B3FD9, qwen #F28C28, tfidf #9AA0A6). Titles = the takeaway sentence, computed from data where a number appears.
Subtitle grey: "n=200 per task · 4-bit GGUF on 8 CPU threads · run 10 Oct 2026". Footer tiny grey: "Source: own runs, logs in results/raw".
Width 1600-2000px at dpi 200, generous padding, no overlapping labels.

1. accuracy_by_task.png: grouped bars per task (spam, toxicity, ticket routing, review stars), value labels placed ABOVE the CI whisker (no overlap).
   For routing add a 4th bar: Qwen label-name variant, computed from results/raw/qwen-local-names__routing.jsonl (accuracy = pred==label), same orange with hatch '//' and lighter alpha, labelled in legend. Legend below the plot, 2 columns. Nice task labels.
2. reliability.png (replaces reliability_spam/toxic): 1x2 panels (spam, toxicity), 5 equal-width bins, only plot bins with >=5 items, marker area ∝ bin count, diagonal "perfect calibration", ECE (from 5 bins, computed here) in legend for d1, qwen, tfidf. Below each panel a small stacked histogram strip of predicted P(yes) for d1 only? -> skip; keep simple.
   Delete the old reliability_spam.png / reliability_toxic.png outputs.
3. latency.png: per task, d1 vs qwen only, horizontal bars for p50 with a thin whisker to p95, LINEAR ms axis, value labels "p50 256 ms / p95 441 ms". Note in subtitle: "TF-IDF baseline: <1 ms (not shown)".
4. cost_per_1k.png: from results/cost_projection.csv. For the spam task and routing task (two panels, or one panel averaged across tasks — choose averaged across 4 tasks, one bar per model), horizontal bars, LOG x axis, USD per 1,000 decisions with labels like "$0.0053 · 1x", "$0.39 · 74x". Title like "Projected API cost per 1,000 decisions (list prices)". Subtitle: "d1 tokens logged from d1-3B; chat models use logged Qwen prompt tokens + 2 output tokens at OpenRouter list prices on 10 Oct 2026. Lower bound for chat (no reasoning tokens)." d1 bar purple, others grey shades, highlight none.
5. multiq.png: keep, friendly title "Asking more questions per request: latency and tokens grow roughly linearly (local llama.cpp)"; annotate points with values; also draw a dashed reference "k × single-question latency".
6. confusion_routing.png: side by side d1 vs qwen (letter prompt) row-normalised, same colormap (Purples for d1, Oranges for qwen), with accuracies in panel titles. Delete routing_confusion_d1.png / routing_confusion_qwen.png.
7. rating_scatter.png: keep, friendly titles with MAE and Spearman from results/summary.json in panel titles.
8. confidence_hist.png: d1 P(spam) histogram split by true label (ham vs spam), 20 bins, purple/grey; title "d1's spam probabilities are mostly near 0 or 1".
9. NEW hero.png (1600x900): a clean summary card: left big text "d1-3B vs a same-size chat model", 4 rows (task | d1 acc | Qwen acc | TF-IDF acc) as a styled table drawn with matplotlib, plus a line "d1 projected hosted cost: $X per 1,000 decisions" (mean across tasks from cost_projection.csv). All numbers read from results/summary.json and cost_projection.csv.

Then run: python3 harness/charts.py and list images/. Do not touch results/raw. Do not call any network endpoints.

# Terminal screenshots (use harness/render_terminal.py; save under images/screens/)
Servers are running on 127.0.0.1:8081 (d1) and 8082 (qwen). Make these, capturing REAL command output to a txt file first (results/screens_src/*.txt), then render:
 a. screen_server_start.png: the first ~12 lines of /tmp/d1server.log (llama.cpp loading d1, "decision model type: lfm2-d1"). Prepend the command line used: "$ llama-server -m d1-3B-Q4_K_M.gguf -c 4096 -t 8 -np 1 --port 8081".
 b. screen_curl_three_questions.png: run a curl to http://127.0.0.1:8081/v1/systemone with the state "I was charged twice this month and the app keeps crashing when I try to download my invoice. Please fix this today." and three questions: team (choice: billing/technical/account/shipping with short descriptions), wants_refund (noul), urgency (score: ["Can wait","Today","Blocking the customer now"]). Show the command (pretty, multi-line) and the response piped through python3 -m json.tool, rounded? no — raw values. Also print "time_total: X s" via curl -w.
 c. screen_qwen_vs_d1.png: show the same spam SMS sent to both: pick the item from results/raw where d1 is right and qwen is wrong with highest d1 confidence (search logs), and display: the SMS text, true label, d1 P(spam), qwen text answer and its P(yes). Pull from logs (do not re-run).
 d. screen_summary.png: render results/summary_stdout.txt (first table only).
 e. screen_code.png: render the core of the d1 request code in harness/models.py (HTTPClient.body + call), titled "harness/models.py".
