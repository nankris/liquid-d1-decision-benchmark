# 001 - CPU experiment issues (10 Oct 2026)

**1. Hosted d1 API not testable.** No `LIQUID_API_KEY` on the test machine; the OpenRouter key returned HTTP 401 "User not found". All results are from the open-weights d1-3B GGUF run locally. Hosted run steps are in `PENDING_HOSTED.md` (expected cost ~$0.007).

**2. Qwen letter-option confusion on routing.** With A-K letter options Qwen2.5-3B scored 60%. Re-run with team-name answers scored 85.5%. Both are reported so the comparison isn't unfair to Qwen.

**3. Routing category descriptions were wrong in the first draft.** Codex's draft described `CANCEL` as cancelling an order; in this dataset it means cancellation fees. Category descriptions were rewritten to match the labels before the reported run.

**4. Chart title overclaimed.** Draft chart said "d1 more accurate on 4 of 4 tasks". Toxicity CIs overlap, so it's 3 of 4 wins plus a tie. Titles now only claim wins where 95% bootstrap CIs don't overlap.

**5. llama.cpp server crashed on multi-question requests with 4 parallel slots.** Running with `-np 1` fixed it. Mentioned in the post's caveats; not root-caused.

**6. Two different ECE numbers for the same model.** `summary.csv` reports top-label ECE (d1 spam 0.032); the calibration chart/post use 5-bin ECE on P(spam) (d1 spam 0.014). README now states which is which. Pick one definition for the final write-up.

**7. Cost numbers are projections, not bills.** Chat-model costs reuse Qwen's token counts and ignore reasoning tokens, so they are lower bounds.

**8. No speed advantage on CPU.** Liquid's 200-300 ms claim is for hosted GPUs; on an 8-thread CPU d1 was not faster than Qwen (except review stars). Motivated the Colab GPU test.
