# Sources (checked 10 Oct 2026)

## Primary (Liquid AI)
- Launch post, "Introducing d1: The most capable decision model, now with vision" (5 Oct 2026): https://www.liquid.ai/blog/d1-decision-model
  - Claims: one forward pass, no generated tokens; "a text decision takes 200 to 300 ms"; "costs 19x to 200x less" than GPT-6.1 Sol and Claude Opus 5.5 on six of Liquid's own applications; "matches or beats GPT-6.1 Sol on four of them"; $0.04 per 1M input tokens, no output tokens; each question billed as its own prompt; also on Vercel and OpenRouter (text only).
  - Methodology note: Liquid's comparison ran each app once per model on 5 Oct 2026 with the d1 Playground comparison script; chat models answered in JSON at default reasoning, sometimes batched; list prices without cache discounts.
- Open-weights post, "Open d1: Edge decision models for text, vision, and audio" (7 Oct 2026): https://www.liquid.ai/blog/d1-open
- Hugging Face blog mirror: https://huggingface.co/blog/LiquidAI/open-d1
- d1-3B model card (benchmarks, latency table, usage): https://huggingface.co/LiquidAI/d1-3B
- d1-3B GGUF (llama.cpp, /v1/systemone endpoint): https://huggingface.co/LiquidAI/d1-3B-GGUF
- License (LFM Open License v1.0; commercial-use threshold defined as US$10M annual revenue): https://huggingface.co/LiquidAI/d1-3B-GGUF/blob/main/LICENSE
- Docs, hosted d1 API (setup, request/response format, Noul/Choice/Score, d1 vs d1:free, image billing): https://docs.liquid.ai/lfm/models/d1
- Docs, Decision Models overview: https://docs.liquid.ai/lfm/models/decision-models
- Docs, Decision Model Guide (LLM → d1 migration): https://docs.liquid.ai/guides/decision-model-guide
- Agent skill file (mentions `d1:free` model id): https://docs.liquid.ai/.well-known/skills/decision-model/skill.md
- Console / API keys: https://console.liquid.ai (Dashboard > API Keys; keys prefixed `liquid_`)

## Third-party
- OpenRouter model page, liquid/d1 ($0.04/M input, $0 output; 65,536 context; P50 0.32 s, P95 ~0.88 s at time of check): https://openrouter.ai/liquid/d1
- OpenRouter System One API reference: https://openrouter.ai/docs/api/api-reference/systemone/submit-a-system-one-request
- llama.cpp release used (b11541): https://github.com/ggml-org/llama.cpp/releases/tag/b11541
- Baseline model: Qwen2.5-3B-Instruct GGUF: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF
- Third-party review noting no published limits for `d1:free` and data-use terms (not independently verified): https://www.anchorterminal.com/tools/liquid-d1

## Datasets
- SMS Spam Collection (UCI), HF `ucirvine/sms_spam`: https://huggingface.co/datasets/ucirvine/sms_spam
- TweetEval offensive, HF `cardiffnlp/tweet_eval` (config `offensive`): https://huggingface.co/datasets/cardiffnlp/tweet_eval
- Bitext customer support (synthetic), HF `bitext/Bitext-customer-support-llm-chatbot-training-dataset`: https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset
- Yelp Review Full, HF `Yelp/yelp_review_full`: https://huggingface.co/datasets/Yelp/yelp_review_full
