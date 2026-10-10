# Hosted d1 run — pending (not executed)

Status on 10 Oct 2026: no LIQUID_API_KEY on the box; OPENROUTER_API_KEY present but rejected by OpenRouter (HTTP 401 "User not found") on /api/v1/key, /api/v1/chat/completions and /api/v1/systemone. No paid or hosted calls were made.

To run it, use either option:
A) Liquid console (direct): sign up at https://console.liquid.ai → Dashboard → API Keys (key starts with `liquid_`).
   Text-only `d1:free` model exists (no published limits); paid `d1` is $0.04 / 1M input tokens.
   export LIQUID_API_KEY=liquid_...
   LIQUID_MODEL=d1:free python3 harness/run.py --model d1-hosted --all      # or leave LIQUID_MODEL unset for paid d1
B) OpenRouter (liquid/d1, text only, $0.04/M input): put a valid key with credit in OPENROUTER_API_KEY, then
   python3 harness/run.py --model openrouter-d1 --all
   Optional frontier chat baseline (add its price to harness/prices.py first; budget guard MAX_USD defaults to $2):
   python3 harness/run.py --model openrouter:anthropic/claude-haiku-5.5 --all

Expected spend for all 800 d1 decisions: ~170k input tokens ≈ $0.007 at list price.
Then: python3 harness/analyze.py && python3 harness/charts.py && python3 harness/cost_projection.py
