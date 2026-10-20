# Changelog

## 0.2.0
- One endpoint for OpenAI, Anthropic (native Messages API with prompt caching), Google Gemini,
  xAI Grok, Perplexity, DeepSeek, Mistral, Moonshot Kimi, Groq, Together, Fireworks,
  OpenRouter, Ollama and vLLM; per-tier fallback chains.
- Ladder: certified routing to the cheapest of N models; bundled ladders for HELM Lite
  (2024 models) and HELM Capabilities (GPT-5.1, Gemini 3 Pro, Grok 4, Claude 4.5, ...).
- Drift guard: anytime-valid e-process over shadow traffic, safe mode on alarm.
- `Thrift` / `AsyncThrift`: OpenAI-SDK-shaped in-process client. `thrift-llm serve --tier`,
  `thrift-llm models`.
- Claude 5 prices; provider prompt-cache discounts in the ledger.

## 0.1.0
- Gateway with semantic cache, compression and two-model routing, each certified with
  Learn-then-Test; ledger, dashboard, Prometheus metrics.
