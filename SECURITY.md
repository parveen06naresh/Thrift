# Security

Thrift sits in front of your model API keys, so please report vulnerabilities privately
through GitHub's "Report a vulnerability" button on this repository rather than in a public issue.

Deployment notes:
- Set `[server] api_keys` when the gateway is reachable by anyone but you.
- Upstream keys are read from environment variables; prefer those over `api_key` in TOML.
- The cache and shadow log store prompts and answers. Put `cache.path` and `shadow.path` on
  storage you would trust with that data, or leave them unset to keep everything in memory.
