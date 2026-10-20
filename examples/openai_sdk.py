"""Use Thrift from the official OpenAI SDK: only the base_url changes.

    thrift-llm serve --demo        # or: thrift-llm serve --config thrift.toml
    python examples/openai_sdk.py
"""
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8787/v1", api_key="unused-unless-server-api_keys-is-set")

for q in ["What is the capital of Australia?", "what is the capital of australia"]:
    raw = client.chat.completions.with_raw_response.create(model="thrift-auto", messages=[{"role": "user", "content": q}])
    r = raw.parse()
    print(r.choices[0].message.content)
    print("  tier:", raw.headers.get("x-thrift-tier"), "| cache:", raw.headers.get("x-thrift-cache"),
          "| cost $", raw.headers.get("x-thrift-cost-usd"), "vs $", raw.headers.get("x-thrift-baseline-usd"))
