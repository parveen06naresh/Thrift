"""Use Thrift's pieces as a library, without running a server."""
import asyncio

from thrift_llm import Compressor, SemanticCache
from thrift_llm.config import demo_config
from thrift_llm.gateway import Gateway

# 1. Compression on its own
doc = "\n\n".join(f"Paragraph {i} is about shipping logistics and port schedules." for i in range(80))
msgs = [{"role": "user", "content": f"{doc}\n\nThe launch date is 14 March.\n\n{doc}\n\nWhen is the launch date?"}]
res = Compressor(keep_ratio=0.3).compress(msgs)
print(f"compression: {res.tokens_before} -> {res.tokens_after} tokens, stages {res.stages}")

# 2. Cache on its own
cache = SemanticCache()
scope = {"model": "gpt-4o-mini", "system": "You are helpful."}
cache.put(scope, "How do I reset my password?", {"answer": "Settings > Security > Reset"})
print("cache:", cache.get(scope, "how do i reset my password"))


# 3. The full pipeline in-process (mock tiers; use config.load("thrift.toml") for real ones)
async def main():
    gw = Gateway(demo_config())
    r = await gw.chat({"model": "thrift-auto", "messages": [{"role": "user", "content": "Explain TCP slow start."}]})
    print("gateway:", r.headers["x-thrift-tier"], r.response["choices"][0]["message"]["content"])
    await gw.aclose()


asyncio.run(main())
