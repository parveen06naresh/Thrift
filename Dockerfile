FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY thrift_llm ./thrift_llm
RUN pip install --no-cache-dir ".[tiktoken]" && python -c "import tiktoken; tiktoken.get_encoding('o200k_base')"
EXPOSE 8787
# Mount your config at /app/thrift.toml (see thrift.example.toml) and pass API keys as env vars.
CMD ["thrift-llm", "serve", "--config", "/app/thrift.toml", "--host", "0.0.0.0", "--port", "8787"]
