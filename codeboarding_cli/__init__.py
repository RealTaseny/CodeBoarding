"""Command-line entry points for CodeBoarding."""

from infra.proxy import normalize_socks_proxy_urls

# The Ollama SDK builds an HTTPX client when the CLI imports the LLM registry.
normalize_socks_proxy_urls()
