import os


def normalize_socks_proxy_urls() -> None:
    """Translate generic SOCKS URLs for Python HTTP clients in this process."""
    for name in tuple(os.environ):
        if name.lower() in {"http_proxy", "https_proxy", "all_proxy"}:
            value = os.environ[name]
            if value.lower().startswith("socks://"):
                os.environ[name] = "socks5h://" + value[len("socks://") :]
