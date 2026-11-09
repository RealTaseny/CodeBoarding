import os
import subprocess
import sys


def test_cli_starts_with_generic_socks_proxy():
    env = {
        key: value
        for key, value in os.environ.items()
        if key.lower() not in {"http_proxy", "https_proxy", "all_proxy", "no_proxy"}
    }
    env["all_proxy"] = "socks://127.0.0.1:7897/"
    process = subprocess.run(
        [sys.executable, "-m", "codeboarding_cli.main", "full", "--help"],
        capture_output=True,
        env=env,
        text=True,
        timeout=30,
    )
    assert process.returncode == 0, process.stderr
    assert "--local" in process.stdout
