import argparse
import os
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    url: str
    out_html: str
    out_ss: str
    out_metrics: str
    timeout_ms: int
    preset: str
    bot_mjs: str


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def parse_args() -> Config:
    ap = argparse.ArgumentParser(description="Wrapper to run bot.mjs with Node.")
    ap.add_argument("--url", type=str, required=True)
    ap.add_argument("--out-html", type=str, required=True)
    ap.add_argument("--out-ss", type=str, required=True)
    ap.add_argument("--out-metrics", type=str, required=True)
    ap.add_argument("--timeout-ms", type=int, default=60000)
    ap.add_argument("--preset", type=str, default="lighthouse-desktop")
    ap.add_argument("--bot-mjs", type=str, default="bot.mjs", help="Path to bot.mjs")
    a = ap.parse_args()

    return Config(
        url=str(a.url),
        out_html=str(a.out_html),
        out_ss=str(a.out_ss),
        out_metrics=str(a.out_metrics),
        timeout_ms=int(a.timeout_ms),
        preset=str(a.preset),
        bot_mjs=str(a.bot_mjs),
    )


def main() -> None:
    cfg = parse_args()

    ensure_dir(os.path.dirname(cfg.out_html))
    ensure_dir(os.path.dirname(cfg.out_ss))
    ensure_dir(os.path.dirname(cfg.out_metrics))

    if not os.path.exists(cfg.bot_mjs):
        raise SystemExit(f"bot.mjs not found at: {cfg.bot_mjs}")

    cmd = [
        "node",
        cfg.bot_mjs,
        "--url",
        cfg.url,
        "--out-html",
        cfg.out_html,
        "--out-ss",
        cfg.out_ss,
        "--out-metrics",
        cfg.out_metrics,
        "--timeout-ms",
        str(cfg.timeout_ms),
        "--preset",
        cfg.preset,
    ]

    p = subprocess.run(cmd)
    raise SystemExit(int(p.returncode))


if __name__ == "__main__":
    main()
