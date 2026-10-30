"""One-command reproduction (works on Windows, macOS and Linux).

    python run_all.py --split test --backend groq
"""
import argparse
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--backend", default="groq", choices=["groq", "mock"])
    ap.add_argument("--systems", nargs="*", default=None)
    args = ap.parse_args()

    if subprocess.call([sys.executable, "-m", "pytest", "-q"]) != 0:
        print("Tests failed; not running the evaluation.")
        return 1
    cmd = [sys.executable, "-m", "evaluation.run_eval", "--split", args.split, "--backend", args.backend]
    if args.systems:
        cmd += ["--systems", *args.systems]
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
