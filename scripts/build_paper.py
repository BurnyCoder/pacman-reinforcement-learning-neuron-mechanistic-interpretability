"""Build paper/paper.pdf from paper/paper.md with pandoc + xelatex (run after `--phase report`).

Usage: uv run python scripts/build_paper.py
Needs pandoc (https://pandoc.org) and a xelatex engine (MiKTeX/TeX Live). A Unicode system font (Cambria on
Windows, DejaVu on Linux) is used so that symbols such as ≥, →, ², × and − render.
"""

import platform
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"


def main() -> None:
    pandoc = shutil.which("pandoc")
    if pandoc is None:
        raise SystemExit("pandoc not found: install from https://pandoc.org/installing.html")
    font = "Cambria" if platform.system() == "Windows" else "DejaVu Serif"
    cmd = [pandoc, "paper.md", "-o", "paper.pdf", "--pdf-engine=xelatex", "--resource-path=.:..",
           "-V", f"mainfont={font}", "-V", "geometry:margin=2.2cm", "-V", "fontsize=10pt", "-V", "colorlinks=true",
           "-V", "linkcolor=blue", "-V", "urlcolor=blue"]  # headings carry their own numbers
    subprocess.run(cmd, cwd=PAPER, check=True)
    print(f"wrote {PAPER / 'paper.pdf'}")


if __name__ == "__main__":
    main()
