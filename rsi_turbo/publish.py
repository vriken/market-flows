"""Copy your journal to the gh-pages branch so the GitHub page shows your trades.

Works in its own worktree of gh-pages at market-flows/.gh-pages, so your checkout and branch are never touched. The
page picks the journal up at its next scheduled build. The dashboard runs this in the background after every save.
"""
import shutil
import subprocess
from pathlib import Path

import scan

REPO = Path(__file__).resolve().parent.parent
PAGES = REPO / ".gh-pages"
TARGET = PAGES / "rsi-turbo" / "data" / "journal.csv"
ATTEMPTS = 2


def git(*args, cwd=PAGES):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def sync_to_remote():
    git("fetch", "-q", "origin", "gh-pages", cwd=REPO)
    if not PAGES.exists():
        git("worktree", "add", "-q", "--detach", str(PAGES), "origin/gh-pages", cwd=REPO)
    git("reset", "-q", "--hard", "origin/gh-pages")


def main():
    for attempt in range(1, ATTEMPTS + 1):
        sync_to_remote()
        TARGET.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(scan.JOURNAL, TARGET)
        git("add", str(TARGET.relative_to(PAGES)))
        if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=PAGES).returncode == 0:
            print("journal already published")
            return
        git("commit", "-q", "-m", "Update RSI Turbo journal")
        try:
            git("push", "-q", "origin", "HEAD:gh-pages")
            print("journal published")
            return
        except subprocess.CalledProcessError as e:
            if attempt == ATTEMPTS:
                raise
            print(f"push rejected, retrying on the latest gh-pages: {e.stderr.strip()}")


if __name__ == "__main__":
    main()
