"""Copy your journal to the gh-pages branch and rebuild the GitHub page so it shows your trades within minutes.

Works in its own worktree of gh-pages at market-flows/.gh-pages, so your checkout and branch are never touched. The
rebuild is started through GitHub's API with the token git keeps in the keychain for the account in the origin URL.
The dashboard runs this in the background after every save; failures go to live/publish.log.
"""
import json
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

import scan

REPO = Path(__file__).resolve().parent.parent
PAGES = REPO / ".gh-pages"
TARGET = PAGES / "rsi-turbo" / "data" / "journal.csv"
WORKFLOW = "rsi-turbo.yaml"
ATTEMPTS = 2


def git(*args, cwd=PAGES, **kwargs):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True, **kwargs)


def sync_to_remote():
    git("fetch", "-q", "origin", "gh-pages", cwd=REPO)
    if not PAGES.exists():
        git("worktree", "add", "-q", "--detach", str(PAGES), "origin/gh-pages", cwd=REPO)
    git("reset", "-q", "--hard", "origin/gh-pages")


def push_journal():
    """True when a new journal was pushed."""
    for attempt in range(1, ATTEMPTS + 1):
        sync_to_remote()
        TARGET.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(scan.JOURNAL, TARGET)
        git("add", str(TARGET.relative_to(PAGES)))
        if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=PAGES).returncode == 0:
            return False
        git("commit", "-q", "-m", "Update RSI Turbo journal")
        try:
            git("push", "-q", "origin", "HEAD:gh-pages")
            return True
        except subprocess.CalledProcessError as e:
            if attempt == ATTEMPTS:
                raise
            print(f"push rejected, retrying on the latest gh-pages: {e.stderr.strip()}")
    return False


def rebuild_page():
    origin = urlsplit(git("remote", "get-url", "origin", cwd=REPO).stdout.strip())
    repo = origin.path.strip("/").removesuffix(".git")
    query = f"protocol=https\nhost={origin.hostname}\nusername={origin.username}\n\n"
    credential = git("credential", "fill", cwd=REPO, input=query, env={**os.environ, "GIT_TERMINAL_PROMPT": "0"}).stdout
    token = next(line.split("=", 1)[1] for line in credential.splitlines() if line.startswith("password="))
    request = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/actions/workflows/{WORKFLOW}/dispatches", method="POST",
        data=json.dumps({"ref": "main", "inputs": {"step": "refresh"}}).encode(),
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(request, timeout=30):
        pass


def main():
    if not push_journal():
        print("journal already published")
        return
    rebuild_page()
    print("journal published, page rebuild started")


if __name__ == "__main__":
    main()
