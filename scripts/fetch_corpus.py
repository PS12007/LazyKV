"""Fetch the long-document corpus used for prompts and teacher-forced quality checks.

Public-domain Project Gutenberg texts, stripped of the Gutenberg header and footer. The
texts are not committed (see .gitignore). data/corpus/manifest.json records source URL,
SHA-256 of the raw download, and token counts under the model tokenizer, so any rerun can
verify it is scoring the same bytes.
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.results import REPO_ROOT  # noqa: E402

CORPUS_DIR = REPO_ROOT / "data" / "corpus"
MODEL_REPO = "unsloth/Llama-3.2-1B-Instruct"
MODEL_REVISION = "5a8abab4a5d6f164389b1079fb721cfab8d7126c"

BOOKS = {
    "pride_and_prejudice": "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
    "moby_dick": "https://www.gutenberg.org/cache/epub/2701/pg2701.txt",
}


def strip_gutenberg(text: str) -> str:
    start = text.find("*** START OF")
    end = text.find("*** END OF")
    if start == -1 or end == -1:
        raise ValueError("Gutenberg markers not found")
    body = text[text.find("\n", start) + 1 : end]
    return body.strip() + "\n"


def main() -> None:
    from transformers import AutoTokenizer

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL_REPO, revision=MODEL_REVISION)
    manifest: dict[str, object] = {"tokenizer": {"repo": MODEL_REPO, "revision": MODEL_REVISION}, "books": {}}
    for name, url in BOOKS.items():
        req = urllib.request.Request(url, headers={"User-Agent": "LazyKV research fetch (single download)"})
        raw = urllib.request.urlopen(req, timeout=60).read()
        body = strip_gutenberg(raw.decode("utf-8-sig"))
        (CORPUS_DIR / f"{name}.txt").write_text(body, encoding="utf-8", newline="\n")
        n_tokens = len(tok(body, add_special_tokens=False)["input_ids"])
        manifest["books"][name] = {  # type: ignore[index]
            "url": url,
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
            "body_chars": len(body),
            "body_tokens": n_tokens,
        }
        print(f"{name}: {n_tokens:,} tokens")
    (CORPUS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
