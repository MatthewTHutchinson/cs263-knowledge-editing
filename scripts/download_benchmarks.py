"""
Download external MQuAKE and RippleEdits benchmark files.

Usage:
    python scripts/download_benchmarks.py --dataset all
    python scripts/download_benchmarks.py --dataset mquake --mquake_variant cf3k-v2
    python scripts/download_benchmarks.py --dataset ripple --ripple_subset popular
"""

import argparse
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


MQUAKE_URLS = {
    "cf3k-v2": (
        "data/mquake/MQuAKE-CF-3k-v2.json",
        "https://raw.githubusercontent.com/princeton-nlp/MQuAKE/main/datasets/MQuAKE-CF-3k-v2.json",
    ),
    "cf3k-v1": (
        "data/mquake/MQuAKE-CF-3k.json",
        "https://raw.githubusercontent.com/princeton-nlp/MQuAKE/main/datasets/MQuAKE-CF-3k.json",
    ),
    "cf-full": (
        "data/mquake/MQuAKE-CF.json",
        "https://raw.githubusercontent.com/princeton-nlp/MQuAKE/main/datasets/MQuAKE-CF.json",
    ),
    "temporal": (
        "data/mquake/MQuAKE-T.json",
        "https://raw.githubusercontent.com/princeton-nlp/MQuAKE/main/datasets/MQuAKE-T.json",
    ),
}

RIPPLE_URLS = {
    "popular": (
        "data/ripple_edits/POPULAR.json",
        [
            "https://raw.githubusercontent.com/edenbiran/RippleEdits/main/data/benchmark/POPULAR.json",
            "https://raw.githubusercontent.com/edenbiran/RippleEdits/main/data/benchmark/popular.json",
        ],
    ),
    "random": (
        "data/ripple_edits/RANDOM.json",
        [
            "https://raw.githubusercontent.com/edenbiran/RippleEdits/main/data/benchmark/RANDOM.json",
            "https://raw.githubusercontent.com/edenbiran/RippleEdits/main/data/benchmark/random.json",
        ],
    ),
    "recent": (
        "data/ripple_edits/RECENT.json",
        [
            "https://raw.githubusercontent.com/edenbiran/RippleEdits/main/data/benchmark/RECENT.json",
            "https://raw.githubusercontent.com/edenbiran/RippleEdits/main/data/benchmark/recent.json",
        ],
    ),
}


def download_one(dest: Path, urls: list[str], overwrite: bool = False) -> None:
    if dest.exists() and not overwrite:
        print(f"exists  {dest}")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    last_error = None
    for url in urls:
        try:
            print(f"fetch   {url}")
            with urlopen(url, timeout=60) as response:
                tmp.write_bytes(response.read())
            os.replace(tmp, dest)
            print(f"wrote   {dest} ({dest.stat().st_size:,} bytes)")
            return
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            if tmp.exists():
                tmp.unlink()
    raise RuntimeError(f"failed to download {dest}: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["all", "mquake", "ripple"], default="all")
    parser.add_argument("--mquake_variant", choices=list(MQUAKE_URLS), default="cf3k-v2")
    parser.add_argument("--ripple_subset", choices=list(RIPPLE_URLS) + ["all"], default="all")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.dataset in ("all", "mquake"):
        dest, url = MQUAKE_URLS[args.mquake_variant]
        download_one(Path(dest), [url], args.overwrite)

    if args.dataset in ("all", "ripple"):
        subsets = RIPPLE_URLS.keys() if args.ripple_subset == "all" else [args.ripple_subset]
        for subset in subsets:
            dest, urls = RIPPLE_URLS[subset]
            download_one(Path(dest), urls, args.overwrite)


if __name__ == "__main__":
    main()
