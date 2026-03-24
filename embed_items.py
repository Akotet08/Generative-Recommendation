"""
Create item embeddings with Sentence-T5 for the RQ-VAE stage.

This script follows the paper's item-text construction closely:
- title
- price
- brand
- category

It writes a torch artifact compatible with main.py:
{
    "item_ids": [...],
    "embeddings": <torch.FloatTensor [num_items, 768]>,
    "texts": [...],
    "model_name": "...",
}
"""

import argparse
import ast
import gzip
import html
import json
import sys
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Embed item metadata with Sentence-T5.")
    parser.add_argument("--items-path", type=Path, default=Path("data/items.txt"))
    parser.add_argument("--metadata-path", type=Path, default=Path("data/metadata_subset.json"))
    parser.add_argument("--output-path", type=Path, default=Path("artifacts/item_embeddings.pt"))
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/sentence-t5-xl",
        help="Closest public match to the paper's 768-dim Sentence-T5 encoder.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-seq-length", type=int, default=256)
    return parser.parse_args()


def load_item_ids(items_path):
    with open(items_path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def open_text(path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def detect_metadata_format(metadata_path):
    with open_text(metadata_path) as handle:
        preview = handle.read(256).lstrip()
    if preview.startswith("{'"):
        return "amazon_lines"
    return "json"


def load_metadata_subset(metadata_path, wanted_item_ids):
    """Load only the requested items from either subset JSON or raw Amazon metadata."""
    wanted = set(wanted_item_ids)
    format_name = detect_metadata_format(metadata_path)

    if format_name == "json":
        with open_text(metadata_path) as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            return {item_id: payload[item_id] for item_id in wanted_item_ids if item_id in payload}
        raise ValueError("Expected metadata JSON to be an object keyed by item_id.")

    metadata_by_asin = {}
    with open_text(metadata_path) as handle:
        for line in handle:
            if len(metadata_by_asin) == len(wanted):
                break
            line = line.strip()
            if not line:
                continue
            record = ast.literal_eval(line)
            asin = record.get("asin")
            if asin in wanted:
                metadata_by_asin[asin] = record
    return {item_id: metadata_by_asin[item_id] for item_id in wanted_item_ids if item_id in metadata_by_asin}


def normalize_text(value):
    text = html.unescape(str(value))
    return " ".join(text.split())


def format_categories(categories):
    if not categories:
        return None

    paths = []
    for path in categories:
        if not path:
            continue
        cleaned = [normalize_text(node) for node in path if node]
        if cleaned:
            paths.append(" > ".join(cleaned))
    if not paths:
        return None
    return " | ".join(dict.fromkeys(paths))


def build_item_text(record):
    """Construct the item sentence used for Sentence-T5 encoding."""
    parts = []

    title = record.get("title")
    if title:
        parts.append(f"title: {normalize_text(title)}")

    price = record.get("price")
    if price not in (None, ""):
        parts.append(f"price: {normalize_text(price)}")

    brand = record.get("brand")
    if brand:
        parts.append(f"brand: {normalize_text(brand)}")

    categories = format_categories(record.get("categories"))
    if categories:
        parts.append(f"category: {categories}")

    asin = record.get("asin")
    if not parts and asin:
        parts.append(f"item: {asin}")

    return ". ".join(parts)


def load_sentence_transformer(model_name, device, max_seq_length):
    project_root = str(Path(__file__).resolve().parent)
    original_sys_path = list(sys.path)
    sys.path = [path for path in sys.path if path not in ("", project_root)]
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        sys.path = original_sys_path
        raise ImportError(
            "Failed to import sentence-transformers. If it is installed, check for "
            "local module name collisions such as a file named `transformers.py` in "
            "this project."
        ) from exc
    sys.path = original_sys_path

    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = max_seq_length

    modules = list(model._modules.values())
    if modules and modules[-1].__class__.__name__ == "Normalize":
        model = SentenceTransformer(modules=modules[:-1], device=device)
        model.max_seq_length = max_seq_length
        print("[embed_items] Removed terminal Normalize module from SentenceTransformer.")

    return model


def main():
    args = parse_args()

    item_ids = load_item_ids(args.items_path)
    metadata_by_asin = load_metadata_subset(args.metadata_path, item_ids)

    missing_items = [item_id for item_id in item_ids if item_id not in metadata_by_asin]
    if missing_items:
        preview = ", ".join(missing_items[:5])
        print(
            f"Warning: missing metadata for {len(missing_items)} items. "
            f"First missing items: {preview}"
        )

    kept_item_ids = [item_id for item_id in item_ids if item_id in metadata_by_asin]
    texts = [build_item_text(metadata_by_asin[item_id]) for item_id in kept_item_ids]
    if not texts:
        raise ValueError("No item metadata was found for the requested items.")

    model = load_sentence_transformer(args.model_name, args.device, args.max_seq_length)
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        convert_to_tensor=True,
        normalize_embeddings=False,
        show_progress_bar=True,
    ).cpu()

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "item_ids": kept_item_ids,
            "embeddings": embeddings.float(),
            "texts": texts,
            "model_name": args.model_name,
        },
        args.output_path,
    )
    norms = embeddings.norm(dim=1)
    print(
        f"[embed_items] norm mean={norms.mean().item():.6f} "
        f"std={norms.std().item():.6f} "
        f"min={norms.min().item():.6f} "
        f"max={norms.max().item():.6f}"
    )
    print(f"Saved {len(kept_item_ids)} item embeddings to {args.output_path}")


if __name__ == "__main__":
    main()
