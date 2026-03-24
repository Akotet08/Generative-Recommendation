"""
Visualize codebook usage and collision bucket statistics from a trained RQ-VAE artifact.

The script expects the artifact written by main.py:
    artifacts/rqvae.pt

It uses:
- state_dict to infer the number of residual codebooks and codebook size
- semantic_ids to compute which codes were used by the trained item set
- base Semantic ID tuples to measure collision bucket sizes
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize RQ-VAE codebook usage.")
    parser.add_argument("--artifact-path", type=Path, default=Path("artifacts/rqvae.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/codebook_usage"))
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def load_artifact(artifact_path):
    artifact = torch.load(artifact_path, map_location="cpu")
    if "state_dict" not in artifact or "semantic_ids" not in artifact:
        raise ValueError(
            "Artifact must contain 'state_dict' and 'semantic_ids'. "
            "Expected the file saved by main.py."
        )
    return artifact


def infer_codebook_config(state_dict):
    codebook_keys = sorted(
        key for key in state_dict.keys() if key.startswith("codebooks.") and key.endswith(".weight")
    )
    if not codebook_keys:
        raise ValueError("Could not infer codebook weights from state_dict.")

    num_codebooks = len(codebook_keys)
    codebook_size = state_dict[codebook_keys[0]].shape[0]
    return num_codebooks, codebook_size


def compute_usage_stats(semantic_ids, num_codebooks, codebook_size):
    base_ids = semantic_ids[:, :num_codebooks].long()
    total_possible_base_slots = codebook_size ** num_codebooks
    usage_counts = []
    summaries = []

    for level in range(num_codebooks):
        counts = torch.bincount(base_ids[:, level], minlength=codebook_size)
        probs = counts.float() / max(counts.sum().item(), 1)
        active_codes = int((counts > 0).sum().item())
        entropy = float(-(probs[probs > 0] * probs[probs > 0].log()).sum().item())
        perplexity = float(math.exp(entropy))

        usage_counts.append(counts)
        summaries.append(
            {
                "level": level,
                "active_codes": active_codes,
                "usage_ratio": active_codes / codebook_size,
                "perplexity": perplexity,
                "max_count": int(counts.max().item()),
                "min_nonzero_count": int(counts[counts > 0].min().item()) if active_codes else 0,
            }
        )

    collision_summary = None
    collision_bucket_hist = None
    if semantic_ids.size(1) > num_codebooks:
        _, bucket_sizes = torch.unique(base_ids, dim=0, return_counts=True)
        collision_bucket_sizes = bucket_sizes[bucket_sizes > 1]

        bucket_histogram = {}
        if collision_bucket_sizes.numel() > 0:
            collision_bucket_hist = torch.bincount(collision_bucket_sizes)
            bucket_histogram = {
                str(bucket_size): int(num_buckets)
                for bucket_size, num_buckets in enumerate(collision_bucket_hist.tolist())
                if bucket_size >= 2 and num_buckets > 0
            }

        collision_summary = {
            "total_possible_base_slots": int(total_possible_base_slots),
            "occupied_base_slots": int(bucket_sizes.numel()),
            "empty_base_slots": int(total_possible_base_slots - bucket_sizes.numel()),
            "occupied_base_slot_ratio": (
                float(bucket_sizes.numel()) / float(total_possible_base_slots)
                if total_possible_base_slots > 0
                else 0.0
            ),
            "unique_base_tuples": int(bucket_sizes.numel()),
            "singleton_buckets": int((bucket_sizes == 1).sum().item()),
            "collision_buckets": int((bucket_sizes > 1).sum().item()),
            "collision_slot_ratio_of_all_slots": (
                float((bucket_sizes > 1).sum().item()) / float(total_possible_base_slots)
                if total_possible_base_slots > 0
                else 0.0
            ),
            "collision_slot_ratio_of_occupied_slots": (
                float((bucket_sizes > 1).sum().item()) / float(bucket_sizes.numel())
                if bucket_sizes.numel() > 0
                else 0.0
            ),
            "items_in_collision_buckets": int(collision_bucket_sizes.sum().item()),
            "collision_item_ratio": (
                float(collision_bucket_sizes.sum().item()) / float(base_ids.size(0))
                if base_ids.size(0) > 0
                else 0.0
            ),
            "extra_items_due_to_collision": int(
                (collision_bucket_sizes - 1).sum().item()
            ) if collision_bucket_sizes.numel() > 0 else 0,
            "extra_items_ratio": (
                float((collision_bucket_sizes - 1).sum().item()) / float(base_ids.size(0))
                if base_ids.size(0) > 0 and collision_bucket_sizes.numel() > 0
                else 0.0
            ),
            "max_bucket_size": int(bucket_sizes.max().item()),
            "mean_collision_bucket_size": (
                float(collision_bucket_sizes.float().mean().item())
                if collision_bucket_sizes.numel() > 0
                else 0.0
            ),
            "bucket_size_histogram": bucket_histogram,
        }

    return usage_counts, summaries, collision_bucket_hist, collision_summary


def _lorenz_curve(counts):
    """Return (x, y) for a Lorenz curve of code usage.
    x = fraction of codes (sorted ascending), y = fraction of items covered."""
    sorted_counts = torch.sort(counts)[0].float()
    cum = torch.cumsum(sorted_counts, dim=0)
    total = cum[-1].clamp_min(1.0)
    x = torch.linspace(0, 1, len(counts)).numpy()
    y = (cum / total).numpy()
    return x, y


def _gini(counts):
    """Gini coefficient — 0 = perfectly uniform, 1 = fully collapsed."""
    n = len(counts)
    if n == 0:
        return 0.0
    sorted_c = torch.sort(counts)[0].float()
    index = torch.arange(1, n + 1, dtype=torch.float)
    return float(((2 * index - n - 1) * sorted_c).sum() / (n * sorted_c.sum().clamp_min(1e-9)))


def plot_usage(usage_counts, summaries, collision_bucket_hist, collision_summary, output_dir):
    """Produce a single comprehensive diagnostic figure."""
    import numpy as np

    num_codebooks = len(usage_counts)
    codebook_size = len(usage_counts[0])
    # 3 columns: [bar chart | Lorenz curve | sorted usage rank plot]
    # plus 1 extra row for cross-level heatmap and collision panel
    n_rows = num_codebooks + 1
    n_cols = 3
    fig = plt.figure(figsize=(18, 4 * n_rows))
    fig.suptitle("RQ-VAE Codebook Diagnostic", fontsize=14, fontweight="bold", y=1.01)

    uniform_per_code = len(usage_counts[0].numpy()) and usage_counts[0].sum().item() / codebook_size

    for level, (counts, summary) in enumerate(zip(usage_counts, summaries)):
        counts_np = counts.numpy()
        active = summary["active_codes"]
        gini = _gini(counts)

        # ── Column 0: bar chart with dead-code highlighting ──────────────────
        ax_bar = fig.add_subplot(n_rows, n_cols, level * n_cols + 1)
        dead_mask = counts_np == 0
        colors = ["#d73027" if d else "#4393c3" for d in dead_mask]
        ax_bar.bar(range(codebook_size), counts_np, width=1.0, color=colors, linewidth=0)
        ax_bar.axhline(uniform_per_code, color="black", linestyle="--", linewidth=0.8,
                       label=f"Uniform ({uniform_per_code:.1f})")
        ax_bar.set_title(
            f"Level {level}  |  active {active}/{codebook_size} ({summary['usage_ratio']:.0%})  "
            f"|  perplexity {summary['perplexity']:.1f}  |  Gini {gini:.3f}",
            fontsize=9,
        )
        ax_bar.set_xlabel("Code index", fontsize=8)
        ax_bar.set_ylabel("Items assigned", fontsize=8)
        ax_bar.legend(fontsize=7)
        ax_bar.tick_params(labelsize=7)
        # dead code count annotation
        n_dead = int(dead_mask.sum())
        if n_dead:
            ax_bar.text(0.99, 0.97, f"{n_dead} dead (red)", transform=ax_bar.transAxes,
                        ha="right", va="top", fontsize=7, color="#d73027")

        # ── Column 1: Lorenz curve ───────────────────────────────────────────
        ax_lorenz = fig.add_subplot(n_rows, n_cols, level * n_cols + 2)
        lx, ly = _lorenz_curve(counts)
        ax_lorenz.plot(lx, ly, color="#4393c3", linewidth=1.5, label="actual")
        ax_lorenz.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="perfect uniform")
        ax_lorenz.fill_between(lx, ly, lx, alpha=0.15, color="#d73027", label=f"Gini={gini:.3f}")
        ax_lorenz.set_title(f"Level {level}  Lorenz curve", fontsize=9)
        ax_lorenz.set_xlabel("Fraction of codes (sorted)", fontsize=8)
        ax_lorenz.set_ylabel("Fraction of items covered", fontsize=8)
        ax_lorenz.legend(fontsize=7)
        ax_lorenz.tick_params(labelsize=7)

        # ── Column 2: sorted rank plot (log scale) ───────────────────────────
        ax_rank = fig.add_subplot(n_rows, n_cols, level * n_cols + 3)
        sorted_counts = sorted(counts_np[counts_np > 0], reverse=True)
        ax_rank.plot(range(1, len(sorted_counts) + 1), sorted_counts,
                     color="#4393c3", linewidth=1.2)
        ax_rank.axhline(uniform_per_code, color="black", linestyle="--",
                        linewidth=0.8, label="uniform")
        ax_rank.set_yscale("log")
        ax_rank.set_title(f"Level {level}  usage rank (log scale, active only)", fontsize=9)
        ax_rank.set_xlabel("Code rank (most → least used)", fontsize=8)
        ax_rank.set_ylabel("Items assigned (log)", fontsize=8)
        ax_rank.legend(fontsize=7)
        ax_rank.tick_params(labelsize=7)

    # ── Bottom row: cross-level heatmap (left) + collision panel (right) ────
    row_offset = num_codebooks * n_cols

    # Cross-level co-occurrence heatmap: levels 0 vs 1
    ax_heat = fig.add_subplot(n_rows, n_cols, row_offset + 1)
    if num_codebooks >= 2 and len(usage_counts[0]) <= 256:
        # Build joint count matrix C[i,j] = # items with code i at L0 and code j at L1
        # Pull semantic_ids from usage_counts indirectly via counts — we need to pass them
        # through; store hint on function via kwarg captured in closure below
        heat = _cross_level_heat  # set by caller if available, else None
        if heat is not None:
            im = ax_heat.imshow(heat, aspect="auto", cmap="YlOrRd",
                                interpolation="nearest")
            ax_heat.set_title("Level 0 × Level 1 co-occurrence\n(bright = many items share both codes)",
                              fontsize=9)
            ax_heat.set_xlabel("Level-1 code", fontsize=8)
            ax_heat.set_ylabel("Level-0 code", fontsize=8)
            ax_heat.tick_params(labelsize=7)
            plt.colorbar(im, ax=ax_heat, fraction=0.03, pad=0.02)
        else:
            ax_heat.axis("off")
            ax_heat.text(0.5, 0.5, "Cross-level heatmap\nnot available",
                         ha="center", va="center", transform=ax_heat.transAxes, fontsize=9)
    else:
        ax_heat.axis("off")

    # Perplexity summary bar (centre)
    ax_perp = fig.add_subplot(n_rows, n_cols, row_offset + 2)
    levels = [f"L{s['level']}" for s in summaries]
    perps = [s["perplexity"] for s in summaries]
    max_perp = codebook_size
    colors_perp = ["#4393c3" if p >= 0.5 * max_perp else "#d73027" for p in perps]
    bars = ax_perp.bar(levels, perps, color=colors_perp)
    ax_perp.axhline(max_perp, color="black", linestyle="--", linewidth=0.8,
                    label=f"Max={max_perp}")
    ax_perp.set_title("Perplexity per level\n(closer to max = more uniform)", fontsize=9)
    ax_perp.set_ylabel("Perplexity", fontsize=8)
    ax_perp.legend(fontsize=7)
    ax_perp.tick_params(labelsize=7)
    for bar, p in zip(bars, perps):
        ax_perp.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max_perp * 0.01,
                     f"{p:.1f}", ha="center", va="bottom", fontsize=8)

    # Collision panel (right)
    ax_coll = fig.add_subplot(n_rows, n_cols, row_offset + 3)
    if collision_bucket_hist is not None and collision_summary is not None:
        sizes = list(range(2, len(collision_bucket_hist)))
        cnts = collision_bucket_hist[2:].numpy()
        if len(sizes) > 0 and cnts.sum() > 0:
            ax_coll.bar(sizes, cnts, width=0.8, color="#4393c3")
            ax_coll.set_title(
                f"Collision bucket sizes\n"
                f"{collision_summary['collision_buckets']} colliding tuples  |  "
                f"{collision_summary['collision_item_ratio']:.1%} items affected  |  "
                f"max bucket={collision_summary['max_bucket_size']}",
                fontsize=9,
            )
            ax_coll.set_xlabel("Items sharing same base SID", fontsize=8)
            ax_coll.set_ylabel("Number of buckets", fontsize=8)
            ax_coll.tick_params(labelsize=7)
    else:
        ax_coll.axis("off")
        ax_coll.text(0.5, 0.5, "No collision data", ha="center", va="center",
                     transform=ax_coll.transAxes, fontsize=9)

    fig.tight_layout()
    usage_path = output_dir / "codebook_usage.png"
    fig.savefig(usage_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return usage_path


def _bucket_stats(base_ids, depth):
    """Compute bucket sizes for items sharing the same prefix of length `depth`."""
    prefix = base_ids[:, :depth]
    _, inverse, counts = torch.unique(prefix, dim=0, return_inverse=True, return_counts=True)
    bucket_sizes = counts
    colliding_sizes = bucket_sizes[bucket_sizes > 1]
    n_items = base_ids.size(0)
    return {
        "bucket_sizes": bucket_sizes,
        "colliding_sizes": colliding_sizes,
        "n_unique": int(bucket_sizes.numel()),
        "n_singletons": int((bucket_sizes == 1).sum().item()),
        "n_collision_buckets": int((bucket_sizes > 1).sum().item()),
        "n_items_colliding": int(colliding_sizes.sum().item()) if colliding_sizes.numel() > 0 else 0,
        "collision_item_ratio": float(colliding_sizes.sum().item()) / max(n_items, 1) if colliding_sizes.numel() > 0 else 0.0,
        "max_bucket": int(bucket_sizes.max().item()),
        "mean_collision_bucket": float(colliding_sizes.float().mean().item()) if colliding_sizes.numel() > 0 else 0.0,
    }


def plot_prefix_collisions(semantic_ids, num_codebooks, output_dir):
    """One row per prefix depth: L0, L0+L1, L0+L1+L2.

    Each row shows:
      - Bucket size distribution bar chart
      - Cumulative distribution of items across bucket sizes
      - Summary stats as text
    """
    base_ids = semantic_ids[:, :num_codebooks].long()
    n_items = base_ids.size(0)
    depths = list(range(1, num_codebooks + 1))
    label_map = {
        1: "L0",
        2: "L0 + L1",
        3: "L0 + L1 + L2",
    }

    n_rows = len(depths)
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4.5 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]
    fig.suptitle(
        "Prefix-level collision analysis\n"
        "(how many items share the same code tuple at each level of specificity)",
        fontsize=13, fontweight="bold",
    )

    for row, depth in enumerate(depths):
        stats = _bucket_stats(base_ids, depth)
        bucket_sizes = stats["bucket_sizes"]
        label = label_map.get(depth, f"L0..L{depth-1}")

        # ── col 0: bucket size histogram ──────────────────────────────────
        ax_hist = axes[row, 0]
        max_size = stats["max_bucket"]
        hist = torch.bincount(bucket_sizes, minlength=max_size + 1)
        sizes_x = list(range(1, len(hist)))
        counts_y = hist[1:].numpy()
        bar_colors = ["#b2d1e8" if s == 1 else "#d73027" if s >= 10 else "#4393c3"
                      for s in sizes_x]
        ax_hist.bar(sizes_x, counts_y, width=0.85, color=bar_colors, linewidth=0)
        ax_hist.set_title(
            f"{label}  —  bucket size distribution\n"
            f"{stats['n_unique']} unique tuples  |  "
            f"{stats['n_collision_buckets']} colliding ({stats['collision_item_ratio']:.1%} of items)  |  "
            f"max bucket = {stats['max_bucket']}",
            fontsize=9,
        )
        ax_hist.set_xlabel("Items sharing the same prefix tuple", fontsize=8)
        ax_hist.set_ylabel("Number of buckets", fontsize=8)
        ax_hist.tick_params(labelsize=7)
        # annotate singleton bar
        singleton_count = int(hist[1].item()) if len(hist) > 1 else 0
        ax_hist.text(
            1, singleton_count,
            f"  singletons\n  {singleton_count}",
            va="bottom", ha="left", fontsize=7, color="#555",
        )

        # ── col 1: cumulative % of items vs bucket size ────────────────────
        ax_cdf = axes[row, 1]
        # x = bucket size threshold, y = % of items living in a bucket <= x
        all_sizes_expanded = bucket_sizes.repeat_interleave(bucket_sizes)  # one entry per item
        sorted_item_sizes, _ = torch.sort(all_sizes_expanded)
        cdf_y = torch.arange(1, n_items + 1, dtype=torch.float) / n_items
        ax_cdf.plot(sorted_item_sizes.numpy(), cdf_y.numpy(), color="#4393c3", linewidth=1.4)
        # mark 50% and 90% lines
        for pct, col in [(0.5, "#f4a582"), (0.9, "#d73027")]:
            idx = int(pct * n_items)
            thresh = int(sorted_item_sizes[min(idx, n_items - 1)].item())
            ax_cdf.axhline(pct, color=col, linestyle="--", linewidth=0.9,
                           label=f"{int(pct*100)}% items in buckets ≤ {thresh}")
        ax_cdf.set_title(f"{label}  —  CDF of bucket size per item", fontsize=9)
        ax_cdf.set_xlabel("Bucket size", fontsize=8)
        ax_cdf.set_ylabel("Fraction of items", fontsize=8)
        ax_cdf.legend(fontsize=7)
        ax_cdf.tick_params(labelsize=7)

        # ── col 2: text summary panel ──────────────────────────────────────
        ax_txt = axes[row, 2]
        ax_txt.axis("off")
        lines = [
            f"Prefix depth : {label}",
            "",
            f"Total items          : {n_items:,}",
            f"Unique tuples        : {stats['n_unique']:,}",
            f"Max possible tuples  : {256**depth:,}",
            f"Vocab utilisation    : {stats['n_unique'] / (256**depth):.2%}",
            "",
            f"Singleton buckets    : {stats['n_singletons']:,}",
            f"Collision buckets    : {stats['n_collision_buckets']:,}",
            f"Items in collisions  : {stats['n_items_colliding']:,}  ({stats['collision_item_ratio']:.1%})",
            "",
            f"Max bucket size      : {stats['max_bucket']}",
            f"Mean collision size  : {stats['mean_collision_bucket']:.2f}",
        ]
        ax_txt.text(
            0.08, 0.92, "\n".join(lines),
            transform=ax_txt.transAxes,
            va="top", ha="left",
            fontsize=9,
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f7f7f7", edgecolor="#cccccc"),
        )

    fig.tight_layout()
    out_path = output_dir / "prefix_collision_analysis.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_summary(output_dir, summaries, collision_summary):
    payload = {
        "codebooks": summaries,
        "collision": collision_summary,
    }
    summary_path = output_dir / "usage_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return summary_path


def print_summary(summaries, collision_summary):
    for summary in summaries:
        print(
            f"Codebook {summary['level'] + 1}: "
            f"active={summary['active_codes']} "
            f"usage_ratio={summary['usage_ratio']:.2%} "
            f"perplexity={summary['perplexity']:.2f} "
            f"max_count={summary['max_count']}"
        )

    if collision_summary is not None:
        print(
            "Collision buckets: "
            f"occupied_slots={collision_summary['occupied_base_slots']}/"
            f"{collision_summary['total_possible_base_slots']} "
            f"({collision_summary['occupied_base_slot_ratio']:.6%}) "
            f"unique_base_tuples={collision_summary['unique_base_tuples']} "
            f"collision_buckets={collision_summary['collision_buckets']} "
            f"collision_slot_ratio_all={collision_summary['collision_slot_ratio_of_all_slots']:.6%} "
            f"collision_slot_ratio_occupied={collision_summary['collision_slot_ratio_of_occupied_slots']:.2%} "
            f"items_in_collisions={collision_summary['items_in_collision_buckets']} "
            f"extra_items={collision_summary['extra_items_due_to_collision']} "
            f"max_bucket_size={collision_summary['max_bucket_size']} "
            f"mean_bucket_size={collision_summary['mean_collision_bucket_size']:.2f}"
        )


def main():
    args = parse_args()
    artifact = load_artifact(args.artifact_path)
    num_codebooks, codebook_size = infer_codebook_config(artifact["state_dict"])
    semantic_ids = artifact["semantic_ids"]

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    usage_counts, summaries, collision_bucket_hist, collision_summary = compute_usage_stats(
        semantic_ids=semantic_ids,
        num_codebooks=num_codebooks,
        codebook_size=codebook_size,
    )

    # Build cross-level co-occurrence heatmap (L0 × L1).
    global _cross_level_heat
    _cross_level_heat = None
    if num_codebooks >= 2 and codebook_size <= 256:
        base_ids = semantic_ids[:, :num_codebooks].long()
        heat = torch.zeros(codebook_size, codebook_size, dtype=torch.long)
        for row in range(base_ids.size(0)):
            heat[base_ids[row, 0], base_ids[row, 1]] += 1
        _cross_level_heat = heat.numpy()

    usage_path = plot_usage(
        usage_counts=usage_counts,
        summaries=summaries,
        collision_bucket_hist=collision_bucket_hist,
        collision_summary=collision_summary,
        output_dir=output_dir,
    )
    prefix_path = plot_prefix_collisions(
        semantic_ids=semantic_ids,
        num_codebooks=num_codebooks,
        output_dir=output_dir,
    )
    summary_path = save_summary(output_dir, summaries, collision_summary)
    print_summary(summaries, collision_summary)

    print(f"Saved codebook usage plot to {usage_path}")
    print(f"Saved prefix collision analysis to {prefix_path}")
    print(f"Saved usage summary to {summary_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
