"""
Train the RQ-VAE quantizer first, then build Semantic IDs once, then train the
seq2seq transformer on user histories expressed in Semantic-ID tokens.
"""

import argparse
import copy
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from dataset import (
    SemanticSequenceDataset,
    build_item_embedding_matrix,
    collate_sequences,
    filter_and_split_user_histories,
    load_item_embeddings,
    load_user_histories,
)
from rqvae import Quantizer
from seq2seq_transformer import Transformer
from utils import (
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    SPECIAL_TOKEN_COUNT,
    ExperimentLogger,
    build_token_sizes,
    build_valid_sid_prefix_map,
    get_default_transformer_steps,
    load_runtime_config,
    semantic_id_to_tokens,
    stable_user_bucket,
    tokens_to_semantic_id,
)


def build_transformer_examples(
    train_histories,
    item_to_semantic_id,
    num_user_buckets,
    max_history_items,
    token_sizes,
):
    """Build prefix -> next-item training pairs for the transformer."""
    semantic_token_count = sum(token_sizes)
    user_token_offset = SPECIAL_TOKEN_COUNT + semantic_token_count
    examples = []

    for user_id, history in train_histories.items():
        semantic_history = [
            item_to_semantic_id[item_id]
            for item_id in history
            if item_id in item_to_semantic_id
        ]
        if len(semantic_history) < 2:
            continue

        user_token = user_token_offset + stable_user_bucket(user_id, num_user_buckets)
        for target_index in range(1, len(semantic_history)):
            prefix = semantic_history[max(0, target_index - max_history_items):target_index]
            src_tokens = [user_token]
            for semantic_id in prefix:
                src_tokens.extend(semantic_id_to_tokens(semantic_id, token_sizes))

            target_semantic_id = semantic_history[target_index]
            target_tokens = [BOS_TOKEN]
            target_tokens.extend(semantic_id_to_tokens(target_semantic_id, token_sizes))
            target_tokens.append(EOS_TOKEN)

            examples.append((src_tokens, target_tokens[:-1], target_tokens[1:]))

    input_vocab_size = user_token_offset + num_user_buckets
    output_vocab_size = SPECIAL_TOKEN_COUNT + semantic_token_count
    max_target_len = len(token_sizes) + 1
    return examples, input_vocab_size, output_vocab_size, max_target_len


def build_candidate_token_bank(item_ids, quantizer, token_sizes):
    """Precompute target token sequences for all candidate items."""
    candidate_item_ids = []
    tgt_inputs = []
    tgt_outputs = []

    for item_id in item_ids:
        semantic_id = quantizer.lookup_semantic_id(item_id)
        if semantic_id is None:
            continue

        target_tokens = [BOS_TOKEN]
        target_tokens.extend(semantic_id_to_tokens(semantic_id, token_sizes))
        target_tokens.append(EOS_TOKEN)

        candidate_item_ids.append(item_id)
        tgt_inputs.append(torch.tensor(target_tokens[:-1], dtype=torch.long))
        tgt_outputs.append(torch.tensor(target_tokens[1:], dtype=torch.long))

    return (
        candidate_item_ids,
        torch.stack(tgt_inputs, dim=0),
        torch.stack(tgt_outputs, dim=0),
    )


def build_eval_queries(eval_records, quantizer, token_sizes, num_user_buckets, max_history_items):
    """Precompute evaluation source tokens once to reduce validation overhead."""
    semantic_token_count = sum(token_sizes)
    user_token_offset = SPECIAL_TOKEN_COUNT + semantic_token_count
    queries = []

    for user_id, context_history, target_item_id in eval_records:
        if quantizer.lookup_semantic_id(target_item_id) is None:
            continue

        semantic_history = []
        for item_id in context_history:
            semantic_id = quantizer.lookup_semantic_id(item_id)
            if semantic_id is not None:
                semantic_history.append(semantic_id)
        semantic_history = semantic_history[-max_history_items:]

        src_tokens = [user_token_offset + stable_user_bucket(user_id, num_user_buckets)]
        for semantic_id in semantic_history:
            src_tokens.extend(semantic_id_to_tokens(semantic_id, token_sizes))

        queries.append((src_tokens, target_item_id))

    return queries


def limit_eval_queries(eval_queries, max_examples):
    """Use an evenly spaced validation subset during training for speed."""
    if max_examples is None or max_examples <= 0 or len(eval_queries) <= max_examples:
        return eval_queries

    selected_queries = []
    for index in range(max_examples):
        query_index = index * len(eval_queries) // max_examples
        selected_queries.append(eval_queries[query_index])
    return selected_queries


def load_rqvae_checkpoint(checkpoint_path, item_embeddings, device):
    """Load a trained RQ-VAE from a checkpoint and rebuild lookup tables.

    Args:
        checkpoint_path: Path to the rqvae.pt artifact.
        item_embeddings: Full item embedding matrix (used to rebuild semantic IDs).
        device: Torch device.

    Returns:
        tuple: (quantizer, semantic_ids) ready for downstream use.
    """
    artifact = torch.load(checkpoint_path, map_location="cpu")
    state_dict = artifact["state_dict"]
    item_ids = artifact["item_ids"]
    semantic_ids = artifact["semantic_ids"]

    in_dim = item_embeddings.size(1)
    model = Quantizer(in_dim=in_dim)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Rebuild the lookup tables from the saved semantic IDs.
    num_codebooks = model.num_codebooks
    code_indices = semantic_ids[:, :num_codebooks]
    model.handle_collisions(code_indices, item_ids=item_ids)

    print(
        f"[RQ-VAE] Loaded checkpoint from {checkpoint_path} "
        f"({len(item_ids)} items, {semantic_ids.size(1)} SID positions)"
    )
    return model, item_ids, semantic_ids


def build_usage_probe(item_embeddings, max_probe_size=8192):
    """Use an evenly spaced subset to monitor codebook usage during RQ-VAE training."""
    if item_embeddings.size(0) <= max_probe_size:
        return item_embeddings

    probe_indices = torch.linspace(
        0,
        item_embeddings.size(0) - 1,
        steps=max_probe_size,
    ).round().long()
    return item_embeddings[probe_indices]


def build_kmeans_init_subset(item_embeddings, max_init_size=20000):
    """Use up to 20k items for RQ-VAE k-means initialization."""
    if item_embeddings.size(0) <= max_init_size:
        return item_embeddings

    init_indices = torch.linspace(
        0,
        item_embeddings.size(0) - 1,
        steps=max_init_size,
    ).round().long()
    return item_embeddings[init_indices]


def train_rqvae(
    item_embeddings,
    device,
    batch_size,
    learning_rate,
    weight_decay,
    epochs,
    log_every,
    kmeans_init_items,
    logger=None,
):
    """Train the RQ-VAE on fixed item embeddings."""
    model = Quantizer(in_dim=item_embeddings.size(1)).to(device)
    train_loader = DataLoader(
        TensorDataset(item_embeddings),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )

    kmeans_init_subset = build_kmeans_init_subset(
        item_embeddings,
        max_init_size=kmeans_init_items,
    ).to(device)
    with torch.no_grad():
        model.initialize_codebooks_kmeans(model.encode_inputs(kmeans_init_subset))

    usage_probe = build_usage_probe(item_embeddings).to(device)
    print(f"[RQ-VAE] k-means init items: {kmeans_init_subset.size(0)}/{item_embeddings.size(0)}")
    print(f"[RQ-VAE] usage probe items: {usage_probe.size(0)}/{item_embeddings.size(0)}")
    init_summary = model.summarize_codebook_usage(usage_probe)
    init_usage_text = ",".join(
        f"{level['active_codes']}/{model.codebook_size}"
        for level in init_summary["levels"]
    )
    print(
        "[RQ-VAE:init] "
        f"latent_norm={init_summary['latent_norm_mean']:.3f} "
        f"latent_std={init_summary['latent_std_mean']:.3f} "
        f"p_unique_ids={init_summary['p_unique_ids']:.4f} "
        f"max_duplicates={init_summary['max_id_duplicates']} "
        f"usage={init_usage_text}"
    )
    if logger is not None:
        init_metrics = {
            "latent_norm": init_summary["latent_norm_mean"],
            "latent_std": init_summary["latent_std_mean"],
            "p_unique_ids": init_summary["p_unique_ids"],
            "max_duplicates": init_summary["max_id_duplicates"],
        }
        for level_index, level_summary in enumerate(init_summary["levels"]):
            init_metrics[f"level_{level_index}_active_codes"] = level_summary["active_codes"]
            init_metrics[f"level_{level_index}_usage_ratio"] = level_summary["usage_ratio"]
            init_metrics[f"level_{level_index}_perplexity"] = level_summary["perplexity"]
        logger.log_metrics(init_metrics, step=0, namespace="rqvae", force=True)

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    progress_bar = tqdm(range(1, epochs + 1), desc="RQ-VAE", dynamic_ncols=True)
    for epoch in progress_bar:
        model.train()
        running_loss = 0.0

        for (batch_embeddings,) in train_loader:
            batch_embeddings = batch_embeddings.to(device)
            loss, _ = model(batch_embeddings)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_embeddings.size(0)

        mean_loss = running_loss / len(item_embeddings)
        if epoch == 1 or epoch == epochs or epoch % log_every == 0:
            usage_summary = model.summarize_codebook_usage(usage_probe)
            usage_text = ",".join(
                f"{level['active_codes']}/{model.codebook_size}"
                for level in usage_summary["levels"]
            )
            progress_bar.set_postfix(
                loss=f"{mean_loss:.6f}",
                latent=f"{usage_summary['latent_norm_mean']:.3f}",
                unique=f"{usage_summary['p_unique_ids']:.3f}",
                dup=f"{usage_summary['max_id_duplicates']}",
                usage=usage_text,
            )
            print(
                f"[RQ-VAE] epoch={epoch} "
                f"loss={mean_loss:.6f} "
                f"latent_norm={usage_summary['latent_norm_mean']:.3f} "
                f"latent_std={usage_summary['latent_std_mean']:.3f} "
                f"p_unique_ids={usage_summary['p_unique_ids']:.4f} "
                f"max_duplicates={usage_summary['max_id_duplicates']} "
                f"rqvae_entropy={usage_summary['rqvae_entropy']:.3f} "
                f"usage={usage_text}"
            )
            if logger is not None:
                rqvae_metrics = {
                    "loss": mean_loss,
                    "latent_norm": usage_summary["latent_norm_mean"],
                    "latent_std": usage_summary["latent_std_mean"],
                    "p_unique_ids": usage_summary["p_unique_ids"],
                    "max_duplicates": usage_summary["max_id_duplicates"],
                    "rqvae_entropy": usage_summary["rqvae_entropy"],
                }
                for level_index, level_summary in enumerate(usage_summary["levels"]):
                    rqvae_metrics[f"level_{level_index}_active_codes"] = level_summary["active_codes"]
                    rqvae_metrics[f"level_{level_index}_usage_ratio"] = level_summary["usage_ratio"]
                    rqvae_metrics[f"level_{level_index}_perplexity"] = level_summary["perplexity"]
                logger.log_metrics(rqvae_metrics, step=epoch, namespace="rqvae", force=True)

    return model


@torch.no_grad()
def beam_search_next_items(
    model,
    memory,
    memory_key_padding_mask,
    quantizer,
    token_sizes,
    valid_sid_prefix_map,
    top_k,
    beam_size,
    max_beam_size,
    beam_growth,
):
    """Autoregressively decode Semantic IDs with beam search constrained to valid SID prefixes."""
    del max_beam_size
    del beam_growth

    requested_top_k = max(int(top_k), 1)
    beam_size = max(int(beam_size), requested_top_k)
    max_decode_steps = len(token_sizes) + 1
    beams = [([BOS_TOKEN], 0.0)]

    for _ in range(max_decode_steps):
        beam_tokens = torch.tensor(
            [tokens for tokens, _ in beams],
            dtype=torch.long,
            device=memory.device,
        )
        batch_size = beam_tokens.size(0)
        memory_batch = memory.expand(batch_size, -1, -1)
        mem_pad_batch = memory_key_padding_mask.expand(batch_size, -1)

        logits = model.decode(
            beam_tokens,
            memory_batch,
            memory_key_padding_mask=mem_pad_batch,
        )
        step_log_probs = logits[:, -1, :].log_softmax(dim=-1)

        expanded_beams = []
        for beam_index, (tokens, score) in enumerate(beams):
            allowed_tokens = valid_sid_prefix_map.get(tuple(tokens))
            if allowed_tokens is None or allowed_tokens.numel() == 0:
                continue

            allowed_log_probs = step_log_probs[beam_index].index_select(dim=0, index=allowed_tokens)
            per_beam_width = min(beam_size, allowed_tokens.numel())
            top_log_probs, top_indices = torch.topk(allowed_log_probs, k=per_beam_width, dim=0)

            for candidate_rank in range(per_beam_width):
                next_token = int(allowed_tokens[top_indices[candidate_rank]].item())
                next_score = score + float(top_log_probs[candidate_rank].item())
                expanded_beams.append((tokens + [next_token], next_score))

        if not expanded_beams:
            return [], {
                "beam_size_used": beam_size,
                "attempts": 1,
                "invalid_ids": 0.0,
                "completed_beams": 0.0,
            }

        expanded_beams.sort(key=lambda item: item[1], reverse=True)
        beams = expanded_beams[:beam_size]

    ranked_items = []
    seen_items = set()
    for tokens, _ in beams:
        semantic_id = tokens_to_semantic_id(tokens[1:-1], token_sizes)
        item_id = quantizer.lookup_item(semantic_id)
        if item_id is None or item_id in seen_items:
            continue
        seen_items.add(item_id)
        if len(ranked_items) < requested_top_k:
            ranked_items.append(item_id)

    return ranked_items, {
        "beam_size_used": beam_size,
        "attempts": 1,
        "invalid_ids": 0.0,
        "completed_beams": float(len(beams)),
    }


def compute_ranking_metrics(rank_position):
    """Recall and NDCG for K in {5, 10} from a zero-based rank."""
    metrics = {
        "recall@5": 0.0,
        "ndcg@5": 0.0,
        "recall@10": 0.0,
        "ndcg@10": 0.0,
    }
    if rank_position < 5:
        metrics["recall@5"] = 1.0
        metrics["ndcg@5"] = 1.0 / math.log2(rank_position + 2)
    if rank_position < 10:
        metrics["recall@10"] = 1.0
        metrics["ndcg@10"] = 1.0 / math.log2(rank_position + 2)
    return metrics


@torch.no_grad()
def evaluate_ranking(
    transformer,
    eval_queries,
    quantizer,
    token_sizes,
    device,
    top_k,
    beam_size,
    max_beam_size,
    beam_growth,
    query_batch_size,
    desc,
):
    """Paper-style leave-one-out evaluation via autoregressive beam decoding."""
    empty_result = {
        "examples": 0,
        "recall@5": 0.0,
        "ndcg@5": 0.0,
        "recall@10": 0.0,
        "ndcg@10": 0.0,
        "avg_beam_size": 0.0,
        "avg_attempts": 0.0,
        "invalid_id_rate": 0.0,
    }
    if not eval_queries:
        return empty_result

    was_training = transformer.training
    transformer.eval()

    filtered_queries = list(eval_queries)
    if not filtered_queries:
        if was_training:
            transformer.train()
        return empty_result

    query_batch_size = max(int(query_batch_size), 1)
    valid_sid_prefix_map = build_valid_sid_prefix_map(quantizer.item_to_semantic_id, token_sizes, device)
    totals = {
        "recall@5": 0.0,
        "ndcg@5": 0.0,
        "recall@10": 0.0,
        "ndcg@10": 0.0,
    }
    decode_totals = {
        "beam_size_used": 0.0,
        "attempts": 0.0,
        "invalid_ids": 0.0,
        "completed_beams": 0.0,
    }

    for start in tqdm(
        range(0, len(filtered_queries), query_batch_size),
        desc=desc,
        leave=False,
        dynamic_ncols=True,
    ):
        batch_queries = filtered_queries[start : start + query_batch_size]
        batch_src = [
            torch.tensor(src_tokens, dtype=torch.long, device=device)
            for src_tokens, _ in batch_queries
        ]
        batch_targets = [target_item_id for _, target_item_id in batch_queries]

        src_padded = nn.utils.rnn.pad_sequence(
            batch_src,
            batch_first=True,
            padding_value=PAD_TOKEN,
        )
        src_pad_mask = src_padded.eq(PAD_TOKEN)
        batch_memories = transformer.encode(src_padded, src_key_padding_mask=src_pad_mask)

        for batch_index, target_item_id in enumerate(batch_targets):
            predicted_items, decode_stats = beam_search_next_items(
                model=transformer,
                memory=batch_memories[batch_index : batch_index + 1],
                memory_key_padding_mask=src_pad_mask[batch_index : batch_index + 1],
                quantizer=quantizer,
                token_sizes=token_sizes,
                valid_sid_prefix_map=valid_sid_prefix_map,
                top_k=top_k,
                beam_size=beam_size,
                max_beam_size=max_beam_size,
                beam_growth=beam_growth,
            )

            for stat_name, stat_value in decode_stats.items():
                decode_totals[stat_name] += stat_value

            if target_item_id in predicted_items:
                rank_position = predicted_items.index(target_item_id)
                metrics = compute_ranking_metrics(rank_position)
                for metric_name, metric_value in metrics.items():
                    totals[metric_name] += metric_value

    evaluated_examples = len(filtered_queries)

    if was_training:
        transformer.train()

    return {
        "examples": evaluated_examples,
        "recall@5": totals["recall@5"] / max(evaluated_examples, 1),
        "ndcg@5": totals["ndcg@5"] / max(evaluated_examples, 1),
        "recall@10": totals["recall@10"] / max(evaluated_examples, 1),
        "ndcg@10": totals["ndcg@10"] / max(evaluated_examples, 1),
        "avg_beam_size": decode_totals["beam_size_used"] / max(evaluated_examples, 1),
        "avg_attempts": decode_totals["attempts"] / max(evaluated_examples, 1),
        "invalid_id_rate": decode_totals["invalid_ids"] / max(decode_totals["completed_beams"], 1.0),
    }


def train_transformer(
    examples,
    input_vocab_size,
    output_vocab_size,
    max_seq_len,
    device,
    batch_size,
    learning_rate,
    train_steps,
    warmup_steps,
    log_every,
    eval_every,
    val_queries,
    quantizer,
    token_sizes,
    eval_top_k,
    eval_beam_size,
    eval_max_beam_size,
    eval_beam_growth,
    eval_query_batch_size,
    logger=None,
):
    """Train the seq2seq transformer on Semantic-ID prediction."""
    model = Transformer(
        input_dim=input_vocab_size,
        output_dim=output_vocab_size,
        max_seq_len=max_seq_len,
    ).to(device)

    train_loader = DataLoader(
        SemanticSequenceDataset(examples),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_sequences,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    optimizer = AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN, label_smoothing=0.1)

    model.train()
    train_iter = iter(train_loader)
    running_loss = 0.0
    total_tokens = 0
    best_state_dict = None
    best_val_metrics = None

    progress_bar = tqdm(range(1, train_steps + 1), desc="Transformer", dynamic_ncols=True)
    for step in progress_bar:
        try:
            src, tgt_input, tgt_output = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            src, tgt_input, tgt_output = next(train_iter)

        if step <= warmup_steps:
            lr = learning_rate * step / warmup_steps
        else:
            lr = learning_rate * math.sqrt(warmup_steps / step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        src = src.to(device)
        tgt_input = tgt_input.to(device)
        tgt_output = tgt_output.to(device)

        src_pad_mask = src.eq(PAD_TOKEN)
        tgt_pad_mask = tgt_input.eq(PAD_TOKEN)
        logits = model(
            src,
            tgt_input,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask,
        )

        loss = criterion(logits.reshape(-1, output_vocab_size), tgt_output.reshape(-1))

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n[Transformer] NaN/Inf loss detected at step {step}, stopping early.")
            break

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        non_pad_tokens = tgt_output.ne(PAD_TOKEN).sum().item()
        running_loss += loss.item() * non_pad_tokens
        total_tokens += non_pad_tokens

        if step == train_steps or step % log_every == 0:
            mean_loss = running_loss / max(total_tokens, 1)
            postfix = {
                "lr": f"{lr:.6f}",
                "train_loss": f"{mean_loss:.6f}",
            }
            transformer_metrics = {
                "train_loss": mean_loss,
                "lr": lr,
            }

            should_eval = val_queries and (step == train_steps or step % eval_every == 0)
            if should_eval:
                val_metrics = evaluate_ranking(
                    transformer=model,
                    eval_queries=val_queries,
                    quantizer=quantizer,
                    token_sizes=token_sizes,
                    device=device,
                    top_k=eval_top_k,
                    beam_size=eval_beam_size,
                    max_beam_size=eval_max_beam_size,
                    beam_growth=eval_beam_growth,
                    query_batch_size=eval_query_batch_size,
                    desc=f"Validation @ step {step}",
                )
                postfix["val_R@10"] = f"{val_metrics['recall@10']:.4f}"
                postfix["val_N@10"] = f"{val_metrics['ndcg@10']:.4f}"
                postfix["val_inv%"] = f"{100.0 * val_metrics['invalid_id_rate']:.2f}"
                postfix["val_beam"] = f"{val_metrics['avg_beam_size']:.1f}"
                transformer_metrics.update(
                    {
                        "val_recall@5": val_metrics["recall@5"],
                        "val_ndcg@5": val_metrics["ndcg@5"],
                        "val_recall@10": val_metrics["recall@10"],
                        "val_ndcg@10": val_metrics["ndcg@10"],
                        "val_invalid_id_rate": val_metrics["invalid_id_rate"],
                        "val_avg_beam_size": val_metrics["avg_beam_size"],
                        "val_avg_attempts": val_metrics["avg_attempts"],
                    }
                )
                is_better = (
                    best_val_metrics is None
                    or val_metrics["recall@10"] > best_val_metrics["recall@10"]
                    or (
                        val_metrics["recall@10"] == best_val_metrics["recall@10"]
                        and val_metrics["ndcg@10"] >= best_val_metrics["ndcg@10"]
                    )
                )
                if is_better:
                    best_val_metrics = val_metrics
                    best_state_dict = copy.deepcopy(model.state_dict())

            if logger is not None:
                logger.log_metrics(transformer_metrics, step=step, namespace="transformer", force=True)

            progress_bar.set_postfix(postfix)
            running_loss = 0.0
            total_tokens = 0

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(
            "[Transformer] loaded best validation checkpoint "
            f"(R@10={best_val_metrics['recall@10']:.4f}, "
            f"N@10={best_val_metrics['ndcg@10']:.4f})"
        )

    return model


def serialize_semantic_id(semantic_id):
    return "-".join(str(token) for token in semantic_id)


def atomic_torch_save(payload, path):
    """Write a torch artifact atomically so other processes never see a partial file."""
    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp_path)
    temp_path.replace(path)


def atomic_json_dump(payload, path):
    """Write a JSON artifact atomically so readers only see complete files."""
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    temp_path.replace(path)


def save_rqvae_artifacts(
    output_dir,
    quantizer,
    item_ids,
    semantic_ids,
    token_sizes,
    num_user_buckets,
    max_history_items,
):
    """Persist quantizer outputs as soon as Semantic IDs are available."""
    output_dir.mkdir(parents=True, exist_ok=True)

    atomic_torch_save(
        {
            "state_dict": quantizer.state_dict(),
            "item_ids": item_ids,
            "semantic_ids": semantic_ids.cpu(),
        },
        output_dir / "rqvae.pt",
    )

    item_to_semantic_id = {
        item_id: list(quantizer.lookup_semantic_id(item_id))
        for item_id in item_ids
    }
    semantic_id_to_item = {
        serialize_semantic_id(semantic_id): item_id
        for semantic_id, item_id in quantizer.semantic_id_to_item.items()
    }

    atomic_json_dump(item_to_semantic_id, output_dir / "item_to_semantic_id.json")
    atomic_json_dump(semantic_id_to_item, output_dir / "semantic_id_to_item.json")
    atomic_json_dump(
        {
            "pad_token": PAD_TOKEN,
            "bos_token": BOS_TOKEN,
            "eos_token": EOS_TOKEN,
            "special_token_count": SPECIAL_TOKEN_COUNT,
            "num_user_buckets": num_user_buckets,
            "max_history_items": max_history_items,
            "token_sizes": token_sizes,
            "base_codebook_size": quantizer.codebook_size,
            "semantic_id_length": int(semantic_ids.size(1)),
        },
        output_dir / "tokenizer_config.json",
    )
    atomic_json_dump(
        {
            "rqvae_complete": True,
            "transformer_complete": False,
        },
        output_dir / "artifact_status.json",
    )


def save_transformer_artifact(output_dir, transformer):
    """Persist the trained transformer as soon as training finishes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_torch_save({"state_dict": transformer.state_dict()}, output_dir / "transformer.pt")
    atomic_json_dump(
        {
            "rqvae_complete": (output_dir / "rqvae.pt").exists(),
            "transformer_complete": True,
        },
        output_dir / "artifact_status.json",
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train RQ-VAE and transformer for generative retrieval.")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--interactions-path", type=Path, default=None)
    parser.add_argument("--item-embeddings-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument(
        "--skip-rqvae-training",
        action="store_true",
        default=None,
        help="Load RQ-VAE from existing checkpoint instead of retraining. "
             "Default: auto-detect (skip if artifacts/rqvae.pt exists).",
    )
    parser.add_argument(
        "--force-rqvae-training",
        action="store_true",
        help="Force RQ-VAE retraining even if a checkpoint exists.",
    )
    parser.add_argument("--strict-paper-vocab", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    full_config, training_config, logging_config = load_runtime_config(args.config)
    data_config = training_config.get("data", {}) or {}
    rqvae_config = training_config.get("rqvae", {}) or {}
    transformer_config = training_config.get("transformer", {}) or {}
    evaluation_config = training_config.get("evaluation", {}) or {}

    eval_top_k = int(evaluation_config.get("top-k", 10))
    if eval_top_k < 10:
        raise ValueError("training.evaluation.top-k must be at least 10 to report Recall/NDCG@10.")

    dataset_label = args.dataset_name
    if dataset_label is None and args.interactions_path is not None:
        dataset_label = args.interactions_path.stem
    if dataset_label is None:
        dataset_label = args.output_dir.name

    experiment_logger = ExperimentLogger(
        config=logging_config,
        output_dir=args.output_dir,
        run_name=dataset_label,
        run_config={
            "args": vars(args),
            "config": full_config,
        },
    )

    try:
        transformer_train_steps = get_default_transformer_steps(
            args.dataset_name,
            args.interactions_path,
            transformer_config.get("train-steps", {}) or {},
        )

        user_histories = load_user_histories(
            dataset_name=args.dataset_name,
            interactions_path=args.interactions_path,
        )
        filtered_histories, train_histories, val_records, test_records = filter_and_split_user_histories(
            user_histories,
            min_reviews=int(data_config.get("min-user-reviews", 5)),
        )
        embedding_by_item = load_item_embeddings(args.item_embeddings_path)
        item_ids, item_embeddings = build_item_embedding_matrix(filtered_histories, embedding_by_item)

        print(
            f"Loaded {len(filtered_histories)} filtered users and {len(item_ids)} unique items. "
            f"Validation users: {len(val_records)}. Test users: {len(test_records)}."
        )
        experiment_logger.log_metrics(
            {
                "filtered_users": len(filtered_histories),
                "unique_items": len(item_ids),
                "validation_users": len(val_records),
                "test_users": len(test_records),
            },
            step=0,
            namespace="data",
            force=True,
        )

        rqvae_checkpoint = args.output_dir / "rqvae.pt"
        status_path = args.output_dir / "artifact_status.json"
        rqvae_available = (
            rqvae_checkpoint.exists()
            and status_path.exists()
            and json.loads(status_path.read_text(encoding="utf-8")).get("rqvae_complete", False)
        )
        should_train_rqvae = args.force_rqvae_training or (
            not rqvae_available if args.skip_rqvae_training is None else not args.skip_rqvae_training
        )

        if should_train_rqvae:
            rqvae = train_rqvae(
                item_embeddings=item_embeddings,
                device=device,
                batch_size=int(rqvae_config.get("batch-size", 1024)),
                learning_rate=float(rqvae_config.get("lr", 5e-4)),
                weight_decay=float(rqvae_config.get("weight-decay", 0.01)),
                epochs=int(rqvae_config.get("epochs", 20_000)),
                log_every=max(int(logging_config.get("wandb", {}).get("log_every_steps", 100)), 1),
                kmeans_init_items=int(rqvae_config.get("kmeans-init-items", 20_000)),
                logger=experiment_logger,
            )

            semantic_ids = rqvae.build_semantic_ids_after_training(
                item_embeddings.to(device),
                item_ids=item_ids,
            ).cpu()

            token_sizes = build_token_sizes(
                semantic_ids=semantic_ids,
                base_codebook_size=rqvae.codebook_size,
                num_codebooks=rqvae.num_codebooks,
                strict_paper_vocab=args.strict_paper_vocab,
            )
            if len(token_sizes) > rqvae.num_codebooks and token_sizes[-1] > rqvae.codebook_size:
                print(
                    "[Warning] Collision token vocabulary exceeds 256. "
                    f"Using adaptive c4 size={token_sizes[-1]} instead of strict paper 256."
                )
            save_rqvae_artifacts(
                output_dir=args.output_dir,
                quantizer=rqvae,
                item_ids=item_ids,
                semantic_ids=semantic_ids,
                token_sizes=token_sizes,
                num_user_buckets=int(data_config.get("num-user-buckets", 2000)),
                max_history_items=int(data_config.get("max-history-items", 20)),
            )
            print(f"[Artifacts] Saved RQ-VAE stage artifacts to {args.output_dir}.")
        else:
            rqvae, item_ids, semantic_ids = load_rqvae_checkpoint(
                checkpoint_path=rqvae_checkpoint,
                item_embeddings=item_embeddings,
                device=device,
            )
            tokenizer_config_path = args.output_dir / "tokenizer_config.json"
            with open(tokenizer_config_path, "r", encoding="utf-8") as handle:
                tokenizer_config = json.load(handle)
            token_sizes = tokenizer_config["token_sizes"]

        examples, input_vocab_size, output_vocab_size, max_target_len = build_transformer_examples(
            train_histories=train_histories,
            item_to_semantic_id=rqvae.item_to_semantic_id,
            num_user_buckets=int(data_config.get("num-user-buckets", 2000)),
            max_history_items=int(data_config.get("max-history-items", 20)),
            token_sizes=token_sizes,
        )
        if not examples:
            raise ValueError("No transformer training examples could be built from the user histories.")

        val_queries = build_eval_queries(
            eval_records=val_records,
            quantizer=rqvae,
            token_sizes=token_sizes,
            num_user_buckets=int(data_config.get("num-user-buckets", 2000)),
            max_history_items=int(data_config.get("max-history-items", 20)),
        )
        full_val_count = len(val_queries)
        val_queries = limit_eval_queries(val_queries, int(evaluation_config.get("val-max-examples", 256)))
        test_queries = build_eval_queries(
            eval_records=test_records,
            quantizer=rqvae,
            token_sizes=token_sizes,
            num_user_buckets=int(data_config.get("num-user-buckets", 2000)),
            max_history_items=int(data_config.get("max-history-items", 20)),
        )
        max_src_len = max(len(src_tokens) for src_tokens, _, _ in examples)

        print(
            f"Validation queries: {len(val_queries)}/{full_val_count} used during training. "
            f"Test queries: {len(test_queries)}."
        )
        experiment_logger.log_metrics(
            {
                "train_examples": len(examples),
                "validation_queries": len(val_queries),
                "validation_queries_full": full_val_count,
                "test_queries": len(test_queries),
                "max_src_len": max_src_len,
                "max_target_len": max_target_len,
                "transformer_train_steps": transformer_train_steps,
            },
            step=0,
            namespace="data",
            force=True,
        )

        transformer = train_transformer(
            examples=examples,
            input_vocab_size=input_vocab_size,
            output_vocab_size=output_vocab_size,
            max_seq_len=max(max_src_len, max_target_len),
            device=device,
            batch_size=int(transformer_config.get("batch-size", 256)),
            learning_rate=float(transformer_config.get("lr", 1e-4)),
            train_steps=transformer_train_steps,
            warmup_steps=int(transformer_config.get("warmup-steps", 10_000)),
            log_every=max(int(logging_config.get("wandb", {}).get("log_every_steps", 100)), 1),
            eval_every=int(evaluation_config.get("every", 10_000)),
            val_queries=val_queries,
            quantizer=rqvae,
            token_sizes=token_sizes,
            eval_top_k=eval_top_k,
            eval_beam_size=int(evaluation_config.get("beam-size", 40)),
            eval_max_beam_size=int(evaluation_config.get("max-beam-size", 320)),
            eval_beam_growth=int(evaluation_config.get("beam-growth", 2)),
            eval_query_batch_size=int(evaluation_config.get("query-batch-size", 256)),
            logger=experiment_logger,
        )
        save_transformer_artifact(args.output_dir, transformer)
        print(f"[Artifacts] Saved transformer stage artifact to {args.output_dir}.")

        test_metrics = evaluate_ranking(
            transformer=transformer,
            eval_queries=test_queries,
            quantizer=rqvae,
            token_sizes=token_sizes,
            device=device,
            top_k=eval_top_k,
            beam_size=int(evaluation_config.get("beam-size", 40)),
            max_beam_size=int(evaluation_config.get("max-beam-size", 320)),
            beam_growth=int(evaluation_config.get("beam-growth", 2)),
            query_batch_size=int(evaluation_config.get("query-batch-size", 256)),
            desc="Test",
        )
        experiment_logger.log_metrics(test_metrics, step=transformer_train_steps, namespace="test", force=True)
        print(
            "[Test] "
            f"examples={test_metrics['examples']} "
            f"R@5={test_metrics['recall@5']:.4f} "
            f"N@5={test_metrics['ndcg@5']:.4f} "
            f"R@10={test_metrics['recall@10']:.4f} "
            f"N@10={test_metrics['ndcg@10']:.4f} "
            f"invalid_id_rate={100.0 * test_metrics['invalid_id_rate']:.2f}% "
            f"avg_beam={test_metrics['avg_beam_size']:.1f}"
        )

        print(f"Artifacts available in {args.output_dir}.")
    finally:
        experiment_logger.close()


if __name__ == "__main__":
    main()
