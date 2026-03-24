"""
Train the RQ-VAE quantizer first, then build Semantic IDs once, then train the
seq2seq transformer on user histories expressed in Semantic-ID tokens.
"""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm.auto import tqdm

from dataset import AmazonDataset
from rqvae import Quantizer
from seq2seq_transformer import Transformer


PAD_TOKEN = 0
BOS_TOKEN = 1
EOS_TOKEN = 2
SPECIAL_TOKEN_COUNT = 3


class SemanticSequenceDataset(Dataset):
    """Seq2seq examples built from user histories and item Semantic IDs."""

    def __init__(self, examples):
        # Pre-tensorize once to avoid per-__getitem__ tensor creation overhead.
        self.examples = [
            (
                torch.tensor(src, dtype=torch.long),
                torch.tensor(tgt_in, dtype=torch.long),
                torch.tensor(tgt_out, dtype=torch.long),
            )
            for src, tgt_in, tgt_out in examples
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def load_user_histories(dataset_name=None, interactions_path=None):
    """Load user -> ordered item history mapping."""
    if interactions_path is not None:
        with open(interactions_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    if dataset_name is None:
        raise ValueError("Provide either --dataset-name or --interactions-path.")
    return AmazonDataset(dataset_name).data


def load_item_embeddings(embeddings_path):
    """Load precomputed item embeddings from a torch artifact."""
    artifact = torch.load(embeddings_path, map_location="cpu")

    if isinstance(artifact, dict) and "item_ids" in artifact and "embeddings" in artifact:
        item_ids = artifact["item_ids"]
        embeddings = artifact["embeddings"]
        if not torch.is_tensor(embeddings):
            embeddings = torch.tensor(embeddings, dtype=torch.float32)
        return {
            item_id: embeddings[index].float()
            for index, item_id in enumerate(item_ids)
        }

    if isinstance(artifact, dict):
        embedding_by_item = {}
        for item_id, vector in artifact.items():
            if torch.is_tensor(vector):
                embedding_by_item[item_id] = vector.float()
            else:
                embedding_by_item[item_id] = torch.tensor(vector, dtype=torch.float32)
        return embedding_by_item

    raise ValueError(
        "Unsupported embeddings artifact. Expected either "
        "{'item_ids': ..., 'embeddings': ...} or {item_id: embedding}."
    )


def filter_and_split_user_histories(user_histories, min_reviews=5):
    """Apply 5-core filtering and chronological leave-one-out evaluation."""
    filtered_histories = {}
    train_histories = {}
    val_records = []
    test_records = []

    for user_id, history in user_histories.items():
        if len(history) < min_reviews:
            continue

        filtered_histories[user_id] = history
        train_histories[user_id] = history[:-2]
        val_records.append((user_id, history[:-2], history[-2]))
        test_records.append((user_id, history[:-1], history[-1]))

    return filtered_histories, train_histories, val_records, test_records


def build_item_embedding_matrix(user_histories, embedding_by_item):
    """Collect unique items from the filtered dataset and stack their fixed embeddings."""
    item_ids = sorted({item_id for history in user_histories.values() for item_id in history})
    missing_items = [item_id for item_id in item_ids if item_id not in embedding_by_item]
    if missing_items:
        preview = ", ".join(missing_items[:5])
        raise ValueError(
            f"Missing embeddings for {len(missing_items)} items. First missing items: {preview}"
        )

    embeddings = torch.stack([embedding_by_item[item_id] for item_id in item_ids], dim=0)
    return item_ids, embeddings


def collate_sequences(batch):
    """Pad variable-length source and target sequences."""
    src_tensors, tgt_input_tensors, tgt_output_tensors = zip(*batch)
    src = nn.utils.rnn.pad_sequence(src_tensors, batch_first=True, padding_value=PAD_TOKEN)
    tgt_input = nn.utils.rnn.pad_sequence(
        tgt_input_tensors,
        batch_first=True,
        padding_value=PAD_TOKEN,
    )
    tgt_output = nn.utils.rnn.pad_sequence(
        tgt_output_tensors,
        batch_first=True,
        padding_value=PAD_TOKEN,
    )
    return src, tgt_input, tgt_output


def stable_user_bucket(user_id, num_buckets):
    """Deterministic user hashing to keep source vocabulary bounded."""
    digest = hashlib.md5(user_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") % num_buckets


def build_token_sizes(semantic_ids, base_codebook_size, num_codebooks, strict_paper_vocab):
    """Create one token block per SID position."""
    if semantic_ids.size(1) < num_codebooks:
        raise ValueError(
            f"Expected at least {num_codebooks} SID positions, got {semantic_ids.size(1)}."
        )

    base_semantic_ids = semantic_ids[:, :num_codebooks]
    if int(base_semantic_ids.max().item()) >= base_codebook_size:
        raise ValueError(
            "Base Semantic IDs exceed the RQ-VAE codebook size. "
            f"Expected values in [0, {base_codebook_size - 1}]."
        )

    token_sizes = [base_codebook_size] * num_codebooks
    if semantic_ids.size(1) > num_codebooks:
        collision_vocab_size = int(semantic_ids[:, num_codebooks:].max().item()) + 1
        if strict_paper_vocab and collision_vocab_size > base_codebook_size:
            raise ValueError(
                "Collision token cardinality exceeds 256. This run does not match the "
                "paper's fixed 256 x 4 item-token vocabulary."
            )
        token_sizes.extend([max(collision_vocab_size, 1)] * (semantic_ids.size(1) - num_codebooks))

    return token_sizes


def semantic_id_to_tokens(semantic_id, token_sizes):
    """Map each SID position to its own token block."""
    offset = SPECIAL_TOKEN_COUNT
    tokens = []
    for position, token in enumerate(semantic_id):
        tokens.append(offset + int(token))
        offset += token_sizes[position]
    return tokens


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

    return model


@torch.no_grad()
def score_all_candidates(
    model,
    memory,
    memory_key_padding_mask,
    candidate_tgt_inputs,
    candidate_tgt_outputs,
    candidate_batch_size,
):
    """Score all candidate items using pre-computed encoder memory.

    Accepts encoder output directly so the same source sequence is never
    re-encoded across candidate batches.
    """
    all_scores = []

    for start in range(0, candidate_tgt_inputs.size(0), candidate_batch_size):
        end = start + candidate_batch_size
        batch_tgt_inputs = candidate_tgt_inputs[start:end]
        batch_tgt_outputs = candidate_tgt_outputs[start:end]
        batch_size = batch_tgt_inputs.size(0)

        memory_batch = memory.expand(batch_size, -1, -1)
        mem_pad_batch = memory_key_padding_mask.expand(batch_size, -1)
        tgt_pad_mask = batch_tgt_inputs.eq(PAD_TOKEN)

        logits = model.decode(
            batch_tgt_inputs,
            memory_batch,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=mem_pad_batch,
        )
        log_probs = logits.log_softmax(dim=-1)
        token_log_probs = log_probs.gather(2, batch_tgt_outputs.unsqueeze(-1)).squeeze(-1)
        token_mask = batch_tgt_outputs.ne(PAD_TOKEN)
        seq_scores = (token_log_probs * token_mask).sum(dim=1)
        all_scores.append(seq_scores)

    return torch.cat(all_scores, dim=0)


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
    candidate_index,
    candidate_tgt_inputs,
    candidate_tgt_outputs,
    device,
    candidate_batch_size,
    desc,
):
    """Paper-style leave-one-out ranking evaluation with Recall/NDCG.

    All query sources are batch-encoded in a single encoder forward pass.
    Per-query scoring then runs the decoder only (no redundant re-encoding).
    """
    empty_result = {
        "examples": 0,
        "recall@5": 0.0,
        "ndcg@5": 0.0,
        "recall@10": 0.0,
        "ndcg@10": 0.0,
    }
    if not eval_queries:
        return empty_result

    was_training = transformer.training
    transformer.eval()

    # Filter to queries with valid targets and pre-tensorize sources.
    src_list = []
    target_indices = []
    for src_tokens, target_item_id in eval_queries:
        target_idx = candidate_index.get(target_item_id)
        if target_idx is None:
            continue
        src_list.append(torch.tensor(src_tokens, dtype=torch.long, device=device))
        target_indices.append(target_idx)

    if not src_list:
        if was_training:
            transformer.train()
        return empty_result

    # Batch-encode all query sources at once (single encoder forward pass).
    src_padded = nn.utils.rnn.pad_sequence(
        src_list, batch_first=True, padding_value=PAD_TOKEN,
    )
    src_pad_mask = src_padded.eq(PAD_TOKEN)
    all_memories = transformer.encode(src_padded, src_key_padding_mask=src_pad_mask)

    totals = {
        "recall@5": 0.0,
        "ndcg@5": 0.0,
        "recall@10": 0.0,
        "ndcg@10": 0.0,
    }

    # Score each query against all candidates (decoder only, no re-encoding).
    for i in tqdm(
        range(len(src_list)),
        desc=desc,
        leave=False,
        dynamic_ncols=True,
    ):
        memory_i = all_memories[i : i + 1]
        mem_pad_i = src_pad_mask[i : i + 1]

        candidate_scores = score_all_candidates(
            model=transformer,
            memory=memory_i,
            memory_key_padding_mask=mem_pad_i,
            candidate_tgt_inputs=candidate_tgt_inputs,
            candidate_tgt_outputs=candidate_tgt_outputs,
            candidate_batch_size=candidate_batch_size,
        )
        top_k = min(10, candidate_scores.size(0))
        top_indices = torch.topk(candidate_scores, k=top_k).indices.tolist()
        target_index = target_indices[i]
        if target_index in top_indices:
            rank_position = top_indices.index(target_index)
            metrics = compute_ranking_metrics(rank_position)
            for metric_name, metric_value in metrics.items():
                totals[metric_name] += metric_value

    evaluated_examples = len(src_list)

    if was_training:
        transformer.train()

    return {
        "examples": evaluated_examples,
        "recall@5": totals["recall@5"] / max(evaluated_examples, 1),
        "ndcg@5": totals["ndcg@5"] / max(evaluated_examples, 1),
        "recall@10": totals["recall@10"] / max(evaluated_examples, 1),
        "ndcg@10": totals["ndcg@10"] / max(evaluated_examples, 1),
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
    candidate_index,
    candidate_tgt_inputs,
    candidate_tgt_outputs,
    candidate_eval_batch_size,
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

        if step == 1 or step == train_steps or step % log_every == 0:
            mean_loss = running_loss / max(total_tokens, 1)
            postfix = {
                "lr": f"{lr:.6f}",
                "train_loss": f"{mean_loss:.6f}",
            }

            should_eval = val_queries and (step == 1 or step == train_steps or step % eval_every == 0)
            if should_eval:
                val_metrics = evaluate_ranking(
                    transformer=model,
                    eval_queries=val_queries,
                    candidate_index=candidate_index,
                    candidate_tgt_inputs=candidate_tgt_inputs,
                    candidate_tgt_outputs=candidate_tgt_outputs,
                    device=device,
                    candidate_batch_size=candidate_eval_batch_size,
                    desc=f"Validation @ step {step}",
                )
                postfix["val_R@10"] = f"{val_metrics['recall@10']:.4f}"
                postfix["val_N@10"] = f"{val_metrics['ndcg@10']:.4f}"
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


def get_default_transformer_steps(dataset_name, interactions_path):
    """Paper uses 200k steps for Beauty/Sports and 100k for Toys."""
    dataset_key = ""
    if dataset_name is not None:
        dataset_key = dataset_name.lower()
    elif interactions_path is not None:
        dataset_key = interactions_path.stem.lower()

    if "toys" in dataset_key:
        return 100_000
    return 200_000


def parse_args():
    parser = argparse.ArgumentParser(description="Train RQ-VAE and transformer for generative retrieval.")
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
    parser.add_argument("--rqvae-batch-size", type=int, default=1024)
    parser.add_argument("--rqvae-lr", type=float, default=5e-4)
    parser.add_argument("--rqvae-weight-decay", type=float, default=0.01)
    parser.add_argument("--rqvae-epochs", type=int, default=20000)
    parser.add_argument("--rqvae-kmeans-init-items", type=int, default=20000)

    parser.add_argument("--transformer-batch-size", type=int, default=256)
    parser.add_argument("--transformer-lr", type=float, default=1e-4)
    parser.add_argument("--transformer-train-steps", type=int, default=None)
    parser.add_argument("--transformer-warmup-steps", type=int, default=10000)

    parser.add_argument("--num-user-buckets", type=int, default=2000)
    parser.add_argument("--max-history-items", type=int, default=20)
    parser.add_argument("--min-user-reviews", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=10000)
    parser.add_argument("--eval-candidate-batch-size", type=int, default=512)
    parser.add_argument("--val-max-examples", type=int, default=256)
    parser.add_argument("--strict-paper-vocab", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    transformer_train_steps = args.transformer_train_steps or get_default_transformer_steps(
        args.dataset_name,
        args.interactions_path,
    )

    user_histories = load_user_histories(
        dataset_name=args.dataset_name,
        interactions_path=args.interactions_path,
    )
    filtered_histories, train_histories, val_records, test_records = filter_and_split_user_histories(
        user_histories,
        min_reviews=args.min_user_reviews,
    )
    embedding_by_item = load_item_embeddings(args.item_embeddings_path)
    item_ids, item_embeddings = build_item_embedding_matrix(filtered_histories, embedding_by_item)

    print(
        f"Loaded {len(filtered_histories)} filtered users and {len(item_ids)} unique items. "
        f"Validation users: {len(val_records)}. Test users: {len(test_records)}."
    )

    # Decide whether to load an existing RQ-VAE checkpoint or train from scratch.
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
            batch_size=args.rqvae_batch_size,
            learning_rate=args.rqvae_lr,
            weight_decay=args.rqvae_weight_decay,
            epochs=args.rqvae_epochs,
            log_every=args.log_every,
            kmeans_init_items=args.rqvae_kmeans_init_items,
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
            num_user_buckets=args.num_user_buckets,
            max_history_items=args.max_history_items,
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
        num_user_buckets=args.num_user_buckets,
        max_history_items=args.max_history_items,
        token_sizes=token_sizes,
    )
    if not examples:
        raise ValueError("No transformer training examples could be built from the user histories.")

    candidate_item_ids, candidate_tgt_inputs, candidate_tgt_outputs = build_candidate_token_bank(
        item_ids=item_ids,
        quantizer=rqvae,
        token_sizes=token_sizes,
    )
    candidate_tgt_inputs = candidate_tgt_inputs.to(device)
    candidate_tgt_outputs = candidate_tgt_outputs.to(device)
    candidate_index = {
        item_id: index for index, item_id in enumerate(candidate_item_ids)
    }
    val_queries = build_eval_queries(
        eval_records=val_records,
        quantizer=rqvae,
        token_sizes=token_sizes,
        num_user_buckets=args.num_user_buckets,
        max_history_items=args.max_history_items,
    )
    full_val_count = len(val_queries)
    val_queries = limit_eval_queries(val_queries, args.val_max_examples)
    test_queries = build_eval_queries(
        eval_records=test_records,
        quantizer=rqvae,
        token_sizes=token_sizes,
        num_user_buckets=args.num_user_buckets,
        max_history_items=args.max_history_items,
    )
    max_src_len = max(len(src_tokens) for src_tokens, _, _ in examples)

    print(
        f"Validation queries: {len(val_queries)}/{full_val_count} used during training. "
        f"Test queries: {len(test_queries)}."
    )

    transformer = train_transformer(
        examples=examples,
        input_vocab_size=input_vocab_size,
        output_vocab_size=output_vocab_size,
        max_seq_len=max(max_src_len, max_target_len),
        device=device,
        batch_size=args.transformer_batch_size,
        learning_rate=args.transformer_lr,
        train_steps=transformer_train_steps,
        warmup_steps=args.transformer_warmup_steps,
        log_every=args.log_every,
        eval_every=args.eval_every,
        val_queries=val_queries,
        candidate_index=candidate_index,
        candidate_tgt_inputs=candidate_tgt_inputs,
        candidate_tgt_outputs=candidate_tgt_outputs,
        candidate_eval_batch_size=args.eval_candidate_batch_size,
    )
    save_transformer_artifact(args.output_dir, transformer)
    print(f"[Artifacts] Saved transformer stage artifact to {args.output_dir}.")

    test_metrics = evaluate_ranking(
        transformer=transformer,
        eval_queries=test_queries,
        candidate_index=candidate_index,
        candidate_tgt_inputs=candidate_tgt_inputs,
        candidate_tgt_outputs=candidate_tgt_outputs,
        device=device,
        candidate_batch_size=args.eval_candidate_batch_size,
        desc="Test",
    )
    print(
        "[Test] "
        f"examples={test_metrics['examples']} "
        f"R@5={test_metrics['recall@5']:.4f} "
        f"N@5={test_metrics['ndcg@5']:.4f} "
        f"R@10={test_metrics['recall@10']:.4f} "
        f"N@10={test_metrics['ndcg@10']:.4f}"
    )

    print(f"Artifacts available in {args.output_dir}.")


if __name__ == "__main__":
    main()
