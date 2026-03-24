"""Dataset and preprocessing utilities for generative retrieval training."""

import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from utils import PAD_TOKEN


class AmazonDataset:
    """Dataset wrapper for the Amazon review sequence files."""

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.data = self.load_data()

    def load_data(self):
        with open(f"data/{self.dataset_name}_user_item_dict.json", "r", encoding="utf-8") as handle:
            return json.load(handle)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, user_id):
        return self.data.get(user_id, [])


class SemanticSequenceDataset(Dataset):
    """Seq2seq examples built from user histories and item Semantic IDs."""

    def __init__(self, examples):
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
