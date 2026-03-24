"""General utilities for config loading, logging, and token bookkeeping."""

import copy
import hashlib
from pathlib import Path

import torch
import yaml


PAD_TOKEN = 0
BOS_TOKEN = 1
EOS_TOKEN = 2
SPECIAL_TOKEN_COUNT = 3

DEFAULT_LOGGING_CONFIG = {
    "wandb": {
        "enabled": False,
        "project": None,
        "entity": None,
        "notes": None,
        "log_every_steps": 100,
    },
    "tensorboard": {
        "enabled": False,
        "log_dir": None,
        "log_every_steps": 100,
        "flush_secs": 30,
    },
}


def deep_update(base, updates):
    """Recursively merge nested dictionaries."""
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_yaml_with_inheritance(config_path):
    """Load a YAML config and merge any inherited config files first."""
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    merged = {}
    inherited_paths = payload.pop("inherit", []) or []
    for inherited_path in inherited_paths:
        parent_payload = load_yaml_with_inheritance(config_path.parent / inherited_path)
        deep_update(merged, parent_payload)

    deep_update(merged, payload)
    return merged


def load_runtime_config(config_path):
    """Load the primary config file and split training/logger sections."""
    payload = load_yaml_with_inheritance(config_path)

    logging_config = copy.deepcopy(DEFAULT_LOGGING_CONFIG)
    for section_name in DEFAULT_LOGGING_CONFIG:
        logging_section = payload.get(section_name, {}) or {}
        logging_config[section_name].update(logging_section)

    training_config = payload.get("training", {}) or {}
    return payload, training_config, logging_config


def make_json_safe(value):
    """Convert Paths/tensors and nested structures into logger-safe values."""
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        if value.numel() != 1:
            return value.tolist()
        return value.item()
    if isinstance(value, dict):
        return {key: make_json_safe(subvalue) for key, subvalue in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    return value


def scalarize_metric(value):
    """Convert a metric value into a scalar float when possible."""
    value = make_json_safe(value)
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


class ExperimentLogger:
    """Thin wrapper around optional WandB and TensorBoard backends."""

    def __init__(self, config, output_dir, run_name=None, run_config=None):
        self.output_dir = Path(output_dir)
        self.run_name = run_name
        self.run_config = make_json_safe(run_config or {})
        self.wandb_cfg = config.get("wandb", {})
        self.tensorboard_cfg = config.get("tensorboard", {})
        self.wandb_run = None
        self.tensorboard_writer = None
        self._last_logged_step = {}

        if self.tensorboard_cfg.get("enabled", False):
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                print("[Logging] TensorBoard enabled in config, but tensorboard is not installed.")
            else:
                log_dir = self.tensorboard_cfg.get("log_dir") or str(self.output_dir / "tensorboard")
                flush_secs = int(self.tensorboard_cfg.get("flush_secs", 30) or 30)
                self.tensorboard_writer = SummaryWriter(log_dir=log_dir, flush_secs=flush_secs)
                print(f"[Logging] TensorBoard enabled at {log_dir}.")

        if self.wandb_cfg.get("enabled", False):
            try:
                import wandb
            except ImportError:
                print("[Logging] WandB enabled in config, but wandb is not installed.")
            else:
                init_kwargs = {
                    "project": self.wandb_cfg.get("project"),
                    "config": self.run_config,
                }
                entity = self.wandb_cfg.get("entity")
                notes = self.wandb_cfg.get("notes")
                if entity:
                    init_kwargs["entity"] = entity
                if notes:
                    init_kwargs["notes"] = notes
                if run_name:
                    init_kwargs["name"] = run_name
                self.wandb_run = wandb.init(**init_kwargs)
                print("[Logging] WandB enabled.")

    def _should_log(self, backend, namespace, step, cadence, force):
        if force or step is None:
            if step is not None:
                self._last_logged_step[(backend, namespace)] = step
            return True

        cadence = max(int(cadence or 1), 1)
        key = (backend, namespace)
        last_step = self._last_logged_step.get(key)
        if last_step is None or step - last_step >= cadence:
            self._last_logged_step[key] = step
            return True
        return False

    def log_metrics(self, metrics, step=None, namespace=None, force=False):
        payload = {}
        for metric_name, metric_value in metrics.items():
            scalar_value = scalarize_metric(metric_value)
            if scalar_value is None:
                continue
            full_name = f"{namespace}/{metric_name}" if namespace else metric_name
            payload[full_name] = scalar_value

        if not payload:
            return

        if self.wandb_run is not None and self._should_log(
            "wandb",
            namespace,
            step,
            self.wandb_cfg.get("log_every_steps", 100),
            force,
        ):
            self.wandb_run.log(payload, step=step)

        if self.tensorboard_writer is not None and self._should_log(
            "tensorboard",
            namespace,
            step,
            self.tensorboard_cfg.get("log_every_steps", 100),
            force,
        ):
            tb_step = 0 if step is None else step
            for metric_name, metric_value in payload.items():
                self.tensorboard_writer.add_scalar(metric_name, metric_value, global_step=tb_step)

    def close(self):
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.flush()
            self.tensorboard_writer.close()
        if self.wandb_run is not None:
            self.wandb_run.finish()


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


def tokens_to_semantic_id(tokens, token_sizes):
    """Invert output token IDs back into the underlying Semantic ID tuple."""
    if len(tokens) != len(token_sizes):
        raise ValueError(f"Expected {len(token_sizes)} tokens, got {len(tokens)}.")

    offset = SPECIAL_TOKEN_COUNT
    semantic_id = []
    for position, token in enumerate(tokens):
        token = int(token)
        token_size = token_sizes[position]
        if token < offset or token >= offset + token_size:
            raise ValueError(
                f"Token {token} is out of range for SID position {position} "
                f"[{offset}, {offset + token_size - 1}]."
            )
        semantic_id.append(token - offset)
        offset += token_size
    return tuple(semantic_id)


def build_position_token_blocks(token_sizes, device):
    """Precompute the valid output token IDs for each SID position."""
    offset = SPECIAL_TOKEN_COUNT
    token_blocks = []
    for token_size in token_sizes:
        token_blocks.append(torch.arange(offset, offset + token_size, device=device))
        offset += token_size
    return token_blocks


def get_default_transformer_steps(dataset_name, interactions_path, train_steps_config):
    """Resolve transformer train steps from config, with dataset-aware defaults."""
    override_steps = train_steps_config.get("override")
    if override_steps:
        return int(override_steps)

    dataset_key = ""
    if dataset_name is not None:
        dataset_key = dataset_name.lower()
    elif interactions_path is not None:
        dataset_key = interactions_path.stem.lower()

    if "toys" in dataset_key:
        return int(train_steps_config.get("default-toys", 100_000))
    return int(train_steps_config.get("default-beauty-and-sports", 200_000))
