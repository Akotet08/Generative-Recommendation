"""
This module contains the implementation of the neural network models used in the project.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Quantizer(nn.Module):
    """RQ-VAE quantizer with residual codebooks and collision-aware Semantic IDs."""

    def __init__(self, in_dim, hidden_dim=32, codebook_size=256, num_codebooks=3,
                 beta=0.25, ema_decay=0.99, dead_code_threshold=2.0):
        """Initialize the quantizer.

        Args:
            in_dim (int): Input embedding dimensionality.
            hidden_dim (int): Latent dimensionality (paper uses 32).
            codebook_size (int): Number of entries in each residual codebook.
            num_codebooks (int): Number of residual quantization levels.
            beta (float): Commitment loss coefficient.
            ema_decay (float): EMA decay rate for codebook updates.
            dead_code_threshold (float): EMA count below which a code is
                considered dead and gets re-initialised.
        """
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks
        self.beta = beta
        self.ema_decay = ema_decay
        self.dead_code_threshold = dead_code_threshold

        # Paper encoder shape (when in_dim=768): 768 -> 512 -> 256 -> 128 -> 32.
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim * 16),
            nn.ReLU(),
            nn.Linear(hidden_dim * 16, hidden_dim * 8),
            nn.ReLU(),
            nn.Linear(hidden_dim * 8, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.latent_norm = nn.LayerNorm(hidden_dim)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim * 8),
            nn.ReLU(),
            nn.Linear(hidden_dim * 8, hidden_dim * 16),
            nn.ReLU(),
            nn.Linear(hidden_dim * 16, in_dim),
        )

        self.codebooks = nn.ModuleList([
            nn.Embedding(codebook_size, hidden_dim) for _ in range(num_codebooks)
        ])
        # Codebooks are updated via EMA, not gradient descent.
        for cb in self.codebooks:
            cb.weight.requires_grad_(False)

        # EMA running statistics for each codebook level.
        for i in range(num_codebooks):
            self.register_buffer(f'ema_count_{i}', torch.zeros(codebook_size))
            self.register_buffer(f'ema_weight_{i}', torch.zeros(codebook_size, hidden_dim))

        self.codebooks_initialized = False
        self.semantic_id_to_item = {}
        self.item_to_semantic_id = {}

    def encode_inputs(self, x):
        """Encode inputs into a scale-stabilized latent space before quantization."""
        return self.latent_norm(self.encoder(x))

    def _squared_l2_distance(self, x, y):
        """Compute squared Euclidean distance between two sets of vectors."""
        x2 = (x ** 2).sum(dim=1, keepdim=True)
        y2 = (y ** 2).sum(dim=1).unsqueeze(0)
        xy = x @ y.t()
        return x2 + y2 - 2 * xy

    @torch.no_grad()
    def _run_kmeans(self, vectors, num_clusters, num_iters=25):
        """Simple torch-only k-means used for first-batch codebook init."""
        num_samples = vectors.size(0)
        if num_samples == 0:
            raise ValueError("Cannot run k-means with an empty tensor.")

        if num_samples >= num_clusters:
            init_idx = torch.randperm(num_samples, device=vectors.device)[:num_clusters]
        else:
            init_idx = torch.randint(0, num_samples, (num_clusters,), device=vectors.device)
        centers = vectors[init_idx].clone()

        for _ in range(num_iters):
            distances = self._squared_l2_distance(vectors, centers)
            assignments = distances.argmin(dim=1)

            new_centers = []
            for cluster_id in range(num_clusters):
                cluster_points = vectors[assignments == cluster_id]
                if cluster_points.numel() == 0:
                    fallback_idx = torch.randint(0, num_samples, (1,), device=vectors.device)
                    new_centers.append(vectors[fallback_idx].squeeze(0))
                else:
                    new_centers.append(cluster_points.mean(dim=0))
            new_centers = torch.stack(new_centers, dim=0)

            if torch.allclose(centers, new_centers):
                centers = new_centers
                break
            centers = new_centers

        return centers

    @torch.no_grad()
    def initialize_codebooks_kmeans(self, encoded_batch, num_iters=25):
        """Initialize all residual codebooks with k-means on the first encoded batch.

        Args:
            encoded_batch (torch.Tensor): Shape [batch_size, hidden_dim].
            num_iters (int): Number of k-means iterations.
        """
        if encoded_batch.dim() != 2 or encoded_batch.size(1) != self.hidden_dim:
            raise ValueError(
                "encoded_batch must have shape [batch_size, hidden_dim]. "
                f"Got {tuple(encoded_batch.shape)}."
            )

        residual = encoded_batch
        for level in range(self.num_codebooks):
            centers = self._run_kmeans(residual, self.codebook_size, num_iters=num_iters)
            self.codebooks[level].weight.copy_(centers)

            # Seed EMA buffers so the first training step starts from k-means.
            ema_count = getattr(self, f'ema_count_{level}')
            ema_weight = getattr(self, f'ema_weight_{level}')
            ema_count.fill_(1.0)
            ema_weight.copy_(centers)

            distances = self._squared_l2_distance(residual, centers)
            nearest_indices = distances.argmin(dim=1)
            quantized_vectors = centers[nearest_indices]
            residual = residual - quantized_vectors

        self.codebooks_initialized = True

    def quantize(self, encoded):
        """Residual-quantize encoded features.

        Args:
            encoded (torch.Tensor): Encoded vectors of shape [batch_size, hidden_dim].

        Returns:
            tuple:
                code_indices: [batch_size, num_codebooks]
                quantizer_rep_st: straight-through quantized representation
                commitment_loss: scalar commitment loss
        """
        residual = encoded
        quantizer_rep = torch.zeros_like(encoded)
        code_indices = []
        commitment_loss = encoded.new_tensor(0.0)

        for level, codebook in enumerate(self.codebooks):
            codebook_vectors = codebook.weight

            distances = self._squared_l2_distance(residual, codebook_vectors)
            nearest_indices = torch.argmin(distances, dim=1)
            quantized_vector = codebook_vectors[nearest_indices]

            # --- EMA codebook update (training only) ---
            if self.training:
                with torch.no_grad():
                    one_hot = F.one_hot(nearest_indices, self.codebook_size).float()
                    batch_count = one_hot.sum(dim=0)
                    batch_sum = one_hot.T @ residual

                    ema_count = getattr(self, f'ema_count_{level}')
                    ema_weight = getattr(self, f'ema_weight_{level}')

                    ema_count.mul_(self.ema_decay).add_(batch_count, alpha=1 - self.ema_decay)
                    ema_weight.mul_(self.ema_decay).add_(batch_sum, alpha=1 - self.ema_decay)

                    # Laplace smoothing to avoid division by zero.
                    n = ema_count.sum()
                    smoothed = (
                        (ema_count + 1e-5)
                        / (n + self.codebook_size * 1e-5)
                        * n
                    )
                    codebook.weight.data.copy_(ema_weight / smoothed.unsqueeze(1))

                    # Dead-code restart: replace unused codes with random
                    # encoder outputs from the current batch.
                    dead_mask = ema_count < self.dead_code_threshold
                    n_dead = dead_mask.sum().item()
                    if n_dead > 0:
                        rand_idx = torch.randint(
                            0, residual.size(0), (n_dead,), device=residual.device,
                        )
                        codebook.weight.data[dead_mask] = residual[rand_idx].clone()
                        ema_count[dead_mask] = 1.0
                        ema_weight[dead_mask] = residual[rand_idx].clone()

            # Commitment loss only (codebooks updated via EMA, not gradients).
            commitment_loss = commitment_loss + self.beta * F.mse_loss(
                residual, quantized_vector.detach(),
            )

            quantizer_rep = quantizer_rep + quantized_vector
            code_indices.append(nearest_indices)
            # Detach quantized vector so residuals across levels are independent.
            residual = residual - quantized_vector.detach()

        code_indices = torch.stack(code_indices, dim=1)
        quantizer_rep_st = encoded + (quantizer_rep - encoded).detach()
        return code_indices, quantizer_rep_st, commitment_loss

    @torch.no_grad()
    def summarize_codebook_usage(self, x):
        """Summarize latent scale and codebook usage for a probe batch."""
        was_training = self.training
        self.eval()

        encoded = self.encode_inputs(x)
        code_indices, _, _ = self.quantize(encoded)

        level_summaries = []
        for level in range(self.num_codebooks):
            counts = torch.bincount(code_indices[:, level], minlength=self.codebook_size)
            probs = counts.float()
            probs = probs / probs.sum().clamp_min(1.0)
            entropy = -(probs[probs > 0] * probs[probs > 0].log()).sum()
            level_summaries.append(
                {
                    "active_codes": int((counts > 0).sum().item()),
                    "usage_ratio": float((counts > 0).float().mean().item()),
                    "max_count": int(counts.max().item()),
                    "perplexity": float(entropy.exp().item()),
                }
            )

        summary = {
            "latent_norm_mean": float(encoded.norm(dim=1).mean().item()),
            "latent_std_mean": float(encoded.std(dim=0).mean().item()),
            "levels": level_summaries,
        }

        unique_ids, duplicate_counts = torch.unique(code_indices, dim=0, return_counts=True)
        duplicate_probs = duplicate_counts.float() / duplicate_counts.sum().clamp_min(1.0)
        duplicate_entropy = -(duplicate_probs * duplicate_probs.log()).sum()
        summary["p_unique_ids"] = float(unique_ids.size(0) / max(code_indices.size(0), 1))
        summary["max_id_duplicates"] = int(duplicate_counts.max().item())
        summary["rqvae_entropy"] = float(duplicate_entropy.exp().item())

        if was_training:
            self.train()
        return summary

    @torch.no_grad()
    def handle_collisions(self, code_indices, item_ids=None):
        """Append collision token c4 and build SID lookup tables.

        If multiple items share the same base residual code tuple, assign an extra
        collision token c4 = 0,1,2,... within each bucket.

        Args:
            code_indices (torch.Tensor): Shape [num_items, num_codebooks].
            item_ids (list | tuple | None): External item identifiers. If None,
                item IDs are [0, 1, ..., num_items-1].

        Returns:
            torch.LongTensor: semantic_ids [num_items, num_codebooks + 1]
        """
        if code_indices.dim() != 2 or code_indices.size(1) != self.num_codebooks:
            raise ValueError(
                "code_indices must have shape [num_items, num_codebooks]. "
                f"Got {tuple(code_indices.shape)}."
            )

        num_items = code_indices.size(0)
        if item_ids is None:
            item_ids = list(range(num_items))
        elif len(item_ids) != num_items:
            raise ValueError(f"item_ids must have length {num_items}, got {len(item_ids)}.")

        code_indices = code_indices.long()
        semantic_ids = torch.zeros(
            num_items,
            self.num_codebooks + 1,
            dtype=torch.long,
            device=code_indices.device,
        )
        semantic_ids[:, :self.num_codebooks] = code_indices

        bucket = {}
        for row_idx in range(num_items):
            base_tuple = tuple(code_indices[row_idx].tolist())
            bucket.setdefault(base_tuple, []).append(row_idx)

        self.semantic_id_to_item = {}
        self.item_to_semantic_id = {}

        for row_indices in bucket.values():
            for collision_token, row_idx in enumerate(row_indices):
                semantic_ids[row_idx, self.num_codebooks] = collision_token
                full_sid = tuple(semantic_ids[row_idx].tolist())
                item_id = item_ids[row_idx]
                self.semantic_id_to_item[full_sid] = item_id
                self.item_to_semantic_id[item_id] = full_sid

        return semantic_ids

    def lookup_item(self, semantic_id):
        """Lookup item ID from full Semantic ID (including collision token)."""
        if isinstance(semantic_id, torch.Tensor):
            semantic_id = tuple(int(v) for v in semantic_id.tolist())
        return self.semantic_id_to_item.get(tuple(semantic_id))

    def lookup_semantic_id(self, item_id):
        """Lookup full Semantic ID tuple from item ID."""
        return self.item_to_semantic_id.get(item_id)

    def forward(self, x):
        """Training forward: encode -> quantize -> decode and return loss.

        Collision detection/fixing is intentionally NOT done here. Per the paper,
        it is performed once after RQ-VAE training is completed.

        Args:
            x (torch.Tensor): Input embeddings [batch_size, in_dim].

        Returns:
            tuple:
                total_loss: reconstruction_loss + quantization_loss
                code_indices: [batch_size, num_codebooks] base Semantic IDs before c4
        """
        encoded = self.encode_inputs(x)
        code_indices, quantizer_rep_st, quantization_loss = self.quantize(encoded)
        decoded = self.decoder(quantizer_rep_st)
        reconstruction_loss = F.mse_loss(decoded, x)
        total_loss = reconstruction_loss + quantization_loss
        return total_loss, code_indices

    @torch.no_grad()
    def build_semantic_ids_after_training(self, x, item_ids=None):
        """One-time post-training collision fixing and lookup-table construction.

        This method should be called once after the RQ-VAE model is trained.

        Args:
            x (torch.Tensor): Item embeddings [num_items, in_dim].
            item_ids (list | tuple | None): Optional item IDs aligned with rows in x.

        Returns:
            torch.LongTensor: semantic_ids [num_items, num_codebooks + 1]
        """
        was_training = self.training
        self.eval()
        encoded = self.encode_inputs(x)
        code_indices, _, _ = self.quantize(encoded)
        semantic_ids = self.handle_collisions(code_indices, item_ids=item_ids)
        if was_training:
            self.train()
        return semantic_ids
