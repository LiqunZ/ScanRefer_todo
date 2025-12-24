import torch
import torch.nn as nn
import torch.nn.functional as F


class LanguageGatedBundleMatchModule(nn.Module):
    """Semantic parallel transport based proposal matcher."""

    def __init__(
        self,
        num_proposals: int = 256,
        lang_size: int = 256,
        hidden_size: int = 128,
        knn_k: int = 16,
        diffusion_steps: int = 3,
        diffusion_alpha: float = 0.5,
        hyper_hidden: int = 128,
    ) -> None:
        super().__init__()
        if num_proposals <= 0:
            raise ValueError("num_proposals must be positive")
        if lang_size <= 0:
            raise ValueError("lang_size must be positive")
        if knn_k <= 0:
            raise ValueError("knn_k must be positive")
        if diffusion_steps <= 0:
            raise ValueError("diffusion_steps must be positive")

        self.num_proposals = num_proposals
        self.lang_size = lang_size
        self.hidden_size = hidden_size
        self.knn_k = knn_k
        self.diffusion_steps = diffusion_steps
        self.diffusion_alpha = diffusion_alpha

        hyper_in_dim = lang_size + 3
        self.theta_mlp = nn.Sequential(
            nn.Linear(hyper_in_dim, hyper_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hyper_hidden, hyper_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hyper_hidden, 1)
        )

        self.transport_norm = nn.LayerNorm(128)
        self.bundle_proj = nn.Sequential(
            nn.Linear(128, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 128)
        )

        score_in_dim = 128 + 128 + lang_size
        self.score_head = nn.Sequential(
            nn.Linear(score_in_dim, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, data_dict):
        xyz = data_dict['aggregated_vote_xyz']
        features = data_dict['aggregated_vote_features']
        lang = data_dict['lang_emb']
        objectness = data_dict['objectness_scores'].max(2)[1].float().unsqueeze(-1)

        knn_idx, delta_xyz = self._build_knn_graph(xyz)
        theta = self._predict_theta(delta_xyz, lang)

        updated = features
        neighbor_mask = self._gather_neighbors(objectness, knn_idx)
        denom = neighbor_mask.sum(dim=2).clamp(min=1.0)

        for _ in range(self.diffusion_steps):
            neighbor_feat = self._gather_neighbors(updated, knn_idx)
            rotated = self._rotate_features(neighbor_feat, theta)
            rotated = rotated * neighbor_mask
            diffused = rotated.sum(dim=2) / denom
            updated = updated + self.diffusion_alpha * (diffused - updated)

        transported = self.bundle_proj(self.transport_norm(updated))
        lang_tiled = lang.unsqueeze(1).expand(-1, self.num_proposals, -1)
        fused = torch.cat([transported, features, lang_tiled], dim=-1)
        confidences = self.score_head(fused).squeeze(-1)
        data_dict['cluster_ref'] = confidences * objectness.squeeze(-1)
        data_dict['bundle_features'] = updated

        return data_dict

    def _build_knn_graph(self, xyz):
        B, N, _ = xyz.shape
        k = min(self.knn_k + 1, N)
        dists = torch.cdist(xyz, xyz)
        knn = torch.topk(dists, k=k, dim=-1, largest=False)[1]
        knn = knn[:, :, 1:]
        neighbor_idx = knn if knn.shape[-1] > 0 else knn.new_zeros((B, N, 0))
        neighbor_xyz = self._gather_neighbors(xyz, neighbor_idx)
        delta = neighbor_xyz - xyz.unsqueeze(2)
        return neighbor_idx, delta

    def _predict_theta(self, delta_xyz, lang):
        B, N, K, _ = delta_xyz.shape
        if K == 0:
            return delta_xyz.new_zeros((B, N, K))
        lang_tiled = lang.unsqueeze(1).unsqueeze(1).expand(-1, N, K, -1)
        theta_input = torch.cat([delta_xyz, lang_tiled], dim=-1)
        theta_flat = theta_input.reshape(-1, theta_input.shape[-1])
        theta = self.theta_mlp(theta_flat).view(B, N, K)
        return torch.tanh(theta) * torch.pi

    @staticmethod
    def _gather_neighbors(src, idx):
        B, N, C = src.shape
        K = idx.shape[-1]
        if K == 0:
            return src.new_zeros((B, N, 0, C))
        idx_flat = idx.reshape(B, -1)
        idx_expand = idx_flat.unsqueeze(-1).expand(-1, -1, C)
        gathered = torch.gather(src, 1, idx_expand)
        return gathered.reshape(B, N, K, C)

    @staticmethod
    def _rotate_features(features, theta):
        if features.shape[-1] % 2 != 0:
            raise ValueError('Feature dimension must be even to apply rotation.')
        B, N, K, C = features.shape
        if K == 0:
            return features
        feat = features.reshape(B, N, K, C // 2, 2)
        theta = theta.unsqueeze(-1)
        cos_val = torch.cos(theta)
        sin_val = torch.sin(theta)
        real = feat[..., 0]
        imag = feat[..., 1]
        rot_real = real * cos_val - imag * sin_val
        rot_imag = real * sin_val + imag * cos_val
        rotated = torch.stack([rot_real, rot_imag], dim=-1)
        return rotated.reshape(B, N, K, C)
