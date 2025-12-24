import math
import os
import sys
from pathlib import Path

import torch
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.language_gated_bundle_module import LanguageGatedBundleMatchModule


def _dummy_data(batch_size=2, num_prop=8, lang_size=64):
    data_dict = {
        "aggregated_vote_xyz": torch.randn(batch_size, num_prop, 3),
        "aggregated_vote_features": torch.randn(batch_size, num_prop, 128, requires_grad=True),
        "lang_emb": torch.randn(batch_size, lang_size),
        "objectness_scores": torch.randn(batch_size, num_prop, 2)
    }
    return data_dict


def test_forward_shapes_and_backward():
    torch.manual_seed(0)
    module = LanguageGatedBundleMatchModule(
        num_proposals=8,
        lang_size=64,
        hidden_size=32,
        knn_k=4,
        diffusion_steps=2,
        diffusion_alpha=0.7
    )
    module.train()
    data_dict = _dummy_data()

    out = module(data_dict)
    assert "cluster_ref" in out
    assert "bundle_features" in out
    assert out["cluster_ref"].shape == (2, 8)
    assert out["bundle_features"].shape == (2, 8, 128)
    assert not torch.isnan(out["cluster_ref"]).any()

    loss = out["cluster_ref"].sum()
    loss.backward()
    grads = data_dict["aggregated_vote_features"].grad
    assert grads is not None
    assert grads.shape == data_dict["aggregated_vote_features"].shape


def test_knn_graph_expected_neighbors():
    module = LanguageGatedBundleMatchModule(num_proposals=4, knn_k=2)
    xyz = torch.tensor([[[0.0, 0.0, 0.0],
                         [1.0, 0.0, 0.0],
                         [2.0, 0.0, 0.0],
                         [3.0, 0.0, 0.0]]])

    idx, delta = module._build_knn_graph(xyz)
    assert idx.shape == (1, 4, 2)
    assert delta.shape == (1, 4, 2, 3)

    # Node 1 should see nodes 0 and 2
    assert set(idx[0, 1].tolist()) == {0, 2}
    # Check delta directions
    expected_delta = torch.tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    node1_delta = delta[0, 1]
    assert torch.allclose(torch.sort(node1_delta, dim=0).values, torch.sort(expected_delta, dim=0).values)


def test_rotate_features_quadrature():
    features = torch.tensor([[[[[1.0, 0.0], [0.0, 1.0]]]]])  # shape (1,1,1,2,2)
    features = features.view(1, 1, 1, 4)
    theta = torch.tensor([[[math.pi / 2]]])

    rotated = LanguageGatedBundleMatchModule._rotate_features(features, theta)
    rotated = rotated.view(1, 1, 1, 2, 2)
    real_part = rotated[..., 0]
    imag_part = rotated[..., 1]

    # 90-degree rotation should map (1,0) -> (0,1) and (0,1) -> (-1,0)
    assert torch.allclose(real_part[0, 0, 0, 0], torch.tensor(0.0), atol=1e-5)
    assert torch.allclose(imag_part[0, 0, 0, 0], torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(real_part[0, 0, 0, 1], torch.tensor(-1.0), atol=1e-5)
    assert torch.allclose(imag_part[0, 0, 0, 1], torch.tensor(0.0), atol=1e-5)
