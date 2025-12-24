import os
import sys
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ensure project root on sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.language_gated_bundle_module import LanguageGatedBundleMatchModule


def build_fake_batch(batch_size=2, num_proposals=32, lang_size=256, device="cuda"):
    """Construct a minimal fake batch matching RefNet+bundle expected keys.

    Keys:
      - aggregated_vote_xyz: (B, N, 3)
      - aggregated_vote_features: (B, N, 128)
      - lang_emb: (B, lang_size)
      - objectness_scores: (B, N, 2)
    """
    xyz = torch.randn(batch_size, num_proposals, 3, device=device)
    feats = torch.randn(batch_size, num_proposals, 128, device=device, requires_grad=True)
    lang = torch.randn(batch_size, lang_size, device=device)

    # make a clear foreground/background pattern in objectness
    objectness_scores = torch.zeros(batch_size, num_proposals, 2, device=device)
    # first half proposals are foreground (class 1), rest background (class 0)
    mid = num_proposals // 2
    objectness_scores[:, :mid, 1] = 10.0  # high logit for foreground
    objectness_scores[:, :mid, 0] = -10.0
    objectness_scores[:, mid:, 0] = 10.0
    objectness_scores[:, mid:, 1] = -10.0

    data_dict = {
        "aggregated_vote_xyz": xyz,
        "aggregated_vote_features": feats,
        "lang_emb": lang,
        "objectness_scores": objectness_scores,
    }
    return data_dict


def run_single_config(device, diffusion_steps, diffusion_alpha, steps=300, print_every=50):
    """Run synthetic overfit for a given (diffusion_steps, diffusion_alpha) pair.

    Returns a dict with losses logged every `print_every` steps.
    """
    batch_size = 2
    num_proposals = 32
    lang_size = 256

    model = LanguageGatedBundleMatchModule(
        num_proposals=num_proposals,
        lang_size=lang_size,
        hidden_size=128,
        knn_k=8,
        diffusion_steps=diffusion_steps,
        diffusion_alpha=diffusion_alpha,
        hyper_hidden=128,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    log = {
        "diffusion_steps": diffusion_steps,
        "diffusion_alpha": float(diffusion_alpha),
        "steps": [],
        "loss": [],
        "scores_sample": [],  # first few scores for quick inspection
    }

    for step in range(steps):
        model.train()
        optimizer.zero_grad()

        data_dict = build_fake_batch(
            batch_size=batch_size,
            num_proposals=num_proposals,
            lang_size=lang_size,
            device=device,
        )

        out = model(data_dict)
        scores = out["cluster_ref"]  # (B, N)

        target = torch.zeros_like(scores)
        target[:, 0] = 1.0

        loss = F.mse_loss(scores, target)
        loss.backward()
        optimizer.step()

        if (step + 1) % print_every == 0 or step == 0:
            with torch.no_grad():
                s = scores[0, :8].detach().cpu().tolist()
                log["steps"].append(step + 1)
                log["loss"].append(float(loss.item()))
                log["scores_sample"].append([round(float(v), 4) for v in s])
                print(
                    f"[steps={diffusion_steps}, alpha={diffusion_alpha}] "
                    f"Step {step+1:04d} | loss={loss.item():.6f} | scores[0,:4]="
                    f"{[round(float(v), 3) for v in s[:4]]}"
                )

    return log


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # sweep configs
    diffusion_steps_list = [1, 2, 3]
    diffusion_alpha_list = [0.3, 0.5, 0.7, 0.9]

    all_logs = []

    for steps_cfg in diffusion_steps_list:
        for alpha_cfg in diffusion_alpha_list:
            print("=" * 80)
            print(f"Running config: diffusion_steps={steps_cfg}, diffusion_alpha={alpha_cfg}")
            cfg_log = run_single_config(
                device=device,
                diffusion_steps=steps_cfg,
                diffusion_alpha=alpha_cfg,
                steps=300,
                print_every=50,
            )
            all_logs.append(cfg_log)

    # save results for later comparison
    out_dir = ROOT_DIR / "outputs" / "bundle_debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "diffusion_sweep.json"
    with open(out_path, "w") as f:
        json.dump(all_logs, f, indent=2)
    print(f"Saved diffusion sweep logs to {out_path}")


if __name__ == "__main__":
    main()
