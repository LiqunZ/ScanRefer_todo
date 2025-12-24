import json
from pathlib import Path
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[1]
path = ROOT_DIR / "outputs" / "bundle_debug" / "diffusion_sweep.json"

with open(path, "r") as f:
    logs = json.load(f)

for cfg in logs:
    label = f"steps={cfg['diffusion_steps']}, alpha={cfg['diffusion_alpha']}"
    plt.plot(cfg["steps"], cfg["loss"], label=label)

plt.xlabel("Training step")
plt.ylabel("MSE loss")
plt.legend()
plt.title("Bundle diffusion hyperparam sweep (synthetic overfit)")
plt.show()
