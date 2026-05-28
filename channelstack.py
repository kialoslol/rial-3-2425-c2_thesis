import cv2
import numpy as np
import torch
from pathlib import Path

# Base directory
base_dir = Path(r"D:\[THESIS]\dataset\DLSU 10-13") #change to whatever dir u need

rgb_dir = base_dir / "FITTED RGB" #change to whatever dir u need
ir_dir  = base_dir / "IR" #change to whatever dir u need
out_dir = base_dir / "FUSED"
out_dir.mkdir(exist_ok=True)

# Loop through all RGB images
for rgb_path in rgb_dir.glob("*.*"):  # match .jpg/.png/etc.
    name = rgb_path.stem
    ir_path = next(
        (p for ext in (".jpg", ".jpeg", ".png") for p in [ir_dir / f"{name}{ext}"] if p.exists()),
        None
    )

    if ir_path is None:
        print(f"Skipping {rgb_path.name}: no matching IR found.")
        continue

    # Load RGB
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if rgb is None:
        print(f"Error reading {rgb_path}")
        continue

    # Load IR (Inferno colormap)
    ir_colormap = cv2.imread(str(ir_path), cv2.IMREAD_COLOR)
    if ir_colormap is None:
        print(f"Error reading {ir_path}")
        continue

    # Convert IR Inferno colormap to grayscale
    ir_gray = cv2.cvtColor(ir_colormap, cv2.COLOR_BGR2GRAY)

    # Resize IR if necessary
    if rgb.shape[:2] != ir_gray.shape[:2]:
        ir_gray = cv2.resize(ir_gray, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Convert RGB to correct color order (BGR → RGB)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    # Stack → (H, W, 4)
    ir_expanded = ir_gray[..., np.newaxis]
    rgba = np.concatenate([rgb, ir_expanded], axis=2)

    # Convert to torch tensor (C,H,W) if needed
    t = torch.from_numpy(rgba).permute(2, 0, 1).float()

    print(f"{name}: tensor {tuple(t.shape)}")

    # Save as TIFF to preserve all 4 channels
    out_path = out_dir / f"{name}.tiff"
    cv2.imwrite(str(out_path), cv2.cvtColor(rgba, cv2.COLOR_RGB2RGBA))

print(f"\n✅ Fusion complete. 4-channel images saved in:\n{out_dir}")