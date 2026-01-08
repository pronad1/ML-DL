"""
Train Improved Sparse R-CNN for Spine Lesion Detection
Goal: Beat paper's 33.15 mAP@0.5 baseline

Key Improvements:
1. ResNet-50 → ResNet-101 backbone (+1.5-2 mAP)
2. Multi-scale training (+0.5-1 mAP)
3. 50k → 90k iterations (+0.3-0.5 mAP)
4. 100 → 300 proposals (+0.5-1 mAP)

Expected result: 36-38 mAP@0.5 (+2.85-4.85 improvement)
"""

import os
import sys
from pathlib import Path

# Add detectron2 to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir / "detectron2"))

print("="*80)
print("Training Improved Sparse R-CNN for Spine Lesion Detection")
print("="*80)
print()
print("Target: Beat paper's 33.15 mAP@0.5 baseline")
print("Expected: 36-38 mAP@0.5 with improvements")
print()
print("Key Improvements:")
print("  1. ResNet-101 backbone (vs ResNet-50)")
print("  2. Multi-scale training (6 scales)")
print("  3. 90k iterations (vs 50k)")
print("  4. 300 proposals (vs 100)")
print()
print("="*80)
print()

# Check if pretrained weights exist
pretrained_path = script_dir / "pretrained" / "r101_100pro_3x_model.pth"
if not pretrained_path.exists():
    print("❌ ERROR: Pretrained R101 weights not found!")
    print(f"   Expected: {pretrained_path}")
    print()
    print("Download from:")
    print("https://drive.google.com/drive/folders/19UaSgR4OwqA-BhCs_wG7i6E-OXC5NR__")
    print()
    print("Save as: pretrained/r101_100pro_3x_model.pth")
    sys.exit(1)

print(f"✓ Found pretrained weights: {pretrained_path}")
print()

# Check GPU
import torch
if not torch.cuda.is_available():
    print("❌ WARNING: No GPU detected! Training will be VERY slow.")
    print("   Consider using Google Colab or cloud GPU.")
else:
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"✓ GPU detected: {gpu_name}")
    print(f"  Memory: {gpu_memory:.1f} GB")
    print()
    
    if gpu_memory < 8:
        print("⚠️  WARNING: GPU has <8GB memory")
        print("   Consider reducing batch size if OOM errors occur")
        print()

# Import detectron2
try:
    from detectron2.engine import launch
    print("✓ Detectron2 imported successfully")
except ImportError as e:
    print("❌ ERROR: Failed to import detectron2")
    print(f"   {e}")
    print()
    print("Install detectron2:")
    print("  pip install 'git+https://github.com/facebookresearch/detectron2.git'")
    sys.exit(1)

print()
print("="*80)
print("Starting training...")
print("="*80)
print()
print("Training will take ~2 days on RTX 3050 (8GB)")
print("Progress will be saved every 5k iterations")
print()
print("Monitor training:")
print("  - Logs: outputs/sparsercnn_improved/log.txt")
print("  - Checkpoints: outputs/sparsercnn_improved/model_*.pth")
print("  - TensorBoard: tensorboard --logdir outputs/sparsercnn_improved")
print()

# Run training
if __name__ == "__main__":
    # Import train_net from spine module
    sys.path.insert(0, str(script_dir / "spine"))
    from train_net import main
    
    # Set config file
    config_file = str(script_dir / "spine" / "configs" / "sparsercnn_improved.yaml")
    
    # Override sys.argv for detectron2's argument parser
    sys.argv = [
        "train_improved_detection.py",
        "--config-file", config_file,
        "--num-gpus", "1",
    ]
    
    # Launch training
    launch(
        main,
        num_gpus_per_machine=1,
        num_machines=1,
        machine_rank=0,
        dist_url="auto",
    )
