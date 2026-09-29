# HCSA-YOLO

Official implementation of **A Hierarchical Context-Fusion and Synergistic Alignment Network for Small Object Detection**.

<img width="861" height="755" alt="HCSA-YOLO architecture" src="https://github.com/user-attachments/assets/8e46fb92-88bc-4fda-8696-261192643ab7" />

## Requirements

- Python 3.9 or later
- PyTorch 2.0 or later
- MMCV 2.0 or later
- MMEngine 0.8 or later

Install PyTorch according to your CUDA version, then install MMCV and MMEngine. The MMCV build must include the compiled `mmcv.ops` extensions required by modulated deformable convolution.

```bash
pip install torch torchvision
pip install mmengine mmcv
```

Ensure that the PyTorch, CUDA, and MMCV versions are mutually compatible before training or inference.

## Model configuration

The HCSA-YOLO model configuration is located at:

```text
ultralytics/cfg/models/HBB/yolo11-hcsa.yaml
```

## Citation

If this work is useful in your research, please cite:

```bibtex
@article{zhang2026hierarchical,
  title={A Hierarchical Context-Fusion and Synergistic Alignment Network for Small Object Detection},
  author={Zhang, Bin and Pan, Weiguo and Xu, Bingxin and Dai, Songyin},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={242},
  pages={19--35},
  year={2026},
  publisher={Elsevier}
}
```
