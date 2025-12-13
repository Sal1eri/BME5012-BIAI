# AD Detection

follow the setup below to initial env

```
1. >> create a env based on Python version = 3.12

2. >> pip install -r requirements.txt

-- if it dont work, try step4

3. >> bash utils/download_kaggle_dataset.sh

4. >> pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121
```


| Method | Params | Accuracy (%) | Macro-F1 (%) | Macro-AUC (%) | Balanced Acc. (%) | Cohen's $\kappa$ (%) | MCC (Multi-class) (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **VGG16 (2D-TL)** | 134.3 M | 41.46 | 42.45 | 80.82 | 44.67 | 24.83 | 29.18 |
| **3D-ResNet** | 33.4 M | 60.97 | 66.20 | 83.51 | 66.66 | 50.34 | 50.64 |
| **DA-ViT** |-- | 73.17 | 77.83 | 91.55 | 78.00 | 65.83 | 66.19 |
| **Swin-CNN ：(Ours)** |-- | 68.29 | 71.42 | 89.16 | 72.67 | 59.80 | 60.35 |


## Supported Models

You can now train the following models by modifying the configuration files in `configs/training/`:

  * **VGG16**: 2D Transfer Learning baseline.
  * **3D-ResNet**: Standard 3D Convolutional Network.
  * **HCCT-ViT**: Hybrid Compact Convolution Transformer.
  * **3D-ViT**: Standard Vision Transformer for 3D volumes.
  * **Swin-CNN**: Combination of Swin Transformer and 3D CNN.

The VGG16 2D IMAGE Come from the combination of 3 slice like the image below.

<p align="center">
  <img src="./slice_views/demo.png" alt="Demo Image" width="600"/>
</p>

<p align="center">
  <img src="./slice_views/axial.png" alt="Axial" width="180" style="display:inline-block; margin-right:10px;"/>
  <img src="./slice_views/coronal.png" alt="Coronal" width="180" style="display:inline-block; margin-right:10px;"/>
  <img src="./slice_views/sagittal.png" alt="Sagittal" width="180" style="display:inline-block;"/>
  <img src="./slice_views/view_sample.png" alt="Sagittal" width="180" style="display:inline-block;"/>
</p>


