# Medical-Image-Processing-

# BreakHis 40x and 200x Classification

This project trains deep learning models (Custom CNN and MobileNetV2) to classify **benign vs. malignant** breast tissue using the **BreaKHis v1 dataset**. It focuses on 40x and 200x magnifications, ensuring patient-level separation to avoid data leakage.

## 1. Create Environment

### Download miniconda
```bash
Invoke-WebRequest -Uri "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe" -OutFile "miniconda.exe"
Start-Process -FilePath ".\miniconda.exe" -ArgumentList "/S", "/D=$env:USERPROFILE\Miniconda3" -Wait
Remove-Item .\miniconda.exe
```
### After installation

Once done, activate Conda:
```bash
& "$env:USERPROFILE\Miniconda3\Scripts\conda.exe" --version
& "$env:USERPROFILE\Miniconda3\Scripts\conda.exe" init powershell
```
close the terminal and reopen it:
``` bash
conda --version # expected output: conda 25.7.0
```

then accept terms (run these commands one by one)
``` bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
```
``` bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```
``` bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2
```

### Then create and activate your environment:
```bash
conda create -n breakhis python=3.10 -y
conda activate breakhis
```

## 2. Install Requirements

Make sure you are in the project root folder, then run:
```bash
pip install -r requirements.txt
```

The `requirements.txt` includes:
```
tensorflow>=2.12.0
scikit-learn>=1.2.0
pandas>=2.0.0
numpy>=1.23.0
matplotlib>=3.7.0
requests>=2.28.0
```


## 3. Run Training

Run the main script:
```bash
python BreakHis_training_notebook.py
```
This will:
- Extract the dataset.
- Create patient-aware train/val/test splits.
- Train custom CNN and MobileNetV2 models for 40x and 200x magnifications.
- Save model checkpoints, histories, and performance reports in `./trained_models`.

## 4. Outputs

- **Models:** `trained_models/mag_40/` and `trained_models/mag_200/`
- **Reports:** JSON summaries of test accuracy, AUC, and sample counts.
- **Histories:** Training curves (loss, accuracy, AUC) saved per model.
