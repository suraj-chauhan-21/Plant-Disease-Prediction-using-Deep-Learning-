# Dataset Directory Structure

This folder contains the dataset files used for training and validation. Due to size limits, dataset images are ignored by Git.

## Recommended Dataset: PlantVillage

This system is configured to work with the **PlantVillage Dataset** (which includes 38 classes of crop leaves, split into healthy and diseased).

### Download Instructions

1. Download the dataset from [Kaggle: New Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) or another official source.
2. Unpack the files so that they are structured as follows:

```text
plant-disease-detection/
└── dataset/
    └── train/
        ├── Apple___Apple_scab/
        │   ├── image1.jpg
        │   └── image2.jpg
        ├── Apple___Black_rot/
        ├── ... (all 38 subdirectories)
        └── Tomato___healthy/
```

3. Ensure the folder is named `dataset/train/` inside the repository root, or configure the `DATASET_PATH` variable in `src/config.py`.
