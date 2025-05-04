# Fruit Disease Classification with Vision Transformer

This project implements a binary classification model to detect diseased and healthy fruits using a Vision Transformer (ViT) model. The model is trained on a dataset of fruit images, leveraging data augmentation, a pre-trained ViT backbone, and evaluation metrics such as accuracy, confusion matrix, and ROC curve.

## Project Structure

- `train_CNN.py`: Training with CNN.
- `train_EfficientNetB0.py`: Training with EfficientNetB0.
- `train_EfficientNetB0_optimized.py`: Training with EfficientNetB0_optimized.
- `train_ResNet34.py`: Training with ResNet34.
- `train_VGG16.py`: Training with VGG16.
- `train_Vision_transformers.py`: Training with VGG16.
- `requirements.txt`: Lists the project dependencies.

## Prerequisites

- Python 3.8 or higher
- A GPU (recommended for faster training)
- Dataset of fruit images organized into `healthy` and `unhealthy` folders

## Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/fruit-disease-classification.git
   cd fruit-disease-classification
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the dataset**:
   - Place your dataset in a directory (e.g., `data/afterpreprocessing`).
   - Ensure the dataset is organized as:
     ```
     data/afterpreprocessing/
     ├── healthy/
     └── unhealthy/
     ```

## Usage

1. **Train the model**:
   Run the following script to train the model:
   ```bash
   python -c "
   import torch
   from model import VisionTransformer
   from data import get_data_loaders
   from train import train_model

   data_dir = 'data/afterpreprocessing'
   train_loader, test_loader = get_data_loaders(data_dir)
   model = VisionTransformer(num_classes=1)
   model = train_model(model, train_loader, test_loader, num_epochs=5)
   "
   ```

2. **Evaluate the model**:
   Evaluate the trained model and generate visualizations:
   ```bash
   python -c "
   import torch
   from model import VisionTransformer
   from data import get_data_loaders
   from evaluate import evaluate_model

   data_dir = 'data/afterpreprocessing'
   train_loader, test_loader = get_data_loaders(data_dir)
   model = VisionTransformer(num_classes=1)
   model.load_state_dict(torch.load('best_model.pth'))
   evaluate_model(model, test_loader)
   "
   ```

3. **Outputs**:
   - The trained model is saved as `best_model.pth`.
   - Visualizations (`confusion_matrix.png`, `roc_curve.png`) are saved in the project directory.

## Results

- The model achieves high accuracy on the test set (refer to the notebook output for exact figures).
- Visualizations include:
  - Confusion matrix showing true vs. predicted labels.
  - ROC curve with AUC score.

## Notes

- Ensure you have sufficient disk space for the dataset and model weights.
- Adjust `data_dir` in the scripts to match your dataset path.
- The model uses a pre-trained ViT backbone from the `timm` library, which requires an internet connection for the initial download.

## License

This project is licensed under the MIT License.
