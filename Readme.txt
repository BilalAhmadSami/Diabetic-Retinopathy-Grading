# Diabetic Retinopathy Grading Scripts

GitHub Repo Link: https://github.com/Tempest-Kohaku/Diabetic-Retinopathy-Grading-Group-34.git 

Website Link: https://huggingface.co/spaces/Kratos7270/Diabetic-Retinopathy-Scanner 

This Zip folder contains three Python scripts for preparing images, training the model, and testing.

## 1. `preprocessing.py`

Prepares retinal fundus images before training.

It can:

- Load images from a folder
- Apply retinal field-of-view cropping
- Pad images to avoid stretching
- Resize images to `384 × 384`
- Apply CLAHE contrast enhancement
- Save processed images
- Optionally create stratified train/test folders using a label CSV

## 2. `train.py`

Trains the diabetic retinopathy grading model.

It:

- Loads preprocessed images and labels
- Creates or reuses a stratified train/validation split
- Applies training augmentations
- Builds the parallel EfficientNet-B4 + Swin Transformer model
- Trains using CORN ordinal loss
- Logs QWK, Macro F1, loss, and per-class recall
- Saves the best model checkpoints based on validation QWK

## 3. `test.py`

Evaluates the trained model on the test set.

It:

- Loads a saved `.pth` checkpoint
- Loads test images and labels
- Runs inference using the same model architecture
- Converts CORN logits into DR grades
- Calculates test metrics such as Accuracy, QWK, Macro F1, Weighted F1, and Recall
- Saves the confusion matrix and prediction results

## Workflow

Run the scripts in this order:
- preprocessing.py
- train.py
- test.py