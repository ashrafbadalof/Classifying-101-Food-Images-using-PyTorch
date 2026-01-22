# Food-101 Image Classification with ResNet-50

This project fine-tunes a ResNet-50 model on Food-101 dataset for multi-class food image classification using PyTorch

The training is done in two stages:
1. Head-only training 
2. Fine-tuning the last ResNet block (layer 4) + classifier

## Dataset
- Food-101
- 101 food categories
- ~ 101,000 images
- Custom train/test/val sets

## Model
- Backbone: ResNet-50 (ImageNet pretrained)
- Classifier head: Fully connected layer (fc)
- Fine tuning strategy
  - Stage 1: Train only **fc**
  - Stage 2: Unfreeze **layer4** + **fc**

## Training Setup
- Framework: PyTorch
- Loss: CrossEntropyLoss
- Optimizer: AdamW
- Batch size: 32
- Early stopping with patience
- Best model saved using validation accuracy

## Results

| Model                  | Top-1 Test Acc | Top-5 Test Acc |
|------------------------|----------------|----------------|
| ResNet-50 (head only)  | 0.60           | 0.83           |
| ResNet-50 (fine-tuned) | **0.77**       | **0.93**       |

### Per-Class Analysis

- Confusion matrix computed on the test set for fine tuned model
- Per-Class accuracy calculated

Worst performing classes include visually similar foods such as:
- steak
- pork_chop
- filet_mignon
- baby_back_ribs

This suggests ambiguity due to visual overlap

## Key Observations
- Fine-tuning only the last ResNet block provides significant gains over head-only training
- Training time does not scale linearly with trainable parameters due to:
  - Frozen early layers
  - Smaller spatial feature maps in deeper layers

## How to Run

1. Install dependencies
2. Train head-only model
3. Fine-tune ResNet-50 by unfreezing `layer4`
4. Evaluate on the test set


## Future Improvements (Optional)
- Grad-CAM visualization for interpretability
- Inference time benchmarking
- Try alternative backbones (EfficientNet, ConvNeXt)
- Label smoothing or mixup

## Author
Ashraf Badalov