# AFA vs Baseline demo (CIFAR-10, ResNet-18)

Streamlit app that classifies an image with two ResNet-18 models side by side:
a baseline and one trained with Auxiliary Fourier-basis Augmentation (AFA).
Optional Fourier-basis or Gaussian noise can be added before classification.

## Run
    pip install -r requirements.txt
    streamlit run app.py

## Files needed in this folder
    models/baseline_latest.pt
    models/afa_3x3_80ep.pt
    sample_images/   (optional, 32x32 CIFAR-10 pngs)

## Input
A JPG or PNG (uploaded, or a bundled sample). It is centre-cropped and resized to 32x32.
Pixels are scaled to 0-1 with no mean/std normalisation, matching training.

## Output
Top-3 class probabilities from each model, plus the original and noisy image.

## Worked example
1. Pick a sample image (e.g. cat), noise "None": both models give the same class.
2. Choose "Gaussian", strength 0.15: compare the two models' confidence.

(Add a screenshot here.)

## Results (CIFAR-10-C, 19 corruptions averaged over 5 severities)
Clean accuracy: baseline 94.69%, AFA 95.14%.
Robustness accuracy: baseline 74.67%, AFA 80.28%.
See cifar10c_results.json for the per-corruption numbers.
