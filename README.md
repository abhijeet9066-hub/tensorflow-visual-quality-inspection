# TensorFlow Visual Quality Inspection

Computer-vision portfolio project that classifies product images as **normal, crack, scratch or dent**. Includes a small synthetic demo image dataset so the repository is runnable immediately.

## Skills demonstrated
TensorFlow/Keras, CNNs, transfer-learning workflow, augmentation, image classification, model inference and AWS deployment design.

## Run
```bash
pip install -r requirements.txt
python src/train.py
python src/predict.py data/images/crack/crack_000.png
```

## AWS target
S3 for image ingestion, SageMaker for hosted inference and CloudWatch for monitoring. See `docs/architecture.md`.
