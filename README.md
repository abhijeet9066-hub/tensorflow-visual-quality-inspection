# TensorFlow Visual Quality Inspection

Computer-vision portfolio project that classifies product images as **normal, crack, scratch or dent**. Includes a small synthetic demo image dataset so the repository is runnable immediately.

## Skills demonstrated

TensorFlow/Keras, CNNs, transfer-learning workflow, augmentation, image classification and model inference.

**AWS deployment architecture:** S3, SageMaker and CloudWatch architecture.

> **Scope note:** the AWS components are a target deployment design. This repository does not claim a verified live AWS deployment.

## Run

```bash
pip install -r requirements.txt
python src/train.py
python src/predict.py data/images/crack/crack_000.png
```

## AWS deployment architecture

S3 for image ingestion/artifacts, SageMaker for hosted inference and CloudWatch for monitoring. See `docs/architecture.md`.

## CV-ready summary

Built a TensorFlow/Keras computer-vision defect-classification workflow with augmentation and designed an AWS deployment architecture using S3, SageMaker and CloudWatch. The AWS layer is an architecture reference rather than a verified production deployment.
