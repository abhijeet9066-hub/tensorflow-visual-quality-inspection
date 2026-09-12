# Visual Inspection AWS Deployment Architecture

**Target architecture design only — not a verified live AWS deployment.**

```text
Camera / Image Upload
  -> S3
  -> preprocessing / inference
  -> SageMaker endpoint
  -> defect class + confidence
  -> CloudWatch
```

For a larger real-world version, replace the synthetic image set with an industrial defect dataset and initialize MobileNetV2/EfficientNet with ImageNet weights.

Production implementation would additionally require IAM least privilege, encryption, private networking, model registry/versioning, monitoring/alerting, artifact governance and cost controls.
