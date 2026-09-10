# Visual Inspection Architecture

`Camera/Image Upload -> S3 -> preprocessing/inference -> SageMaker endpoint -> defect class + confidence -> CloudWatch`

For a larger real-world version, replace the synthetic image set with an industrial defect dataset and initialize MobileNetV2/EfficientNet with ImageNet weights.
