## Overview

To narrow our search for applicable object detection models, the following key considerations are made:

1. Fast response (close to real-time detection)
2. Detection of moving/static objects (Increase usefulness for our depth detection)
3. Detection of specific set of objects (not all objects) catered towards pedestrians

Additionally, the model should be implementable on the follow device options:

1. Mobile devices
2. Edge devices (Jetson, RaspberryPi, etc.)
3. Smart glasses

These considerations narrows the scope of research.

When considering fast models for object detection, YOLO is a well established model framework. The following YOLO  YOLO (You Only Look Once) frameworks have potential for our project:

+ YOLOv8
+ YOLOv8-Nano 
+ YOLO-NAS
+ YOLOv5
+ YOLOv4-Tiny
+ NanoDet

The following outlines each listed model's features + applications.

## YOLOv8

Features:

+ State-of-the-art real-time object detection.
+ Improved backbone and neck architecture over YOLOv5.
+ Supports object detection, segmentation, and classification.
+ Efficient model scaling for different versions (Nano, Small, Medium, Large, XLarge).
+ Stronger anchor-free detection mechanism.
+ Optimized for deployment on edge devices.

Applications:

+ Autonomous vehicles.
+ Smart surveillance.
+ Retail analytics.
+ Industrial defect detection.
+ Assistive navigation for visually impaired users.



## YOLOv8-Nano

Features:

+ Extremely lightweight variant of YOLOv8.
+ Designed for mobile and edge AI applications.
+ Lower computational cost with minimal accuracy drop.
+ Efficient in real-time processing on embedded devices.

Applications:

+ AI-powered smart glasses for visually impaired users.
+ Drones and robotics with low-power consumption.
+ IoT-based security and monitoring.
+ Mobile applications requiring object detection.



## YOLO-NAS

Features:

+ Neural Architecture Search (NAS) optimized version of YOLO.
+ Highly efficient with better accuracy vs. speed trade-off.
+ Designed for real-time deployment on edge devices.
+ Improved detection robustness for small and occluded objects.
+ Supports multi-task learning (detection, segmentation).

Applications:

+ Edge AI applications (e.g., smart cameras, embedded systems).
+ Traffic monitoring and vehicle counting.
+ Smart city infrastructure.
+ Medical image analysis (e.g., tumor detection).



## 4. YOLOv5
Features:

+ Highly popular and widely used for object detection.
+ Multiple model sizes: Nano, Small, Medium, Large, Extra-Large.
+ Good balance between speed and accuracy.
+ Robust pre-trained models available.
+ Easy to fine-tune and deploy.

Applications:

+ Automated security systems.
+ Drone-based object detection.
+ Retail product recognition.
+ Waste sorting in recycling plants.



## YOLOv4-Tiny

Features:

+ Lightweight version of YOLOv4 for embedded systems.
+ Lower latency and power consumption.
+ Uses CSPDarknet backbone for better efficiency.
+ Faster inference compared to YOLOv4 but with reduced accuracy.

Applications:

+ Real-time video surveillance.
+ Autonomous drones with limited hardware.
+ Mobile-based object detection applications.
+ Wildlife monitoring in low-power setups.


## NanoDet

Features:

+ Ultralightweight object detection model.
+ Optimized for mobile CPUs and ARM-based devices.
+ Anchor-free detection with minimal computational overhead.
+ Lower accuracy compared to larger YOLO models but highly efficient.

Applications:

+ Embedded AI systems (e.g., Raspberry Pi, Jetson Nano).
+ Low-power edge devices for real-time monitoring.
+ Augmented reality applications with object detection.
+ IoT-based automation and smart home integration.


Here's a comparative chart for the features of the mentioned object detection models:  

| Feature          | YOLOv8        | YOLOv8-Nano | YOLO-NAS      | YOLOv5        | YOLOv4-Tiny   | NanoDet       |
|-----------------|--------------|-------------|--------------|--------------|--------------|--------------|
| **Model Type**  | Anchor-free  | Anchor-free | NAS-optimized | Anchor-based | Anchor-based | Anchor-free  |
| **Speed**       | Fast         | Very Fast   | Fast         | Fast         | Very Fast    | Extremely Fast |
| **Accuracy**    | High         | Moderate    | Very High    | High         | Moderate     | Moderate     |
| **Size**        | Medium-Large | Small       | Medium       | Medium       | Small        | Very Small   |
| **Efficiency**  | High         | Very High   | High         | Moderate     | High         | Extremely High |
| **Edge AI Suitability** | Moderate | Excellent | Good         | Moderate     | Excellent    | Excellent    |
| **Training Complexity** | Moderate  | Low       | High        | Low          | Low          | Low          |
| **Best for Low-Light** | Yes       | Yes       | Yes         | Moderate     | No           | No           |
| **Best for Mobile/IoT** | No       | Yes       | No          | No           | Yes          | Yes          |
| **Best for Large Datasets** | Yes   | No        | Yes         | Yes         | No           | No           |


These ratings are given based on benchmark assessments found in the sources in the source section of this document.

## Benchmark Assessment

### Speed (Inference Time on Edge/GPUs)

Measured in FPS (Frames Per Second) or latency (ms per image).

YOLOv8-Nano: ~5ms per image
YOLOv4-Tiny: ~4-5ms per image
NanoDet: ~3-4ms per image
YOLO-NAS: ~10ms per image

### Accuracy (Mean Average Precision - mAP on COCO dataset)

mAP measures object detection performance.
Larger, more advanced architectures perform better due to improved feature extraction.

YOLO-NAS: 55-57%
YOLOv8: 50-54%
YOLOv5: 48-52%
YOLOv4-Tiny: 35-40%
NanoDet: 30-35%

### Model Size (File Size & Parameter Count)

Smaller models are better for edge devices (smaller file size, lower memory usage).
Measured in Megabytes (MB) or number of parameters.

NanoDet: ~1 MB
YOLOv8-Nano: ~3-4 MB
YOLOv4-Tiny: ~6 MB
YOLOv5-Small: ~14 MB
YOLO-NAS: ~30 MB


### Efficiency (Computational Resource Usage)

Efficiency = Accuracy vs. Compute Trade-off.
Smaller models are efficient, larger models demand more power.

NanoDet runs efficiently on ARM CPUs & Raspberry Pi.
YOLOv8-Nano is optimized for mobile but keeps some accuracy.
YOLO-NAS needs GPUs or TPUs to perform well.

### Edge AI Suitability (Deployment on Mobile & IoT Devices)

Can the model run smoothly on mobile CPUs (e.g., Snapdragon, Jetson Nano)?

NanoDet runs on Raspberry Pi, Jetson Nano
YOLOv8-Nano works on Android/iOS
YOLO-NAS & YOLOv8 need stronger hardware

### Low-Light Performance (Robustness in Dark Environments)

Low-light performance depends on feature extraction quality.
Larger models capture more details, improving detection in the dark.

YOLOv8 handles low-light well due to feature enhancement
YOLOv4-Tiny struggles due to fewer layers capturing fine details


## Existing Datasets


### Cityscapes Dataset
https://www.cityscapes-dataset.com/
![alt text](image-2.png)

### Kaist Dataset
https://github.com/SoonminHwang/rgbt-ped-detection
![alt text](image-1.png)

### FLIR Dataset
https://www.flir.com/oem/adas/adas-dataset-form/
![alt text](image.png)


+ https://www.nightowls-dataset.org/
+ https://github.com/cs-chan/Exclusively-Dark-Image-Dataset
+ https://www.kaggle.com/datasets/kalvinquackenbush/caltechpedestriandataset


## Above models in use (and their limitations)

## Sources

https://www.mdpi.com/2073-431X/13/12/336
https://www.mdpi.com/2504-4990/5/4/83
https://github.com/srebroa/awesome-yolo
https://github.com/roboflow/notebooks/discussions/125
https://www.vietanh.dev/blog/yolo-nas
https://www.youtube.com/watch?pp=ygUII3lvbG92OGw%3D&v=_ON9oiT_G0w
https://learnopencv.com/yolo-nas/
https://thefuturefeed.medium.com/yolo-nas-decis-new-sota-object-detection-model-that-outperforms-yolov5-yolov6-yolov7-and-yolov8-a3fc1785320a
https://www.researchgate.net/publication/379748121_YOLOv5_vs_YOLOv8_Performance_Benchmarking_in_Wildfire_and_Smoke_Detection_Scenarios
https://blog.roboflow.com/guide-to-yolo-models/
