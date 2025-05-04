# NaviGatr Demonstration Overview

## 🧭 Demonstration Setup

A mock path was created with:

- **Start Line** and **Finish Line**
- **Chair** at the **10 o'clock position**, **1 meter away**
- **Person** at the **1 o'clock position**, **1 meter past the chair**
- **Ground Obstacle** to test edge-case detection

This layout mimics real-world navigation challenges for visually impaired users, including vertical and horizontal obstacles.

---

## 🎤 General Talking Points (2 minutes)

Our system is designed to assist visually impaired individuals in navigating real-world environments using a wearable computer vision system.

**Workflow Summary:**
1. The **camera** captures a frame.
2. The **object detection model** identifies objects and outputs:
   - Object **label** (e.g., chair, person)
   - Object **bounding box coordinates** (x, y pixels)
3. These coordinates are used to extract corresponding pixels from the **depth map**, which returns **distance values**.
4. If the object is a **person** and is within **1 meter**, the **face region** from the bounding box is passed to the **emotion model**.
5. The **emotion model** classifies the person's emotion — this enables **social awareness** for blind users.

This modular approach keeps the system efficient and scalable.

---

## 🧠 Micah (2 minutes)

**Object Detection Model Overview:**
- Explored multiple models:
  - **YOLOv5** – fast and accurate, good for edge devices.
  - **NanoDet** – lightweight but less accurate.
  - **YOLOv8 ONNX export** – chosen for our setup due to its ONNX compatibility and optimized inference.
- Decision Criteria:
  - **Speed**: must run every ~3 seconds.
  - **Accuracy**: correctly detect people, chairs, ground obstacles.
  - **Compatibility** with Coral TPU and Raspberry Pi.

---

## 🌊 Eliav (2 minutes)

**Depth Detection Model Overview:**
- Explored:
  - **MiDaS** – high accuracy but heavy.
  - **FastDepth** – lighter, but lower resolution.
  - **Intel RealSense** (not used) – hardware-based, not ideal for wireless wearables.
- Our current model runs **in the cloud** due to TPU limitations.
- Outputs a **2D array** with distance values indexed by pixel positions.
- Pixel-aligned with object model outputs for efficient fusion.

Future iterations may use **on-device LiDAR** or more capable microcontrollers with built-in depth sensing.

---

## 😄 Naitik (2 minutes)

**Emotion Detection Model:**
- Trained a custom **8-layer CNN** using the **FER2013 dataset**:
  - Contains **20000 greyscale** images of size **224x224**
  - Covers 8 major emotions: angry, disgust, fear, happy, sad, surprise, neutral, contempt
- Key optimization:
  - Initially used full frame (ineffective)
  - Now extracts only **face region** from object detection to improve inference
- **Runs on Coral TPU** for speed; inference every ~3 seconds.

**Hardware Design:**
- CAD designed for **comfort and usability**:
  - **Raspberry Pi 4 (64-bit OS)**, screen, fans on one side
  - **Battery** on the opposite side for balance
  - **Camera** mounted at eye level
- Uses **GPIO-based screen** that switches to HDMI automatically when connected
- Fans enclosed to prevent hair entanglement
- **Cloud-based depth model**, but local object and emotion models
- Modular, fast (~3s per inference cycle), and practical for real-world mobility

---

