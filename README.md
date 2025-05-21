![Navigatr Logo](imgs/logo.png)
UMD, ENEE408 Capstone Project Spr. '25


NaviGatr is a Machine Learning driven approach to footpath navigation.

# How it Works

The application is designed to ingest frames (from images or video feed), run model inferencing on specified frames, and output useful information related to type of in-view objects, how far in-view objects are, emotions of interacting persons, where to go to avoid objects, and where to go to become closer to speified destination.

The application is designed to be prototyped on Raspberry Pi5 Single-Board Conmputer with periphials that includes speaker, fan (cooling system for hardware), Google Coral TPU, and a powerbank. The desired end product is an implementation using state-of-the-art smart glasses such as Meta's Orion    glasses.

# Python version 
Its important to use python3.9 for this project for compatibility reasons with the coral TPU. 

# Creating a virtual environment and installing dependencies 
Its important to create a virtual environment using "python -m venv <envname>". This environment then needs to be activated by running command "source envname/bin/activate". Then you can run pip install requirements.txt to install necessary python libraries. 

# Installing PyCoral and TFlite
These are the two libraries not available through pip or wget. These would be found in the project repo and can simply be pip installed when in the same directory. 

# How to Run

To run this threaded code, you need to simply need to run run_Pi_Thread.py in the src directory of Navigator. This calls the three major models. If you want to learn more about these models, you can go into the specific subfolders.

# File directory (for reference)
```
NaviGatr
    ├── docs
    │   ├── LogoCreator.docx
    │   ├── Milestone_Report.pdf
    │   └── Milestone_Review.pptx.pdf
    ├── pycoral-2.0.0-cp39-cp39-linux_aarch64.whl
    ├── README.md
    ├── requirements.txt
    ├── src
    │   ├── distance
    │   │   ├── cert.pem
    │   │   ├── get_depth.py
    │   │   └── __pycache__
    │   │       └── get_depth.cpython-39.pyc
    │   ├── EmotionDetec
    │   │   ├── emotion_tflite.py
    │   │   ├── fer_emotion_model_int8_edgetpu.tflite
    │   │   └── __pycache__
    │   │       └── emotion_tflite.cpython-39.pyc
    │   ├── objectDetection
    │   │   ├── coral_models
    │   │   │   ├── coco_labels.txt
    │   │   │   └── ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite
    │   │   ├── main.py
    │   │   └── __pycache__
    │   │       └── main.cpython-39.pyc
    │   ├── __pycache__
    │   │   └── text_to_speech.cpython-39.pyc
    │   ├── run_Pi_Thread.py
    │   └── text_to_speech.py
    └── tflite_runtime-2.5.0.post1-cp39-cp39-linux_aarch64.whl
```



# Sources 
TFLite: https://pypi.org/project/tflrt/#files
PyCoral: https://github.com/google-coral/pycoral/releases