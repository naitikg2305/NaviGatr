# Introduction

This file will discuss the object detection component of the NaviGatr. The codebase structure, adjustable settings, constraints, product specifications, API instructions, and interfacing with other NaviGatr components will be discussed.

The outline is as follows:

1. Object detection component purpose

2. Object detection component process overview

3. Performance specifications

4. Constraint considerations

5. Component's application interface

6. Adjustable settings

7. Interface protocols to other components

8. Object detection codebase structure

</br></br>

# Object Detection Component Purpose



# Object Detection Component Process Overview



# Performance Specifications



# Constraint Considerations



# Component's Application Interface



# Adjustable Settings

The objection detection model has a config.json file that hold all knobs and switches for customized behavior. The following are the available fields:

+ config_file - holds the path to the model's description of architecture (.yml file)

+ model_file - holds the path to the model weights (.pth file)

+ ip_cam_addr - holds the address to the IP camera's live-feed

+ ip_cam_toggle - determines if model is provided an IP Camera (true) for input or PiCam (false)

+ test_toggle - determines whether to enable test features


# Interface Protocols to Other Components

To run as a standalone model, simply invoke the following:

1. Activate the environment using NaviGatr/docs/environment.yml. I called this environment 'navi_env'

    `conda env create --file environment.yml`
    
    then

    `conda activate navi_env`

2. `git clone https://github.com/RangiLyu/nanodet.git`

3. `cd nanodet`

4. `python setup.py develop`

5. `python -m src.obj_detect.run_model.py` from Navigatr root directory

When not standalone, the model will comunicate and be called over interfaces.


# Object Detection Codebase Structure

