# Setup Guide for NanoDet testing

The reference material used are [A] https://github.com/RangiLyu/nanodet/blob/main/demo/demo-inference-with-pytorch.ipynb and [B] https://github.com/RangiLyu/nanodet/blob/main/requirements.txt and [C] https://github.com/RangiLyu/nanodet?tab=readme-ov-file#model-zoo

1. Ensure a python venv has been created and activated

2. Follow the demo-inference-with-pytorch.ipynb with the following changes:

    + `pip install torch==1.11.0` instead of `pip install torch==1.7.1`

    + `pip install torchvision==0.12.0`

    + After `pip install -r requirements.txt` run `pip install --upgrade numpy==1.26.4`

3. Change `device = torch.device('cuda')` in code block 1 to `device = torch.device('cpu')` is using a cpu instead of gpu to run model.

4. In code block 4, update all file paths.

    + Get the model weights (a .pth file) using the reference [C]. Once downloaded, update the `model_path`

    + Update `config_path` by matching the model associated with the downloaded weights in the previous step to the model file found in the NanoDet repository under the directory `config`

    + Update `image_path` to be the path to the image being tested.



# Steps Forward

1. Export .pth model test from `NanoDet_test_deployment.ipynb` to OpenVINO friendly format

2. Leverage OpenVINO inference engine on same image used in .pth framework

3. Use sample video in OpenVINO inference engine


