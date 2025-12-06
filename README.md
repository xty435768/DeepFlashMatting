# Let Your Light Shine: Foreground Portrait Matting via Deep Flash Priors (TMLR 2025)
<p align="center">
    <a href="https://openreview.net/forum?id=vxUiVJp2eM">
        <img src="https://img.shields.io/badge/Paper-OpenReview-blue.svg"/>
    </a>
    <a href="https://youtube.com/">
        <img src="https://img.shields.io/badge/Video-Youtube-red"/>
    </a>   
</p>



<!-- ![Teaser]() -->
<img src="https://shengfenghe.github.io/images/TMLR26A.jpg" alt="drawing" width="500" style="display: block; margin: auto; "/>


This is the implementation of the paper ``Let Your Light Shine: Foreground Portrait Matting via Deep Flash Priors``.

## Installation Instructions

To set up the environment for this project, follow these steps:

1. **Create a Conda Environment**  
   Create a new Conda environment with Python 3.11:  
   ```bash
   conda create -n fnfmat python=3.11
   ```

2. **Install PyTorch**  
   Install PyTorch environment. This code is tested with ``torch==2.3.0`` and ``CUDA 11.8``.
   ```bash
   pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Install Additional Dependencies**  
   Install the remaining required libraries listed in `requirements.txt`:  
   ```bash
   pip install -r requirements.txt
   ```

4. **Install spconv**  
   Install the `spconv` library, ensuring compatibility with your CUDA version. Refer to the official [spconv GitHub repository](https://github.com/traveller59/spconv) for detailed installation instructions specific to your CUDA version.


## Download Checkpoints
The checkpoints are published on [HERE](https://www.modelscope.ai/models/cstyxiang/DeepFlashMatting). Download the checkpoint `best_model.pth` to the `checkpoints/` folder.

## Inference on Single Image Pair
To perform inference on a single image pair, use the following command:
```bash
python inference_single_fnf_image_pair.py \
    --flash-img-path demo_img_in/f.jpg \
    --noflash-img-path demo_img_in/nf.jpg \
    --save-path demo_img_out/out.jpg
```
The result will be saved to ``demo_img_out``.

## Dataset Access
Please email to [xty435768@gmail.com](mailto:xty435768@gmail.com) to request access to the FNF dataset proposed in this work.