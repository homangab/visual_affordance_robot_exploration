# Visual Affordance Prediction for Guiding Robot Exploration

This repo contains code for the paper Visual Affordance Prediction for Guiding Robot Exploration

### Installation 

First create a conda environment and install PyTorch
```
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```

Then do the following to install all dependencies in the conda environment
``` 
python setup.py build develop 
```

Install pyrealsense if using an Intel RealSense camera
```
pip install pyrealsense2
```

Follow the installation instructions for FrankaPy in `https://github.com/iamlab-cmu/frankapy`


### For Robot Exploration and Policy Training

```
python scripts/bc.py --config-file configs/Transformer.yaml
``

Please refer to the file `scripts/bc.py` to change configs like epochs, horizon etc. If you want to change policy architectures and other details, modify `scripts/policy.py` as needed. 


### For Training Affordance Model

To train VQVAE
```
python tools/train_net.py --config-file configs/VQVAE.yaml --num-gpus 2 OUTPUT_DIR experiments/vqvae
```

To generate latent codes
```
python tools/train_net.py --eval-only --config-file configs/VQVAE.yaml OUTPUT_DIR experiments/vqvae TEST.EVALUATORS "CodesExtractor" 
```

To train Transformer

```
 python tools/train_net.py --config-file configs/Transformer.yaml --num-gpus 8 OUTPUT_DIR experiments/transformer
```


### References
If you find this repository helpful, please consider citing our paper

```
@article{BharadhwajVisual,	
  Author = {Homanga Bharadhwaj and Abhinav Gupta and Shubham Tulsiani},	
  Journal={IEEE International Conference on Robotics and Automation (ICRA)},	
  Year = {2023},	
  Title = {Visual Affordance Prediction for Guiding Robot Exploration}	
}    	
```

### Contact
If you have any questions about the repo or the paper, please contact Homanga Bharadhwaj at `hbharadh AT cs DOT cmu DOT edu` 