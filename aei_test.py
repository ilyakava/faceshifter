"""Visualize inference on a folder of images.

Example usage:
python aei_test.py -c config/p4d.24xlarge.yaml --checkpoint_path /SHARED/epoch1.ckpt --target_image_dir /SHARED/data/test-tiny/ --source_image_dir /SHARED/data/test-tiny/ --output_path /SHARED/eval/v0
"""

import argparse
import numpy as np
import imageio
from tqdm import tqdm
import os
from PIL import Image
from omegaconf import OmegaConf

import torch
from torchvision import transforms

from aei_net import AEINet

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", type=str, default="config/train.yaml",
                    help="path of configuration yaml file")
parser.add_argument("--checkpoint_path", type=str, required=True,
                    help="path of aei-net pre-trained file")
parser.add_argument("--target_image_dir", type=str, required=True,
                    help="path of preprocessed target face image")
parser.add_argument("--source_image_dir", type=str, required=True,
                    help="path of preprocessed source face image")
parser.add_argument("--output_path", type=str, default="output.png",
                    help="path of output image")
parser.add_argument("--gpu_num", type=int, default=0,
                    help="number of gpu")
args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu_num}" if torch.cuda.is_available() else 'cpu')

hp = OmegaConf.load(args.config)
model = AEINet.load_from_checkpoint(args.checkpoint_path, hp=hp)
model.eval()
model.freeze()
model.to(device)

target_images = [(os.path.join(args.target_image_dir, fn), os.path.splitext(fn)[0]) for fn in os.listdir(args.target_image_dir)]
source_images = [(os.path.join(args.source_image_dir, fn), os.path.splitext(fn)[0]) for fn in os.listdir(args.source_image_dir)]
os.makedirs(args.output_path, exist_ok=True)

pbar = tqdm(total=len(target_images) * len(source_images), desc='Testing')

with torch.no_grad():
    for target_img_path, target_name in target_images:
        for source_img_path, source_name in source_images:
            output_img_path = os.path.join(args.output_path, '{}_X_{}.jpeg'.format(source_name, target_name))
            with Image.open(target_img_path) as target_image:
                with Image.open(source_img_path) as source_image:
                    target_img = transforms.ToTensor()(target_image).unsqueeze(0).to(device)
                    source_img = transforms.ToTensor()(source_image).unsqueeze(0).to(device)
            
                    output, _, _, _, _ = model.forward(target_img, source_img)
                    output_image = transforms.ToPILImage()(output.cpu().squeeze().clamp(0, 1))
                    
                    row = [source_image, target_image, output_image]
                    row = np.hstack([np.array(myimg) for myimg in [source_image, target_image, output_image]])
                    # output.save(args.output_path)
                    imageio.imsave(output_img_path, row.astype(np.uint8))
                    
                    pbar.n += 1
                    pbar.refresh()
pbar.close()
