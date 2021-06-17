"""
Example Usage:

python preprocess.py --root /DATA --output_dir /RESULT 

Alternate Example with txt file input:

find . -type f | head | sed 's/^\./\/DATA/' > ./filelist.txt

python preprocess.py --root /DATA --output_dir /RESULT --inputs_list filelist.txt
"""


import os
import PIL
import dlib
import random
import argparse
import numpy as np
import scipy.ndimage
from PIL import Image
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Value

import torch

def _align_img_wrapper(kwargs):
    try:
        return align_img(**kwargs)
    except Exception as e:
        print('Failed for arguments {} with exception {}'.format(kwargs, str(e)))
        return 1


def align_img(img_file, output_img_name, output_dir, enable_padding, transform_size, output_size):
    global pbar, counter, detector, shape_predictor

    if counter:
        with counter.get_lock():
            counter.value += 1

    output_img = os.path.join(output_dir, output_img_name)
    
    img = dlib.load_rgb_image(img_file)
    dets = detector(img, 1)
    if not len(dets):
        return 1
    else:
        shape = shape_predictor(img, dets[0])
        points = np.empty([68, 2], dtype=int)
        for b in range(68):
            points[b, 0] = shape.part(b).x
            points[b, 1] = shape.part(b).y
        lm = points
    # lm = fa.get_landmarks(input_img)[-1]
    # lm = np.array(item['in_the_wild']['face_landmarks'])
    lm_chin          = lm[0  : 17]  # left-right
    lm_eyebrow_left  = lm[17 : 22]  # left-right
    lm_eyebrow_right = lm[22 : 27]  # left-right
    lm_nose          = lm[27 : 31]  # top-down
    lm_nostrils      = lm[31 : 36]  # top-down
    lm_eye_left      = lm[36 : 42]  # left-clockwise
    lm_eye_right     = lm[42 : 48]  # left-clockwise
    lm_mouth_outer   = lm[48 : 60]  # left-clockwise
    lm_mouth_inner   = lm[60 : 68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left     = np.mean(lm_eye_left, axis=0)
    eye_right    = np.mean(lm_eye_right, axis=0)
    eye_avg      = (eye_left + eye_right) * 0.5
    eye_to_eye   = eye_right - eye_left
    mouth_left   = lm_mouth_outer[0]
    mouth_right  = lm_mouth_outer[6]
    mouth_avg    = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    img = PIL.Image.open(img_file)
    img = img.convert('RGB')



    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
    
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    # Transform.
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

    # Save aligned image.
    img.save(output_img)

    if pbar:
        with pbar.get_lock():
            if pbar.n < counter.value:
                pbar.n = counter.value
                pbar.refresh()

    return 0

def main(args):
    global pbar, counter, detector, shape_predictor
    pbar = None
    counter = Value('i', 0)

    detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    torch.backends.cudnn.benchmark = False

    os.makedirs(args.output_dir, exist_ok=True)

    if not args.inputs_list:
        img_files = [
                    os.path.join(path, filename)
                    for path, dirs, files in os.walk(args.root)
                    for filename in files
                    if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg")
                ]
    else:
        img_files = []
        with open(args.inputs_list, 'r') as f:
            lines = [l.strip() for l in f.readlines()]

            img_files = [
                    filename
                    for filename in lines
                    if (filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg")) and os.path.exists(filename) and filename.startswith(args.root)
                ]
            assert len(lines) == len(img_files), "Input file list contained invalid images."

    assert len(img_files), "There are no images to process."
    img_files.sort()
    
    args_list = []
    for cnt, img_file in enumerate(img_files):
        if args.rename_outputs:
            output_img_name = f"{cnt:08}.png"
        else:
            # delete the root filepath from the individual filename
            img_file_no_root = img_file[len(args.root):]
            filename_parts = [p for p in img_file_no_root.split('/') if p]
            if args.outputs_prefix:
                filename_parts.insert(0, args.outputs_prefix)
            new_name = '-'.join(filename_parts)
            new_name_no_ext, _ = os.path.splitext(new_name)
            output_img_name = new_name_no_ext + ".png"


        args_list.append({
            'img_file': img_file,
            'output_img_name': output_img_name,
            'output_dir': args.output_dir,
            'enable_padding': not args.no_padding,
            'transform_size': args.transform_size,
            'output_size': args.output_size
        })

    print("{} images ready to process on {} workers".format(len(args_list), args.num_threads))
    pbar = tqdm(total=len(args_list), desc='Aligning')

    if args.num_threads > 1:
        with mp.Pool(processes=args.num_threads) as p:
            res = p.map(_align_img_wrapper, args_list)
    else:
        res = []
        for cnt, work_item in enumerate(args_list):
            res.append(_align_img_wrapper(work_item))
            pbar.n = cnt + 1
            pbar.refresh()

    pbar.n = len(args_list)
    pbar.close()

    print("Finished with {} failures for {} inputs.".format(sum(res), len(res)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--inputs_list", type=str, default='')

    parser.add_argument("--output_size", type=int, default=256)
    parser.add_argument("--transform_size", type=int, default=4096)
    parser.add_argument('--no_padding', action='store_true', help='...')
    parser.add_argument('--rename_outputs', action='store_true', help='...')
    parser.add_argument("--num_threads", type=int, default=1)
    parser.add_argument("--outputs_prefix", type=str, default='')


    args = parser.parse_args()
    main(args)