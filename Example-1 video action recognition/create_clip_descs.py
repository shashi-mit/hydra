###
#  * Copyright (c) 2024 Shashi Kant
#  *
#  * Permission is hereby granted, free of charge, to any person obtaining a copy
#  * of this software and associated documentation files (the "Software"), to deal
#  * in the Software without restriction, including without limitation the rights
#  * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  * copies of the Software, and to permit persons to whom the Software is
#  * furnished to do so, subject to the following conditions:
#  *
#  * The above copyright notice and this permission notice shall be included in all
#  * copies or substantial portions of the Software.
#  *
#  * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  * SOFTWARE.

import glob
import os
import clip
import numpy as np
import torch
from joblib import dump, load
from PIL import Image

# # Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

import av

def read_video(video_url, skip_secs=-1.0):
    frame_images = []
    #content = av.datasets.curated(video_url)
    with av.open(video_url) as container:
        # Signal that we only want to look at keyframes.
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            frame_images.append(frame.to_image())
    return frame_images

def process_video(video_path, tgt_folder):
    try:
        

        skips = 0
        tgt_filename = os.path.join(tgt, os.path.splitext(os.path.basename(video_path))[0])
        if os.path.isfile(tgt_filename):
            # print("[*] Skipping: ", video_path)
            skips += 1
            return
        batch_size = 32
        frame_images = read_video(video_path, skip_secs=-1.0)
        features_list = []
        for i in range(0, len(frame_images), batch_size):
            batch_frames = frame_images[i:i+batch_size]
            frame_pil_images = [preprocess(img) for img in batch_frames]
            features = model.encode_image(torch.stack(frame_pil_images).to(device))
            f = features.cpu().detach().numpy()
            features_list.append(f)

        # Save the concatenated features to a file
        dump(features_list, "{}/{}.job".format(tgt, os.path.splitext(os.path.basename(video_path))[0]))

    except Exception as e:
        print(e)

    return



# skips = 0
paths = ["./videos"]
for path in paths:
    src_folders = os.listdir(path)
    print("[+] Total folders: ", len(src_folders))
    for folder_name in src_folders:
        tgt = os.path.join('./descriptors5', folder_name)
        if not os.path.isdir(tgt):
            os.mkdir(tgt)
        else:
            print("[*] Folder exists: ", tgt)
            continue

        video_files = glob.glob("{}/{}/**/*.*".format(path,folder_name), recursive=True)
        print("[+] ",folder_name ,len(video_files))
        for video_path in video_files:
            print("[+] Processing: ", video_path)
            process_video(video_path, tgt)
