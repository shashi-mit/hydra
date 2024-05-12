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


import joblib
import os
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import glob
import requests
import numpy as np
import csv
import glob
import os
import clip
import numpy as np
import torch
from joblib import dump, load
from PIL import Image
import glob
import os
import numpy as np
import torch
from joblib import dump, load
from PIL import Image
import av
import shutil
import time

# # Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)


xgb_model = xgb.Booster()
xgb_model.load_model('models/xgb_mc_model.json')  
le = joblib.load('models/label_encoder.job')

def create_sliding_window(descs):
    windowed_descs = []
    #offsets = range(0, len(descs)-2)
    win_sizes = range(2, len(descs))
    for ws in win_sizes:
        try:
            d2 = descs[:ws]
            d2_len = len(d2)
            win = np.mean(d2, axis=0)
            windowed_data = np.append(win, d2_len)
            windowed_descs.append(windowed_data)
        except ValueError:
            #print(f"Skipping window size {ws} due to incompatible shapes.")
            pass
    return windowed_descs

def create_sliding_window2(descs):
    windowed_descs = []
    offsets = range(0, len(descs)-2)
    win_sizes = range(2, len(descs))
    for offset in offsets:
        for ws in win_sizes:
            try:
                d2 = descs[offset:ws]
                d2_len = len(d2)
                win = np.mean(d2, axis=0)
                windowed_data = np.append(win, [d2_len, offset])
                windowed_descs.append(windowed_data)
            except ValueError:
                #print(f"Skipping window size {ws} due to incompatible shapes.")
                pass
    return windowed_descs

def read_video(video_url):
    frame_images = []
    with av.open(video_url) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            frame_images.append(frame.to_image())
    return frame_images

def process_video(video_path):
    features_list = []

    try:
        batch_size = 32
        frame_images = read_video(video_path)
        for i in range(0, len(frame_images), batch_size):
            batch_frames = frame_images[i:i+batch_size]
            frame_pil_images = [preprocess(img) for img in batch_frames]
            features = model.encode_image(torch.stack(frame_pil_images).to(device))
            f = features.cpu().detach().numpy()
            features_list.append(f)

    except Exception as e:
        print(e)

    return features_list


def download_video(url, save_path):
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        # Open a file for writing in binary mode
        with open(save_path, 'wb') as file:
            # Write the streamed content to the file
            for chunk in response.iter_content(chunk_size=1024*1024):  # Adjust chunk size to preference
                if chunk:  # filter out keep-alive new chunks
                    file.write(chunk)
        print("Download completed successfully.")
        return save_path
    else:
        print(f"Failed to download video. Status code: {response.status_code}")
        return None

# List of csv files containing the video URLs
def process_csv():
    if not os.path.exists('results'):
        os.makedirs('results')

    with open("falling_events_results.csv", "w") as results:
        with open('normal-sample-10000.csv', 'r') as file:
            reader = csv.DictReader(file)
            for n,row in enumerate(reader):
                try:
                    if n < 5000: continue
                    print("[+] Processing video: ", row["s3_unannotated_video"])
                    video_url = row["s3_unannotated_video"]
                    # Download the video   
                    video_file = download_video(video_url, 'video.mp4')
                    start_time = time.time()

                    features_list = process_video(video_file)
                    sw_features = create_sliding_window(features_list)
                    # Predict the class of the video
                    preds = []
                    dtest = xgb.DMatrix(sw_features)
                    preds = xgb_model.predict(dtest)
                    duration = time.time() - start_time
                    print(f"Duration: {duration}")
                    combined_pred = np.mean(preds, axis=0)
                    # anom_prob = int(100 * combined_pred)
                    # max_class = "Anomaly"


                    # for i, pred in enumerate(combined_pred):
                    #     print(f"{le.inverse_transform([i])[0]} : {pred}")
                    max_class = le.inverse_transform([np.argmax(combined_pred)])[0]
                    max_prob = int(np.max(combined_pred) * 100)
                    tgt_folder = os.path.join('results', max_class)
                    if not os.path.exists(tgt_folder):
                        os.makedirs(tgt_folder)
                    results.write(f"{video_url},{max_class},{max_prob}\n")
                    if max_class != "some":
                        tgt_filename = os.path.join(tgt_folder, "{}_{}_{}.mp4".format(max_class, max_prob, n))
                        shutil.move(video_file, tgt_filename)
                    #if n > 5000: break
                except Exception as e:
                    print(e)
                    continue
            #print(combined_pred)
            

def process_folder(parent_folder):
    with open("UCF101_results.csv", "w") as results:
        folders = os.listdir(parent_folder)
        matches = 0;mismatches = 0

        for folder in folders:
            #files = glob.glob(f"{foldername}/*.avi", recursive=True)
            files = glob.glob("./{}/{}/**/*.avi".format(parent_folder,folder), recursive=True)
            #print("Total files: ",folder ,len(files))
            # quit()
            for n,video_file in enumerate(files):
                try:
                    _,filename = os.path.split(video_file)
                    #print("[+] Processing video: ", video_file, n)
                    features_list = process_video(video_file)
                    sw_features = create_sliding_window(features_list)
                    # Predict the class of the video
                    preds = []
                    dtest = xgb.DMatrix(sw_features)
                    preds = xgb_model.predict(dtest)
                    combined_pred = np.mean(preds, axis=0)
                    max_class = le.inverse_transform([np.argmax(combined_pred)])[0]
                    max_prob = int(np.max(combined_pred) * 100)
                    results.write(f"{filename},{max_class},{max_prob},{folder}\n")
                    if max_class == folder:
                        matches += 1
                    else:
                        mismatches += 1

                    # tgt_folder = os.path.join('results3', max_class)
                    # if not os.path.exists(tgt_folder):
                    #     os.makedirs(tgt_folder)
                    # tgt_filename = os.path.join(tgt_folder, "{}_{}_{}.mp4".format(max_class, max_prob, n))
                    # shutil.move(video_file, tgt_filename)
                except Exception as e:
                    print(e)
                    continue
            print(f"Matches: {matches}, Mismatches: {mismatches}")
    return

#process_csv()
process_folder('UCF-101_test')
