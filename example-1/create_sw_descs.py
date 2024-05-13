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
from sklearn.metrics import accuracy_score
import glob
from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report
import numpy as np

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

# List of joblib files containing the datasets
def load_descs():
    X=[];y=[]
    subfolders = [f.path for f in os.scandir('./descriptors') if f.is_dir()]
    for subfolder in subfolders:
        jobfiles = glob.glob("./{}/**/*.*".format(subfolder), recursive=True)
        subfolder_name = os.path.basename(subfolder)
        tgt_folder = os.path.join('./sw_descriptors', subfolder_name)
        if not os.path.isdir(tgt_folder):
            os.mkdir(tgt_folder)

        print("[+] Processing: ", subfolder, len(jobfiles))
        cname = 1
        for jobfile in jobfiles:
            try:
                descs = joblib.load(jobfile)
                win_descs = create_sliding_window(descs)
                _,filename = os.path.split(jobfile)
                joblib.dump(win_descs, "{}/{}".format(tgt_folder, filename))
            except Exception as e:
                print(e)
    return X,y

X,y = load_descs()
