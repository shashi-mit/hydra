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
import pprint as pp
import numpy as np

#sliding window implementation containing windowed-mean and including window size and offset
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

#sliding window implementation containing windowed-mean and only including window size
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
# Load sliding window descriptors from pre-created sliding window joblib files
def load_descs():
    X=[];y=[];y_dict={}
    subfolders = [f.path for f in os.scandir('./sw_descriptors') if f.is_dir()]
    for subfolder in subfolders:
        jobfiles = glob.glob("./{}/**/*.*".format(subfolder), recursive=True)
        print("[+] Processing: ", subfolder, len(jobfiles))
        for jobfile in jobfiles:
            win_descs = joblib.load(jobfile)
            X.extend(win_descs)
            cname = os.path.basename(subfolder)
            y.extend([cname] * len(win_descs))
            if cname not in y_dict:
                y_dict[cname] = 0
            y_dict[cname] += len(win_descs)

    return X,y,y_dict

X,y,y_dict = load_descs()

# print the class distribution
pp.pprint(y_dict)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


# Split the data into training and testing sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

#adjust class weights to handle imbalanced classes
weights = np.zeros_like(y_train)
unique_classes = np.unique(y_train)
for cls in unique_classes:
    class_weight = 1.0 / np.mean(y_train == cls)
    weights[y_train == cls] = class_weight

# Prepare data in DMatrix format, which is optimized for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train,weight=weights)
dvalid = xgb.DMatrix(X_valid, label=y_valid)  # Assuming X_valid and y_valid are already created

# Define the parameters for the training process in a dictionary
params = {
    "objective": "multi:softprob",        # Objective for multiclass classification
    "eval_metric": "mlogloss",            # Metric for multiclass classification
    "learning_rate": 0.1,                 # Learning rate
    "max_depth": 10,                       # Depth of the trees
    "num_class": len(np.unique(y_train)), # Number of unique classes
    "verbosity": 2,                       # Verbosity of printing messages
    "device": "cuda"                      # Use CUDA for GPU computation
}

# Specify number of boosting rounds (trees to build)
num_boost_round = 1000

# Specify early stopping rounds
early_stopping_rounds = 10

# Train the model with early stopping
model = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    evals=[(dvalid, 'validation')],  # Provide validation set for monitoring
    early_stopping_rounds=early_stopping_rounds,
    verbose_eval=True  # Optional: provide more verbose output
)

# evaluate the model or perform further actions
print(f"[+] Training complete. Best iteration: {model.best_iteration}")
# Prediction on validation data
y_pred = model.predict(dvalid)
# Convert probabilities to class predictions
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate metrics
accuracy = accuracy_score(y_valid, y_pred_classes)
precision = precision_score(y_valid, y_pred_classes, average='macro')  # 'macro' average for multiclass
recall = recall_score(y_valid, y_pred_classes, average='macro')  # 'macro' average for multiclass

print("[+] Validation Metrics:")
print(f"    Accuracy: {accuracy:.4f}")
print(f"    Precision: {precision:.4f}")
print(f"    Recall: {recall:.4f}")

# Optional: Detailed classification report
# Note: This will provide detailed metrics for each class, in most cases, this would appear to overfit
# however the results should speak for themselves when running against new data
# print("\n[+] Detailed Classification Report:")
# print(classification_report(y_valid, y_pred_classes))

# Save the model
model.save_model('./models/xgb_mc_model.json')
joblib.dump(label_encoder, './models/label_encoder.job')
print("[+] Model saved.")