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
from random import shuffle
#from imblearn.over_sampling import SMOTE


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
    X=[];y=[];y_dict={}
    subfolders = [f.path for f in os.scandir('./sw_descriptors') if f.is_dir()]
    for subfolder in subfolders:
        jobfiles = glob.glob("./{}/**/*.*".format(subfolder), recursive=True)
        print("[+] Processing: ", subfolder, len(jobfiles))
        for jobfile in jobfiles:
            win_descs = joblib.load(jobfile)
            # shuffle(win_descs)
            # if "falls" not in jobfile: #["falls","fall","lying_down"]:
            #     win_descs = win_descs[:50]

            #win_descs = create_sliding_window(descs)
            #print("[+] Windowed descs: ", len(win_descs), len(descs))
            X.extend(win_descs)
            cname = os.path.basename(subfolder)
            y.extend([cname] * len(win_descs))
            if cname not in y_dict:
                y_dict[cname] = 0
            y_dict[cname] += len(win_descs)

    return X,y,y_dict

X,y,y_dict = load_descs()


pp.pprint(y_dict)
#quit()

# # print(set(y))

# quit()

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


# Split the data into training and testing sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# smote = SMOTE()
# X_train_res, y_train_res = smote.fit_resample(X_train, y_train)


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

# Optionally, evaluate the model or perform further actions
print(f"[+] Training complete. Best iteration: {model.best_iteration}")
# Prediction on validation data
y_pred = model.predict(dvalid)
# Convert probabilities to class predictions
y_pred_classes = np.argmax(y_pred, axis=1)

# Calculate metrics
accuracy = accuracy_score(y_valid, y_pred_classes)
precision = precision_score(y_valid, y_pred_classes, average='macro')  # 'macro' average for multiclass
recall = recall_score(y_valid, y_pred_classes, average='macro')  # 'macro' average for multiclass

print(f"[+] Training complete. Best iteration: {model.best_iteration}")
print("[+] Validation Metrics:")
print(f"    Accuracy: {accuracy:.4f}")
print(f"    Precision: {precision:.4f}")
print(f"    Recall: {recall:.4f}")

# Optional: Detailed classification report
print("\n[+] Detailed Classification Report:")
print(classification_report(y_valid, y_pred_classes))

# Save the model
model.save_model('./models/xgb_mc_model.json')
joblib.dump(label_encoder, './models/label_encoder.job')
print("[+] Model saved.")