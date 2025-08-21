# Component 1: Classifier using LightGBM, K-Means, and F1 threshold tuning with GridSearchCV
import os
import numpy as np
import tensorflow as tf  # kept for pad_sequences
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import class_weight
from sklearn.cluster import KMeans
import random

print("--- Component 1: Classifier using LightGBM and K-Means ---")

# -----------------------------
# 1. Data Loading and Feature Engineering
# -----------------------------

base_dir = '/content/ADFA-LD_unzipped/ADFA-LD'
normal_train_dir = os.path.join(base_dir, 'Training_Data_Master')
normal_val_dir = os.path.join(base_dir, 'Validation_Data_Master')
attack_dir = os.path.join(base_dir, 'Attack_Data_Master')

def load_traces(directory):
    traces = []
    if not os.path.exists(directory):
        print(f"Warning: Directory not found: {directory}")
        return traces
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.txt'):
                file_path = os.path.join(root, filename)
                with open(file_path, 'r') as f:
                    try:
                        trace = [int(x) for x in f.read().split()]
                        if trace:
                            traces.append(trace)
                    except ValueError as e:
                        print(f"Skipping malformed file {file_path}: {e}")
    return traces

print("Loading data...")
normal_train_traces = load_traces(normal_train_dir)
normal_val_traces = load_traces(normal_val_dir)
attack_traces = load_traces(attack_dir)

normal_traces = normal_train_traces + normal_val_traces
print(f"Number of normal traces loaded: {len(normal_traces)}")
print(f"Number of attack traces loaded: {len(attack_traces)}")

all_traces = normal_traces + attack_traces
if not all_traces:
    print("\nError: No data loaded. Please ensure the ADFA-LD dataset is correctly placed.")
    exit()

all_syscalls = [sc for trace in all_traces for sc in trace]
syscall_vocab = sorted(list(set(all_syscalls)))
syscall_to_int = {sc: i for i, sc in enumerate(syscall_vocab)}

print(f"Total number of unique system calls: {len(syscall_vocab)}")

def traces_to_features(traces, vocab_size):
    features = np.zeros((len(traces), vocab_size), dtype=np.int32)
    for i, trace in enumerate(traces):
        for sc in trace:
            if sc in syscall_to_int:
                features[i, syscall_to_int[sc]] += 1
    return features

X_normal = traces_to_features(normal_traces, len(syscall_vocab))
X_attack = traces_to_features(attack_traces, len(syscall_vocab))

y_normal = np.zeros(len(X_normal))
y_attack = np.ones(len(X_attack))

X = np.concatenate((X_normal, X_attack))
y = np.concatenate((y_normal, y_attack))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Labels in test set: {np.unique(y_test, return_counts=True)}")

# Calculate class weights
class_weights = class_weight.compute_class_weight(
    'balanced', classes=np.unique(y_train), y=y_train
)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
print(f"Calculated class weights: {class_weights_dict}")

# -----------------------------
# 2. K-Means Feature Enrichment
# -----------------------------

print("\n--- Applying K-Means for feature enrichment ---")
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans.fit(X_train)

X_train_clusters = kmeans.predict(X_train).reshape(-1, 1)
X_test_clusters = kmeans.predict(X_test).reshape(-1, 1)

X_train_final = np.concatenate((X_train, X_train_clusters), axis=1)
X_test_final = np.concatenate((X_test, X_test_clusters), axis=1)

print(f"Final training data shape with cluster feature: {X_train_final.shape}")

# -----------------------------
# 3. LightGBM with GridSearchCV
# -----------------------------

print("\n--- Hyperparameter Tuning with GridSearchCV ---")
param_grid = {
    "num_leaves": [31, 64, 128],
    "max_depth": [-1, 10, 20],
    "learning_rate": [0.1, 0.05, 0.01],
    "n_estimators": [200, 500, 800],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "reg_alpha": [0.0, 0.1],
    "reg_lambda": [0.0, 0.1]
}

lgb_model = lgb.LGBMClassifier(objective="binary", boosting_type="gbdt",
                               class_weight="balanced", random_state=42)

grid = GridSearchCV(estimator=lgb_model, param_grid=param_grid,
                    scoring="f1", cv=3, verbose=2, n_jobs=-1)

grid.fit(X_train_final, y_train)

print("\nBest Parameters:", grid.best_params_)
print("Best F1 Score (CV):", grid.best_score_)

model = grid.best_estimator_

# -----------------------------
# 4. Evaluation with Threshold Tuning
# -----------------------------

print("\n--- Model Performance Metrics ---")

y_pred_default = model.predict(X_test_final)

print("\nDefault Threshold (0.5) Results:")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_default):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_default, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_default, zero_division=0):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred_default, zero_division=0):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_default))

y_proba = model.predict_proba(X_test_final)[:, 1]
best_f1, best_thresh = 0, 0.5

for thresh in np.arange(0.1, 0.9, 0.01):
    y_pred_thresh = (y_proba >= thresh).astype(int)
    f1 = f1_score(y_test, y_pred_thresh, zero_division=0)
    if f1 > best_f1:
        best_f1, best_thresh = f1, thresh

y_pred_best = (y_proba >= best_thresh).astype(int)

print(f"\nBest Threshold Tuning Results (threshold={best_thresh:.2f}):")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_best):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_best, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred_best, zero_division=0):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred_best, zero_division=0):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_best))

# -----------------------------
# 5. Select Malicious Log
# -----------------------------

malicious_indices = np.where(y_test == 1)[0]
selected_malicious_log = None
if len(malicious_indices) > 0:
    correctly_classified = [i for i in malicious_indices if y_pred_best[i] == 1]
    if correctly_classified:
        selected_index = random.choice(correctly_classified)
        selected_malicious_log = X_test[selected_index]
        print("\n--- Malicious Log Selected for Reinforcement Learning ---")
        print(f"Pre-processed log (first 100 values): {selected_malicious_log.tolist()[:100]}...")
    else:
        selected_malicious_log = X_test[random.choice(malicious_indices)]
        print("\nNo correctly classified malicious logs, picking random.")
        print(f"Pre-processed log (first 100 values): {selected_malicious_log.tolist()[:100]}...")
else:
    print("\nNo malicious logs found in the test set.")

global trained_classifier_model, malicious_log_for_rl, kmeans_model_for_rl
trained_classifier_model = model
malicious_log_for_rl = selected_malicious_log
kmeans_model_for_rl = kmeans

