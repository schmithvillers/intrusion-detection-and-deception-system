# Classifier with Hyperparameter Effect Visualization - with graph for more than 10 runs
import os
import numpy as np
import tensorflow as tf
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import class_weight
from sklearn.cluster import KMeans
import random
import matplotlib.pyplot as plt

print("--- Component: LightGBM + KMeans + Hyperparameter Visualization ---")

# -----------------------------
# 1. Data Loading
# -----------------------------
base_dir = '/content/ADFA-LD_unzipped/ADFA-LD'
normal_train_dir = os.path.join(base_dir, 'Training_Data_Master')
normal_val_dir = os.path.join(base_dir, 'Validation_Data_Master')
attack_dir = os.path.join(base_dir, 'Attack_Data_Master')

def load_traces(directory):
    traces = []
    if not os.path.exists(directory):
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
                    except ValueError:
                        continue
    return traces

normal_train_traces = load_traces(normal_train_dir)
normal_val_traces = load_traces(normal_val_dir)
attack_traces = load_traces(attack_dir)
normal_traces = normal_train_traces + normal_val_traces

all_traces = normal_traces + attack_traces
all_syscalls = [sc for trace in all_traces for sc in trace]
syscall_vocab = sorted(list(set(all_syscalls)))
syscall_to_int = {sc: i for i, sc in enumerate(syscall_vocab)}

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

# KMeans feature enrichment
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans.fit(X_train)
X_train_clusters = kmeans.predict(X_train).reshape(-1, 1)
X_test_clusters = kmeans.predict(X_test).reshape(-1, 1)
X_train_final = np.concatenate((X_train, X_train_clusters), axis=1)
X_test_final = np.concatenate((X_test, X_test_clusters), axis=1)

# -----------------------------
# 2. Function to Evaluate Hyperparameters
# -----------------------------
def evaluate_hyperparams(hyperparams, runs=10):
    """Run multiple experiments with varying hyperparameters and return metrics."""
    results = {hp: {"accuracy": [], "precision": [], "recall": [], "f1": []} for hp in hyperparams}
    
    for hp, values in hyperparams.items():
        for val in values:
            acc_list, prec_list, rec_list, f1_list = [], [], [], []
            
            for run in range(runs):
                params = {
                    "objective": "binary",
                    "boosting_type": "gbdt",
                    "class_weight": "balanced",
                    "random_state": 42 + run,
                }
                params[hp] = val
                
                model = lgb.LGBMClassifier(**params)
                model.fit(X_train_final, y_train)
                y_pred = model.predict(X_test_final)
                
                acc_list.append(accuracy_score(y_test, y_pred))
                prec_list.append(precision_score(y_test, y_pred, zero_division=0))
                rec_list.append(recall_score(y_test, y_pred, zero_division=0))
                f1_list.append(f1_score(y_test, y_pred, zero_division=0))
            
            # store average across runs
            results[hp]["accuracy"].append(np.mean(acc_list))
            results[hp]["precision"].append(np.mean(prec_list))
            results[hp]["recall"].append(np.mean(rec_list))
            results[hp]["f1"].append(np.mean(f1_list))
    
    return results

# -----------------------------
# 3. Function to Plot Results
# -----------------------------
def plot_hyperparameter_effects(results, hyperparams):
    """Plot graphs of metrics vs hyperparameters."""
    for hp, metrics in results.items():
        values = hyperparams[hp]
        plt.figure(figsize=(10, 6))
        plt.plot(values, metrics["accuracy"], marker="o", label="Accuracy")
        plt.plot(values, metrics["precision"], marker="s", label="Precision")
        plt.plot(values, metrics["recall"], marker="^", label="Recall")
        plt.plot(values, metrics["f1"], marker="x", label="F1-Score")
        plt.title(f"Effect of {hp} on Evaluation Metrics")
        plt.xlabel(hp)
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True)
        plt.show()

# -----------------------------
# 4. Run Experiments
# -----------------------------
hyperparams_to_test = {
    "num_leaves": [16, 32, 64, 128, 256],
    "max_depth": [-1, 5, 10, 20, 40],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "n_estimators": [100, 200, 500, 800, 1000]
}

results = evaluate_hyperparams(hyperparams_to_test, runs=10)
plot_hyperparameter_effects(results, hyperparams_to_test)
