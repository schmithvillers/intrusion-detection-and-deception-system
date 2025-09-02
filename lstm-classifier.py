import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import class_weight

print("--- Component 1: Neural Network Classifier ---")

base_dir = '/content/sample_data/ADFA-LD/ADFA-LD'
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
all_syscalls = [sc for trace in all_traces for sc in trace]
syscall_vocab = sorted(list(set(all_syscalls)))
syscall_to_int = {sc: i + 1 for i, sc in enumerate(syscall_vocab)}
int_to_syscall = {i + 1: sc for i, sc in enumerate(syscall_vocab)}

vocab_size = len(syscall_vocab) + 1
print(f"Total number of unique system calls: {vocab_size - 1}")

X_normal = [[syscall_to_int[sc] for sc in trace] for trace in normal_traces]
X_attack = [[syscall_to_int[sc] for sc in trace] for trace in attack_traces]

max_length = 500
X_normal_padded = pad_sequences(X_normal, maxlen=max_length, padding='post', truncating='post')
X_attack_padded = pad_sequences(X_attack, maxlen=max_length, padding='post', truncating='post')

y_normal = np.zeros(len(X_normal_padded))
y_attack = np.ones(len(X_attack_padded))

X = np.concatenate((X_normal_padded, X_attack_padded))
y = np.concatenate((y_normal, y_attack))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"Labels in test set: {np.unique(y_test, return_counts=True)}")

class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
print(f"Calculated class weights: {class_weights_dict}")

embedding_dim = 128
lstm_units = 64

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    LSTM(units=lstm_units, dropout=0.5, recurrent_dropout=0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("\n--- Model Architecture and Parameters ---")
model.summary()

print("\nTraining the model with class weights...")
model.fit(
    X_train, 
    y_train, 
    epochs=10, 
    batch_size=32, 
    validation_split=0.1, 
    verbose=1,
    class_weight=class_weights_dict
)

print("\n--- Model Performance Metrics ---")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype("int32")

print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
print(f"Recall: {recall_score(y_test, y_pred, zero_division=0):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred, zero_division=0):.4f}")
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

malicious_indices = np.where(y_test == 1)[0]
selected_malicious_log = None
if len(malicious_indices) > 0:
    correctly_classified_malicious_indices = [i for i in malicious_indices if y_pred[i] == 1]

    if correctly_classified_malicious_indices:
        selected_index = random.choice(correctly_classified_malicious_indices)
        selected_malicious_log = X_test[selected_index]
        print("\n--- Malicious Log Selected for Reinforcement Learning ---")
        print("Selected a random malicious log that was correctly classified.")
        print(f"Pre-processed log (first 100 values): {selected_malicious_log[:100].tolist()}...")
    else:
        print("\nCould not find a malicious log that the model correctly classified. Using a random one.")
        selected_malicious_log = X_test[random.choice(malicious_indices)]
        print(f"Pre-processed log (first 100 values): {selected_malicious_log[:100].tolist()}...")
else:
    print("\nNo malicious logs found in the test set. Cannot proceed with the next component.")

global trained_classifier_model, malicious_log_for_rl
trained_classifier_model = model
malicious_log_for_rl = selected_malicious_log
