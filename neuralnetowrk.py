import os
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Your dataset paths ---
base_dir = '/content/ADFA-LD_unzipped/ADFA-LD'
normal_train_dir = os.path.join(base_dir, 'Training_Data_Master')
normal_val_dir = os.path.join(base_dir, 'Validation_Data_Master')
attack_dir = os.path.join(base_dir, 'Attack_Data_Master')

# --------------------------
# Loading ADFA-LD traces
# --------------------------
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

# --------------------------
# Load ART log (custom attack)
# --------------------------
def load_art_log(file_path, string_to_int_map, next_id):
    traces = []
    if not os.path.exists(file_path):
        print(f"Warning: Log file not found: {file_path}")
        return traces, next_id

    with open(file_path, 'r', errors='ignore') as f:
        content = f.read()

    # clean RTF artifacts
    content = re.sub(r'\{\\\*?\\[^{}]*\}', '', content)
    content = content.replace('\\par', '\n').replace('\\b', '').replace('\\', '')

    lines = content.split('\n')
    current_trace = []
    for line in lines:
        line = line.strip()
        if line.startswith('Error reading') or line.startswith('[/var/log/vmware'):
            if 'Permission denied' in line:
                key = 'Permission denied'
            elif 'Couldn\'t get VMCI socket family info' in line:
                key = 'Couldn\'t get VMCI socket family info'
            elif 'EventToCore: Not implemented yet' in line:
                key = 'EventToCore: Not implemented yet'
            elif 'Failed to send RPCI message' in line:
                key = 'Failed to send RPCI message'
            else:
                continue

            if key not in string_to_int_map:
                string_to_int_map[key] = next_id
                next_id += 1

            current_trace.append(string_to_int_map[key])

    if current_trace:
        traces.append(current_trace)

    return traces, next_id


# --------------------------
# Prepare Dataset
# --------------------------
print("Loading ADFA-LD data...")
normal_train_traces = load_traces(normal_train_dir)
normal_val_traces = load_traces(normal_val_dir)
attack_traces = load_traces(attack_dir)

# Load ART logs as attack traces
string_to_int_map = {}
next_id_counter = 5000
art_log_path = 'art.log.rtf'
art_log_traces, next_id_counter = load_art_log(art_log_path, string_to_int_map, next_id_counter)
attack_traces.extend(art_log_traces)

# Combine dataset
X = normal_train_traces + normal_val_traces + attack_traces
y = [0] * (len(normal_train_traces) + len(normal_val_traces)) + [1] * len(attack_traces)

print(f"Total samples: {len(X)} (Normal={y.count(0)}, Attack={y.count(1)})")

# Pad sequences to fixed length
MAX_LEN = 200  # adjust based on ADFA trace lengths
X_padded = pad_sequences(X, maxlen=MAX_LEN, padding='post', truncating='post')

# Scale features (important for MLP)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_padded)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# --------------------------
# Train Neural Network
# --------------------------
print("Training Neural Network on ADFA-LD traces...")
nn_model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, activation='relu', solver='adam', random_state=42)

nn_model.fit(X_train, y_train)

# --------------------------
# Evaluation
# --------------------------
y_pred = nn_model.predict(X_test)
print("\n--- Model Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model + scaler
joblib.dump(nn_model, "./adfa_nn_model.pkl")
joblib.dump(scaler, "./adfa_scaler.pkl")
print("Model and scaler saved.")
