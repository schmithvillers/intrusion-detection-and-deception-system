Loading ADFA-LD data...
Total samples: 5952 (Normal=5205, Attack=747)
Training Neural Network on ADFA-LD traces...

--- Model Evaluation ---
Accuracy: 0.8724

Classification Report:
               precision    recall  f1-score   support

           0       0.90      0.96      0.93      1042
           1       0.48      0.28      0.35       149

    accuracy                           0.87      1191
   macro avg       0.69      0.62      0.64      1191
weighted avg       0.85      0.87      0.86      1191

Model and scaler saved.

#LSTM Model output

--- Component 1: LSTM Classifier (Updated) ---
Loading data...
Number of normal traces loaded: 5205
Number of attack traces loaded: 746
Total number of unique system calls: 175
Training data shape: (4760, 500)
Test data shape: (1191, 500)
Labels in test set: (array([0., 1.]), array([1042,  149]))
Calculated class weights: {0: np.float64(0.5717030987268796), 1: np.float64(3.986599664991625)}

--- Model Architecture and Parameters ---
/usr/local/lib/python3.11/dist-packages/keras/src/layers/core/embedding.py:97: UserWarning: Argument `input_length` is deprecated. Just remove it.
  warnings.warn(
Model: "sequential_6"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ embedding_6 (Embedding)         │ ?                      │   0 (unbuilt) │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ lstm_6 (LSTM)                   │ ?                      │   0 (unbuilt) │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_12 (Dense)                │ ?                      │   0 (unbuilt) │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_6 (Dropout)             │ ?                      │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_13 (Dense)                │ ?                      │   0 (unbuilt) │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 0 (0.00 B)
 Trainable params: 0 (0.00 B)
 Non-trainable params: 0 (0.00 B)

Training the model with class weights...
Epoch 1/10
134/134 ━━━━━━━━━━━━━━━━━━━━ 91s 646ms/step - accuracy: 0.6344 - loss: 0.6823 - val_accuracy: 0.3109 - val_loss: 0.6618
Epoch 2/10
134/134 ━━━━━━━━━━━━━━━━━━━━ 144s 658ms/step - accuracy: 0.7695 - loss: 0.5883 - val_accuracy: 0.3025 - val_loss: 0.6590
Epoch 3/10
134/134 ━━━━━━━━━━━━━━━━━━━━ 139s 633ms/step - accuracy: 0.6102 - loss: 0.6220 - val_accuracy: 0.8634 - val_loss: 0.5734
Epoch 4/10
134/134 ━━━━━━━━━━━━━━━━━━━━ 143s 641ms/step - accuracy: 0.7615 - loss: 0.6238 - val_accuracy: 0.8529 - val_loss: 0.5992
Epoch 5/10
134/134 ━━━━━━━━━━━━━━━━━━━━ 86s 644ms/step - accuracy: 0.8212 - loss: 0.5924 - val_accuracy: 0.8739 - val_loss: 0.5726
Epoch 6/10
134/134 ━━━━━━━━━━━━━━━━━━━━ 85s 636ms/step - accuracy: 0.8449 - loss: 0.5891 - val_accuracy: 0.8592 - val_loss: 0.6024
Epoch 7/10
134/134 ━━━━━━━━━━━━━━━━━━━━ 86s 639ms/step - accuracy: 0.7759 - loss: 0.5948 - val_accuracy: 0.8676 - val_loss: 0.5963
Epoch 8/10
134/134 ━━━━━━━━━━━━━━━━━━━━ 144s 651ms/step - accuracy: 0.8105 - loss: 0.5712 - val_accuracy: 0.8697 - val_loss: 0.5814
Epoch 9/10
134/134 ━━━━━━━━━━━━━━━━━━━━ 86s 640ms/step - accuracy: 0.7760 - loss: 0.5790 - val_accuracy: 0.8697 - val_loss: 0.5938
Epoch 10/10
134/134 ━━━━━━━━━━━━━━━━━━━━ 87s 649ms/step - accuracy: 0.7242 - loss: 0.5782 - val_accuracy: 0.8697 - val_loss: 0.5803

--- Model Performance Metrics ---
38/38 ━━━━━━━━━━━━━━━━━━━━ 6s 140ms/step
Test Accuracy: 0.8841
Precision: 0.5797
Recall: 0.2685
F1-Score: 0.3670

Confusion Matrix:
[[1013   29]
 [ 109   40]]

--- Malicious Log Selected for Reinforcement Learning ---
Selected a random malicious log that was correctly classified.
Pre-processed log (first 100 values): [149, 86, 149, 28, 2, 86, 149, 2, 86, 86, 2, 2, 149, 86, 149, 86, 2, 2, 149, 86, 86, 28, 2, 149, 149, 28, 28, 149, 2, 149, 28, 2, 149, 28, 86, 149, 28, 105, 175, 86, 149, 28, 2, 149, 28, 149, 86, 149, 28, 2, 149, 28, 86, 86, 2, 86, 28, 149, 2, 86, 2, 86, 86, 2, 149, 2, 86, 149, 28, 86, 2, 86, 2, 86, 28, 2, 2, 86, 149, 2, 149, 86, 2, 2, 2, 149, 149, 149, 2, 86, 2, 86, 149, 28, 2, 149, 28, 149, 149, 2]...



#LightGBM + Kmean model output RUN 1
--- Component 1: Classifier using LightGBM and K-Means ---
Loading data...
Number of normal traces loaded: 5205
Number of attack traces loaded: 746
Total number of unique system calls: 175
Training data shape: (4760, 175)
Test data shape: (1191, 175)
Labels in test set: (array([0., 1.]), array([1042,  149]))
Calculated class weights: {0: np.float64(0.5717030987268796), 1: np.float64(3.986599664991625)}

--- Applying K-Means for feature enrichment ---
Final training data shape with cluster feature: (4760, 176)

--- Training the LightGBM Classifier ---
[LightGBM] [Info] Number of positive: 597, number of negative: 4163
[LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.009985 seconds.
You can set `force_row_wise=true` to remove the overhead.
And if memory is not enough, you can set `force_col_wise=true`.
[LightGBM] [Info] Total Bins 3546
[LightGBM] [Info] Number of data points in the train set: 4760, number of used features: 111
[LightGBM] [Info] [binary:BoostFromScore]: pavg=0.500000 -> initscore=0.000000
[LightGBM] [Info] Start training from score 0.000000

--- Model Performance Metrics ---
Test Accuracy: 0.9631
Precision: 0.8302
Recall: 0.8859
F1-Score: 0.8571

Confusion Matrix:
[[1015   27]
 [  17  132]]

--- Malicious Log Selected for Reinforcement Learning ---
Selected a random malicious log that was correctly classified.
Pre-processed log (first 100 values): [0, 31, 0, 62, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]...
/usr/local/lib/python3.12/dist-packages/sklearn/utils/validation.py:2739: UserWarning: X does not have valid feature names, but LGBMClassifier was fitted with feature names
  warnings.warn(

#LightGBM + Kmean model output RUN 1
--- Component 1: Classifier using LightGBM and K-Means ---
Loading data...
Number of normal traces loaded: 5205
Number of attack traces loaded: 746
Total number of unique system calls: 175
Training data shape: (4760, 175)
Test data shape: (1191, 175)
Labels in test set: (array([0., 1.]), array([1042,  149]))
Calculated class weights: {0: np.float64(0.5717030987268796), 1: np.float64(3.986599664991625)}

--- Applying K-Means for feature enrichment ---
Final training data shape with cluster feature: (4760, 176)

--- Training the LightGBM Classifier ---
[LightGBM] [Info] Number of positive: 597, number of negative: 4163
[LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.004680 seconds.
You can set `force_row_wise=true` to remove the overhead.
And if memory is not enough, you can set `force_col_wise=true`.
[LightGBM] [Info] Total Bins 3549
[LightGBM] [Info] Number of data points in the train set: 4760, number of used features: 111
[LightGBM] [Info] [binary:BoostFromScore]: pavg=0.500000 -> initscore=0.000000
[LightGBM] [Info] Start training from score 0.000000

--- Model Performance Metrics ---

Default Threshold (0.5) Results:
Accuracy:  0.9698
Precision: 0.8844
Recall:    0.8725
F1-Score:  0.8784
Confusion Matrix:
[[1025   17]
 [  19  130]]
/usr/local/lib/python3.12/dist-packages/sklearn/utils/validation.py:2739: UserWarning: X does not have valid feature names, but LGBMClassifier was fitted with feature names
  warnings.warn(
/usr/local/lib/python3.12/dist-packages/sklearn/utils/validation.py:2739: UserWarning: X does not have valid feature names, but LGBMClassifier was fitted with feature names
  warnings.warn(

Best Threshold Tuning Results (threshold=0.64):
Accuracy:  0.9731
Precision: 0.9091
Recall:    0.8725
F1-Score:  0.8904
Confusion Matrix:
[[1029   13]
 [  19  130]]

--- Malicious Log Selected for Reinforcement Learning ---
Pre-processed log (first 100 values): [0, 31, 0, 62, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]...


#LightGBM + Kmean model output RUN 3
--- Component 1: Classifier using LightGBM and K-Means ---
Loading data...
Number of normal traces loaded: 5205
Number of attack traces loaded: 747
Total number of unique system calls/log messages: 178
Training data shape: (4761, 178)
Test data shape: (1191, 178)
Labels in test set: (array([0., 1.]), array([1042,  149]))
Calculated class weights: {0: np.float64(0.5718232044198895), 1: np.float64(3.980769230769231)}

--- Applying K-Means for feature enrichment ---
Final training data shape with cluster feature: (4761, 179)

--- Hyperparameter Tuning with GridSearchCV ---
Fitting 3 folds for each of 1296 candidates, totalling 3888 fits
[LightGBM] [Info] Number of positive: 598, number of negative: 4163
[LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.005467 seconds.
You can set `force_row_wise=true` to remove the overhead.
And if memory is not enough, you can set `force_col_wise=true`.
[LightGBM] [Info] Total Bins 3549
[LightGBM] [Info] Number of data points in the train set: 4761, number of used features: 111
[LightGBM] [Info] [binary:BoostFromScore]: pavg=0.500000 -> initscore=0.000000
[LightGBM] [Info] Start training from score 0.000000

Best Parameters: {'colsample_bytree': 1.0, 'learning_rate': 0.1, 'max_depth': 10, 'n_estimators': 800, 'num_leaves': 128, 'reg_alpha': 0.0, 'reg_lambda': 0.0, 'subsample': 0.8}
Best F1 Score (CV): 0.8798085994810405

--- Model Performance Metrics ---

Default Threshold (0.5) Results:
Accuracy:  0.9715
Precision: 0.9078
Recall:    0.8591
F1-Score:  0.8828
Confusion Matrix:
[[1029   13]
 [  21  128]]
/usr/local/lib/python3.12/dist-packages/sklearn/utils/validation.py:2739: UserWarning: X does not have valid feature names, but LGBMClassifier was fitted with feature names
  warnings.warn(
/usr/local/lib/python3.12/dist-packages/sklearn/utils/validation.py:2739: UserWarning: X does not have valid feature names, but LGBMClassifier was fitted with feature names
  warnings.warn(

Best Threshold Tuning Results (threshold=0.17):
Accuracy:  0.9723
Precision: 0.8766
Recall:    0.9060
F1-Score:  0.8911
Confusion Matrix:
[[1023   19]
 [  14  135]]

--- Malicious Log Selected for Reinforcement Learning ---
Pre-processed log (first 100 values): [0, 36, 0, 87, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]...
