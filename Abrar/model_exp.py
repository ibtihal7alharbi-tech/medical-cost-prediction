
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers, callbacks

# Load data
train_data = pd.read_csv('Dataset/train/train_data.csv')
test_data  = pd.read_csv('Dataset/test/test_data.csv')

# Split features and 
# Train
X_train = train_data.drop('charges', axis=1)
y_train = train_data['charges']

# Test
X_test = test_data.drop('charges', axis=1)
y_test = test_data['charges']

# Feature scaling 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Model definition 
def build_model_drop(input_dim: int) -> keras.Model:
    model = keras.Sequential([
        keras.Input(shape=(input_dim,)),
        

        # Hidden layer 1 — 64 units
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
 

        # Hidden layer 2 — 32 units
        layers.Dense(32, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
 
        # Output layer — single neuron (regression)
        layers.Dense(1)
    ])
    return model

def build_model(input_dim: int) -> keras.Model:
    model = keras.Sequential([
        keras.Input(shape=(input_dim,)),
        

        # Hidden layer 1 — 64 units
        layers.Dense(64, activation='relu'),

        # Hidden layer 2 — 32 units
        layers.Dense(32, activation="relu"),
 
        # Output layer — single neuron (regression)
        layers.Dense(1)
    ])
    return model

model = build_model(input_dim=X_train.shape[1])

# Compile 
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss="mae",
    metrics=["mae"],
)

model.summary()

# Callbacks 
early_stop = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1,
)

# Training 
history = model.fit(
    X_train, y_train,
    validation_split=0.1,     # carves out 10% of train set for validation
    epochs=500,
    batch_size=16,
    callbacks=[early_stop],
    verbose=1,
)

# Evaluation 
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest MAE : {test_mae:.4f}")

# Save model 
# model.save("ann_regression.keras")
# print("Model saved to ann_regression.keras")

# Plot training & validation loss 
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(history.history['mae'],     label='Train MAE', linewidth=2)
plt.plot(history.history['val_mae'], label='Val MAE',   linewidth=2, linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.title('Training vs Validation MAE')
plt.legend()
plt.tight_layout()
plt.savefig('loss curves plots/loss_curveOrg.png', dpi=150) #change based on model name
plt.show()
print("Plot saved to loss_curveOrg.png at loss curves plots folder")#change based on model name

# Save losses to CSV 
loss_df = pd.DataFrame({
    'epoch'    : range(1, len(history.history['mae']) + 1),
    'train_mae': history.history['mae'],
    'val_mae'  : history.history['val_mae'],
})
loss_df.to_csv('training_lossesOrg.csv', index=False)#change based on model name
print("Losses saved to training_lossesOrg.csv")#change based on model name
print(loss_df.tail())

# Save predictions vs ground truth
y_pred = model.predict(X_test).flatten()

results_df = pd.DataFrame({
    'ground_truth': y_test.values,
    'predicted'   : y_pred,
    'absolute_error': np.abs(y_test.values - y_pred),
})
results_df.to_csv('saved prid vs truth/predictions_vs_truthOrg.csv', index=False)#change based on model name
print("Predictions saved to predictions_vs_truthOrg.csv at saved prid vs truth folder")#change based on model name
print(results_df.head(10))
