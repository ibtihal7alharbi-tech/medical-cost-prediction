import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import matplotlib.pyplot as plt

# Load data
train_data = pd.read_csv('Dataset/train/train_data.csv')
test_data = pd.read_csv('Dataset/test/test_data.csv')

# Shuffle the training data
# frac=1 means 100% of the data, random_state ensures reproducibility
train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Split features and target
X_train = train_data.drop('charges', axis=1)
y_train = train_data['charges']

X_test = test_data.drop('charges', axis=1)
y_test = test_data['charges']

# Model builder
def build_tuned_model(units, layers_count, lr):
    model = keras.Sequential()
    model.add(layers.Dense(units, activation='relu', input_shape=(X_train.shape[1],)))
    
    for _ in range(layers_count - 1):
        model.add(layers.Dense(units, activation='relu'))
        
    model.add(layers.Dense(1))
    
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

# Hyperparameter space
lrs = [0.01, 0.001]
batches = [16, 32]
units_list = [32, 64]
layers_list = [1, 2, 3]

best_mae = float('inf')
best_params = {}
best_model = None
best_history = None

# Tuning loop
for lr in lrs:
    for batch in batches:
        for units in units_list:
            for num_layers in layers_list:
                print(f"Testing: LR={lr}, Batch={batch}, Units={units}, Layers={num_layers}")
                
                model = build_tuned_model(units, num_layers, lr)

                early_stop = keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )

                history = model.fit(
                    X_train, y_train,
                    validation_split=0.1,
                    epochs=500,
                    batch_size=batch,
                    callbacks=[early_stop],
                    verbose=1
                )

                val_mae = min(history.history['val_mae'])

                if val_mae < best_mae:
                    best_mae = val_mae
                    best_params = {
                        'lr': lr,
                        'batch': batch,
                        'units': units,
                        'layers': num_layers
                    }
                    best_model = model          #  save trained model
                    best_history = history      #  save correct history

print("\n--- Tuning Complete ---")
print(f"Best Val MAE: {best_mae:.2f}")
print(f"Best Configuration: {best_params}")

best_model.summary()

# Evaluate on test set
test_loss, test_mae = best_model.evaluate(X_test, y_test, verbose=0)
print(f"Best Model Test MAE: {test_mae}")

# Plot loss curves (BEST run)
plt.plot(best_history.history['loss'], label='Train Loss')
plt.plot(best_history.history['val_loss'], label='Validation Loss')
plt.title('Best Tuned Model - Loss Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()