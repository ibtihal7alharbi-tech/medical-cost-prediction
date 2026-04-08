import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.metrics import mean_absolute_error
from tqdm.keras import TqdmCallback

#load data
train_data = pd.read_csv('Dataset/train/train_data.csv')
test_data = pd.read_csv('Dataset/test/test_data.csv')

#split features and target
#train
X_train = train_data.drop('charges', axis=1)
y_train = train_data['charges']

#val
X_test = test_data.drop('charges', axis=1)
y_test = test_data['charges']

#baseline model
def build_baseline(input_shape):
    model = tf.keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1) # Linear output for regression
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

baseline_model = build_baseline(X_train.shape[1])
history_baseline = baseline_model.fit(X_train, y_train, validation_split=0.1, epochs=500, batch_size=32, verbose=1)

# Evaluate on test set
test_loss, test_mae = baseline_model.evaluate(X_test, y_test, verbose=0)
print(f"Baseline Test MAE: {test_mae}")

#svae test loss and trainign loss
# train_loss = history_baseline.history['loss']
# val_loss = history_baseline.history['val_loss']
import matplotlib.pyplot as plt
plt.plot(history_baseline.history['loss'], label='Train Loss')
plt.plot(history_baseline.history['val_loss'], label='Validation Loss')

plt.title('Baseline Model - Loss Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()