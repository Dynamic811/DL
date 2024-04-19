import numpy as np
from sklearn.model_selection import KFold
from tensorflow import keras
from tensorflow.keras import layers

# Generate sample data
num_samples = 1000
num_features = 20
num_classes = 5

X = np.random.randn(num_samples, num_features)
y = np.random.randint(0, num_classes, size=num_samples)

# Define the architecture of the neural network
def create_model():
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(num_features,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Perform k-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True)
fold_accuracies = []

for train_idx, test_idx in kfold.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = create_model()
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    fold_accuracies.append(accuracy)

# Calculate and print average accuracy
average_accuracy = np.mean(fold_accuracies)
print(f'Average accuracy: {average_accuracy:.4f}')
