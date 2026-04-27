import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

SEED = 42
EPOCHS = 10
BATCH_SIZE = 32
FIGURES_DIR = "figures"
RESULTS_DIR = "results"

np.random.seed(SEED)
tf.random.set_seed(SEED)

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

def directories() -> None:
    """Make output directories if they don't already exist."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

def load_and_prepare_data():
    """Loads CIFAR10 dataset and prepares it for training."""
    (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.cifar10.load_data()

    y_train_full = y_train_full.flatten()
    y_test = y_test.flatten()

    x_train_full = x_train_full.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    x_valid = x_train_full[:5000]
    y_valid = y_train_full[:5000]
    x_train = x_train_full[5000:]
    y_train = y_train_full[5000:]

    return x_train, y_train, x_valid, y_valid, x_test, y_test

def plot_sample_images(x_train: np.ndarray, y_train: np.ndarray) -> None:
    """Plot small sample of training images and saves figure."""
    plt.figure(figsize=(10, 5))

    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x_train[i])
        plt.title(str(CLASS_NAMES[y_train[i]]))
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "sample_images.png"), dpi=300)
    plt.close()

def build_baseline_model() -> keras.Model:
    """Build baseline model for CIFAR10 dataset."""
    model = keras.Sequential([
        keras.layers.Input(shape=(32, 32, 3)),
        keras.layers.Flatten(),
        keras.layers.Dense(300, activation="relu"),
        keras.layers.Dense(100, activation="relu"),
        keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    return model

def build_cnn_model() -> keras.Model:
    """Build a simple CNN model with Batch Normalization and Dropout."""
    model = keras.Sequential([
        keras.layers.Input(shape=(32, 32, 3)),
        
        keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        keras.layers.BatchNormalization(), # Added for stability
        keras.layers.MaxPooling2D(pool_size=2),
        
        keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        keras.layers.BatchNormalization(), # Added for stability
        keras.layers.MaxPooling2D(pool_size=2),
        
        keras.layers.Flatten(),
        keras.layers.Dense(100, activation="relu"),
        keras.layers.Dropout(0.3), # Added to prevent overfitting
        keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    return model

def train_model(model: keras.Model, x_train: np.ndarray, y_train: np.ndarray,
                x_valid: np.ndarray, y_valid: np.ndarray, model_name: str):
    """Train model with a Learning Rate Scheduler and save learning curve plot."""
    print(f"\nTraining {model_name}...")
    
    # Callback to reduce learning rate when progress plateaus
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=2, 
        verbose=1
    )

    history = model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_valid, y_valid),
        callbacks=[lr_scheduler], # Integrated the new callback
        verbose=1
    )

    plot_learning_curves(history, model_name)

    return history

def plot_learning_curves(history, model_name: str):
    """Saves accuracy and loss curves for trained models."""
    history_dict = history.history
    epochs_range = range(1, len(history.history["loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history_dict["accuracy"], label="Training accuracy")
    plt.plot(epochs_range, history_dict["val_accuracy"], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name} Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"{model_name.lower()}_accuracy.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history_dict["loss"], label="Training loss")
    plt.plot(epochs_range, history_dict["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"{model_name.lower()}_loss.png"), dpi=300)
    plt.close()

def plot_predictions(model: keras.Model, x_test: np.ndarray, y_test: np.ndarray, model_name: str) -> None:
    """Save example predictions from trained model."""
    predictions = model.predict(x_test[:10], verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)

    plt.figure(figsize=(12, 5))

    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x_test[i])

        pred_name = CLASS_NAMES[predicted_labels[i]]
        true_name = CLASS_NAMES[y_test[i]]

        plt.title(f"Prediction: {pred_name}\nTrue: {true_name}", fontsize=9)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f"{model_name.lower()}_predictions.png"), dpi=300)
    plt.close()

def evaluate_model(model: keras.Model, x_test: np.ndarray, y_test: np.ndarray, model_name: str):
    """Evaluate trained model and return its test metrics."""
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

    print(f"{model_name} Test Loss: {loss:.4f}")
    print(f"{model_name} Test Accuracy: {accuracy:.4f}")

    return {"model": model_name, "test_loss": loss, "test_accuracy": accuracy}

def save_results(results: list[dict]) -> None:
    """Save results to a text file."""
    output_path = os.path.join(RESULTS_DIR, "model_results.txt")
    with open(output_path, "w", encoding="utf-8") as file:
        file.write("CIFAR-10 Model Comparison Results\n")

        for result in results:
            file.write(f"Model: {result['model']}\n")
            file.write(f"Test Loss: {result['test_loss']:.4f}\n")
            file.write(f"Test Accuracy: {result['test_accuracy']:.4f}\n\n")

def main() -> None:
    directories()

    print("Loading CIFAR-10 dataset...")
    x_train, y_train, x_valid, y_valid, x_test, y_test = load_and_prepare_data()

    print("Training set shape:", x_train.shape)
    print("Validation set shape:", x_valid.shape)
    print("Test set shape:", x_test.shape)

    plot_sample_images(x_train, y_train)

    baseline_model = build_baseline_model()
    train_model(baseline_model, x_train, y_train, x_valid, y_valid, "Baseline")
    baseline_results = evaluate_model(baseline_model, x_test, y_test, "Baseline")
    plot_predictions(baseline_model, x_test, y_test, "Baseline")

    cnn_model = build_cnn_model()
    train_model(cnn_model, x_train, y_train, x_valid, y_valid, "CNN")
    cnn_results = evaluate_model(cnn_model, x_test, y_test, "CNN")
    plot_predictions(cnn_model, x_test, y_test, "CNN")

    save_results([baseline_results, cnn_results])

if __name__ == "__main__":
    main()
