#!/usr/bin/env python3
"""
Train a simple 784 -> 64 -> 10 neural network on MNIST
and export the weights as a JSON file for the browser visualization.
"""
import numpy as np
import gzip
import struct
import json
import os
import urllib.request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MNIST_DIR = os.path.join(SCRIPT_DIR, 'mnist_data')
WEIGHTS_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'weights.json')

os.makedirs(MNIST_DIR, exist_ok=True)
os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)

URLS = {
    'train_images': 'https://github.com/golbin/TensorFlow-MNIST/raw/master/mnist/data/train-images-idx3-ubyte.gz',
    'train_labels': 'https://github.com/golbin/TensorFlow-MNIST/raw/master/mnist/data/train-labels-idx1-ubyte.gz',
    'test_images': 'https://github.com/golbin/TensorFlow-MNIST/raw/master/mnist/data/t10k-images-idx3-ubyte.gz',
    'test_labels': 'https://github.com/golbin/TensorFlow-MNIST/raw/master/mnist/data/t10k-labels-idx1-ubyte.gz',
}


def download_mnist():
    for name, url in URLS.items():
        path = os.path.join(MNIST_DIR, f'{name}.gz')
        if not os.path.exists(path):
            print(f'Downloading {name}...')
            try:
                urllib.request.urlretrieve(url, path)
            except Exception as e:
                print(f'Failed to download {name}: {e}')
                return False
    return True


def load_images(path):
    with gzip.open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num, rows * cols).astype(np.float32) / 255.0


def load_labels(path):
    with gzip.open(path, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8)


class SimpleNN:
    def __init__(self, input_size=784, hidden_size=64, output_size=10):
        self.hidden_size = hidden_size
        self.W1 = np.random.randn(input_size, hidden_size).astype(np.float32) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size, dtype=np.float32)
        self.W2 = np.random.randn(hidden_size, output_size).astype(np.float32) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(output_size, dtype=np.float32)

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        e = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2

    def backward(self, X, y_onehot, lr=0.01):
        m = X.shape[0]
        dz2 = self.a2 - y_onehot
        dW2 = (self.a1.T @ dz2) / m
        db2 = dz2.mean(axis=0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (self.z1 > 0).astype(np.float32)
        dW1 = (X.T @ dz1) / m
        db1 = dz1.mean(axis=0)
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

    def train(self, X_train, y_train, X_test, y_test, epochs=20, batch_size=128, lr=0.1):
        m = X_train.shape[0]
        y_onehot = np.eye(10)[y_train]
        for epoch in range(epochs):
            idx = np.random.permutation(m)
            X_shuffled = X_train[idx]
            y_shuffled = y_onehot[idx]
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                self.forward(X_batch)
                self.backward(X_batch, y_batch, lr)
            lr *= 0.95
            preds = self.forward(X_test).argmax(axis=1)
            acc = (preds == y_test).mean()
            print(f'Epoch {epoch + 1}/{epochs} - Test accuracy: {acc:.4f}')
        return acc

    def export_weights(self, path):
        def to_list(arr, decimals=4):
            return np.round(arr, decimals).tolist()

        weights = {
            'W1': to_list(self.W1),
            'b1': to_list(self.b1),
            'W2': to_list(self.W2),
            'b2': to_list(self.b2),
            'architecture': [784, 64, 10],
        }
        with open(path, 'w') as f:
            json.dump(weights, f)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f'Weights saved to {path} ({size_mb:.2f} MB)')


def generate_digit_templates():
    templates = {}

    def make_grid():
        return np.zeros((28, 28), dtype=np.float32)

    g = make_grid()
    for y in range(6, 22):
        for x in range(8, 20):
            dx = (x - 14) / 6
            dy = (y - 14) / 8
            if 0.5 < dx * dx + dy * dy < 1.2:
                g[y, x] = 1.0
    templates[0] = g

    g = make_grid()
    for y in range(5, 23):
        g[y, 13] = 1.0
        g[y, 14] = 1.0
    g[22, 11:17] = 1.0
    templates[1] = g

    g = make_grid()
    g[6, 9:18] = 1.0
    g[7, 18] = 1.0; g[8, 18] = 1.0; g[9, 18] = 1.0
    g[10, 17] = 1.0; g[11, 16] = 1.0; g[12, 15] = 1.0
    g[13, 14] = 1.0; g[14, 13] = 1.0; g[15, 12] = 1.0
    g[16, 11] = 1.0; g[17, 10] = 1.0
    g[18, 9:19] = 1.0
    templates[2] = g

    g = make_grid()
    g[6, 9:17] = 1.0
    for y in range(7, 12): g[y, 17] = 1.0
    g[12, 11:17] = 1.0
    for y in range(13, 19): g[y, 17] = 1.0
    g[19, 9:17] = 1.0
    templates[3] = g

    g = make_grid()
    for y in range(5, 14): g[y, 9] = 1.0
    g[14, 9:19] = 1.0
    for y in range(5, 22): g[y, 16] = 1.0
    templates[4] = g

    g = make_grid()
    g[6, 9:19] = 1.0
    for y in range(7, 12): g[y, 9] = 1.0
    g[12, 9:17] = 1.0
    for y in range(13, 19): g[y, 17] = 1.0
    g[19, 9:17] = 1.0
    templates[5] = g

    g = make_grid()
    g[6, 10:18] = 1.0
    for y in range(7, 19): g[y, 9] = 1.0
    g[12, 10:17] = 1.0
    for y in range(13, 19): g[y, 17] = 1.0
    g[19, 10:17] = 1.0
    templates[6] = g

    g = make_grid()
    g[6, 9:19] = 1.0
    for y in range(7, 22):
        x = 18 - int((y - 7) * 0.5)
        if 8 <= x <= 19: g[y, x] = 1.0
    templates[7] = g

    g = make_grid()
    g[6, 10:17] = 1.0; g[12, 10:17] = 1.0; g[19, 10:17] = 1.0
    for y in range(7, 12): g[y, 9] = 1.0; g[y, 17] = 1.0
    for y in range(13, 19): g[y, 9] = 1.0; g[y, 17] = 1.0
    templates[8] = g

    g = make_grid()
    g[6, 10:17] = 1.0; g[12, 10:17] = 1.0
    for y in range(7, 12): g[y, 9] = 1.0; g[y, 17] = 1.0
    for y in range(13, 20): g[y, 17] = 1.0
    g[20, 10:17] = 1.0
    templates[9] = g

    return templates


def augment_template(template, n=500):
    from scipy.ndimage import gaussian_filter
    samples = []
    for _ in range(n):
        g = template.copy()
        dx, dy = np.random.randint(-3, 4), np.random.randint(-3, 4)
        g = np.roll(np.roll(g, dx, axis=1), dy, axis=0)
        g += np.random.randn(28, 28).astype(np.float32) * 0.1
        g *= np.random.uniform(0.8, 1.2)
        g = gaussian_filter(g, sigma=np.random.uniform(0.5, 1.5))
        g = np.clip(g, 0, 1)
        samples.append(g.flatten())
    return np.array(samples)


if __name__ == '__main__':
    np.random.seed(42)

    success = download_mnist()

    if success:
        print("Loading MNIST data...")
        X_train = load_images(os.path.join(MNIST_DIR, 'train_images.gz'))
        y_train = load_labels(os.path.join(MNIST_DIR, 'train_labels.gz'))
        X_test = load_images(os.path.join(MNIST_DIR, 'test_images.gz'))
        y_test = load_labels(os.path.join(MNIST_DIR, 'test_labels.gz'))
        print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    else:
        print("MNIST download failed. Generating synthetic training data...")
        from scipy.ndimage import gaussian_filter
        templates = generate_digit_templates()
        X_list, y_list = [], []
        for digit, tmpl in templates.items():
            augmented = augment_template(tmpl, n=1000)
            X_list.append(augmented)
            y_list.append(np.full(len(augmented), digit, dtype=np.uint8))
        X_train = np.vstack(X_list)
        y_train = np.concatenate(y_list)
        idx = np.random.permutation(len(X_train))
        X_train = X_train[idx]
        y_train = y_train[idx]
        split = int(len(X_train) * 0.9)
        X_test = X_train[split:]
        y_test = y_train[split:]
        X_train = X_train[:split]
        y_train = y_train[:split]
        print(f"Synthetic training set: {X_train.shape}, Test set: {X_test.shape}")

    nn = SimpleNN(784, 64, 10)
    acc = nn.train(X_train, y_train, X_test, y_test, epochs=20, batch_size=128, lr=0.1)
    print(f"\nFinal test accuracy: {acc:.4f}")

    nn.export_weights(WEIGHTS_PATH)
    print("Done!")
