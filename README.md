# How AI Sees Numbers

A real-time neural network visualizer that lets you draw digits and watch an AI classify them — seeing every neuron fire, every connection light up, and every prediction form.

This project was created as a study exercise to understand how neural networks work at a fundamental level — specifically how inputs, biases, and outputs interact to produce decisions, much like the building blocks behind large language models and other AI systems.

Built entirely with vanilla JavaScript, HTML, and CSS. No frameworks, no build tools, no server required. Just open `index.html` and start drawing.

![Architecture: 784 → 64 → 10](https://img.shields.io/badge/architecture-784→64→10-00f5ff?style=flat-square)
![Accuracy: ~97%](https://img.shields.io/badge/accuracy-~97%25-00ff88?style=flat-square)
![No Dependencies](https://img.shields.io/badge/dependencies-none-ff00aa?style=flat-square)

## What This Is

Most neural network demos show you a black box: input goes in, answer comes out. This project cracks it open.

When you draw a digit, you can see:

- **64 hidden neurons** lighting up based on what features they detect (edges, curves, strokes)
- **Weighted connections** between layers — green for positive weights, magenta for negative — pulsing with activation strength
- **10 output neurons** competing via softmax, with confidence arcs showing how certain the network is
- **Real-time predictions** updating as you draw each stroke

The network is a simple feedforward architecture (784 inputs → 64 hidden with ReLU → 10 outputs with Softmax) trained on MNIST handwritten digits. It runs entirely in the browser — the forward pass is pure JavaScript, no libraries.

## Demo

Open `index.html` in any browser. Draw on the 28×28 grid on the left. Watch the network think on the right.

Hosted version: *[Add your GitHub Pages URL here]*

## Project Structure

```
neural-network-numbers/
├── index.html              # Entry point
├── style.css               # Dark theme UI
├── js/
│   ├── main.js             # App init, event wiring, animation loop
│   ├── network.js          # NeuralNetwork class — forward pass in JS
│   ├── canvas.js           # 28×28 drawing grid with mouse/touch input
│   └── visualizer.js       # Network graph rendering (neurons + connections)
├── data/
│   ├── weights.js          # Pre-trained weights (loaded via <script> tag)
│   └── weights.json        # Same weights in JSON format
└── train/
    └── train_model.py      # NumPy-only MNIST trainer
```

## How It Works

### The Drawing Canvas

You draw on a 28×28 pixel grid — the same resolution as MNIST training images. The canvas uses interpolated brush strokes for smooth lines and applies a light Gaussian blur to match the style of the original MNIST handwriting samples.

### The Forward Pass

When you draw, the 784 pixel values feed into the network:

1. **Input layer (784 neurons):** Each pixel is a value from 0.0 (black) to 1.0 (white)
2. **Hidden layer (64 neurons):** `ReLU(input × W1 + b1)` — detects low-level features
3. **Output layer (10 neurons):** `Softmax(hidden × W2 + b2)` — produces probability for each digit 0–9

The entire forward pass runs in ~1ms in the browser. No WebGL, no WASM, just loops and math.

### The Visualization

- **Hidden neurons** are small squares whose brightness maps to activation level
- **Output neurons** are circles with confidence arcs that fill proportionally to probability
- **Connections** are drawn for the top 120 strongest weights, colored by sign (green = positive, magenta = negative) with opacity driven by activation strength

## How to Train

The included training script uses only NumPy — no PyTorch, no TensorFlow.

### Requirements

```bash
pip install numpy
# scipy is optional (only needed if MNIST download fails and synthetic data fallback is used)
```

### Train

```bash
cd train
python train_model.py
```

This will:

1. Download the MNIST dataset (~11MB)
2. Train for 20 epochs with mini-batch SGD (learning rate 0.1, batch size 128, 0.95 decay)
3. Print test accuracy each epoch (expect ~97%)
4. Export weights to `data/weights.json`

After training, convert to the JS format used by the browser:

```bash
cd ..
echo "const PRETRAINED_WEIGHTS = $(cat data/weights.json);" > data/weights.js
```

### Customizing Training

Edit `train/train_model.py` to experiment:

- Change `hidden_size` (default 64) for a wider or narrower network
- Adjust `epochs`, `batch_size`, or `lr` for different training dynamics
- The architecture must remain a single hidden layer for the visualizer to render correctly

## Running Locally

No server needed. Just open:

```
index.html
```

Weights are embedded as a `<script>` tag, so `file://` protocol works fine.

If you prefer a local server (e.g., for development):

```bash
python3 -m http.server
# Open http://localhost:8000
```

## Deploying to GitHub Pages

Push to a GitHub repository and enable Pages from Settings → Pages → Source: main branch. Everything is static — no build step required.

## Tech Stack

- **Frontend:** Vanilla JavaScript, HTML5 Canvas, CSS3
- **Training:** Python 3 + NumPy
- **Fonts:** Space Grotesk, JetBrains Mono (Google Fonts)
- **Design:** Dark theme with cyan/magenta accents

## License

MIT
