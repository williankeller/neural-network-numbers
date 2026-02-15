(function () {
    const network = new NeuralNetwork();
    const drawingCanvas = new DrawingCanvas(document.getElementById('drawing-canvas'));
    const visualizer = new NetworkVisualizer(document.getElementById('network-canvas'));

    // UI elements
    const outputGrid = document.getElementById('output-grid');
    const outputCards = outputGrid.querySelectorAll('.output-card');
    const btnClear = document.getElementById('btn-clear');
    const brushSlider = document.getElementById('brush-size');

    // Load weights from global (loaded via <script> tag)
    network.loadWeights();
    visualizer.setWeights(network);

    // Update output cards
    function updateOutputDisplay(activations) {
        let maxIdx = 0;
        let maxVal = 0;
        for (let i = 0; i < 10; i++) {
            if (activations[i] > maxVal) {
                maxVal = activations[i];
                maxIdx = i;
            }
        }

        outputCards.forEach((card) => {
            const digit = parseInt(card.dataset.digit);
            const confidence = activations[digit] || 0;
            const pct = (confidence * 100).toFixed(1);

            card.querySelector('.confidence-fill').style.width = `${confidence * 100}%`;
            card.querySelector('.confidence-value').textContent = `${pct}%`;

            if (digit === maxIdx && maxVal > 0.1) {
                card.classList.add('winner');
            } else {
                card.classList.remove('winner');
            }
        });
    }

    function resetOutputDisplay() {
        outputCards.forEach((card) => {
            card.querySelector('.confidence-fill').style.width = '0%';
            card.querySelector('.confidence-value').textContent = '0%';
            card.classList.remove('winner');
        });
    }

    // Clear button
    btnClear.addEventListener('click', () => {
        drawingCanvas.clear();
        network.hiddenActivations.fill(0);
        network.outputActivations.fill(0);
        visualizer.update(network.hiddenActivations, network.outputActivations);
        resetOutputDisplay();
    });

    // Brush size
    brushSlider.addEventListener('input', (e) => {
        drawingCanvas.setBrushSize(parseFloat(e.target.value));
    });

    // Animation loop
    function animate() {
        if (drawingCanvas.dirty) {
            drawingCanvas.dirty = false;
            drawingCanvas.render();

            if (!drawingCanvas.isEmpty() && network.loaded) {
                const pixels = drawingCanvas.getPixels();
                network.forward(pixels);
                updateOutputDisplay(network.outputActivations);
                visualizer.update(network.hiddenActivations, network.outputActivations);
            }
        }

        visualizer.render();
        requestAnimationFrame(animate);
    }

    animate();
})();
