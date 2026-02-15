class NeuralNetwork {
    constructor() {
        this.loaded = false;
        this.W1 = null;
        this.b1 = null;
        this.W2 = null;
        this.b2 = null;
        this.hiddenActivations = new Float32Array(64);
        this.outputActivations = new Float32Array(10);
    }

    loadWeights() {
        const data = PRETRAINED_WEIGHTS;
        this.W1 = data.W1;
        this.b1 = data.b1;
        this.W2 = data.W2;
        this.b2 = data.b2;
        this.loaded = true;
    }

    forward(pixels) {
        if (!this.loaded) return this.outputActivations;

        const hidden = new Float32Array(64);
        for (let j = 0; j < 64; j++) {
            let sum = this.b1[j];
            for (let i = 0; i < 784; i++) {
                sum += pixels[i] * this.W1[i][j];
            }
            hidden[j] = Math.max(0, sum); // ReLU
        }
        this.hiddenActivations = hidden;

        const output = new Float32Array(10);
        let maxVal = -Infinity;
        for (let j = 0; j < 10; j++) {
            let sum = this.b2[j];
            for (let i = 0; i < 64; i++) {
                sum += hidden[i] * this.W2[i][j];
            }
            output[j] = sum;
            if (sum > maxVal) maxVal = sum;
        }

        // Softmax
        let expSum = 0;
        for (let j = 0; j < 10; j++) {
            output[j] = Math.exp(output[j] - maxVal);
            expSum += output[j];
        }
        for (let j = 0; j < 10; j++) {
            output[j] /= expSum;
        }
        this.outputActivations = output;

        return output;
    }

    getPrediction() {
        let maxIdx = 0;
        let maxVal = this.outputActivations[0];
        for (let i = 1; i < 10; i++) {
            if (this.outputActivations[i] > maxVal) {
                maxVal = this.outputActivations[i];
                maxIdx = i;
            }
        }
        return { digit: maxIdx, confidence: maxVal };
    }
}
