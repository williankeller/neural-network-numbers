class NetworkVisualizer {
    constructor(canvasElement) {
        this.canvas = canvasElement;
        this.ctx = canvasElement.getContext('2d');
        this.hiddenActivations = new Float32Array(64);
        this.outputActivations = new Float32Array(10);
        this.weights = null;
        this.topConnections = [];
    }

    setWeights(network) {
        this.weights = network;
        this._computeTopConnections();
    }

    _computeTopConnections() {
        if (!this.weights || !this.weights.W2) return;

        // Pre-compute top connections by magnitude for hidden->output
        const connections = [];
        for (let i = 0; i < 64; i++) {
            for (let j = 0; j < 10; j++) {
                connections.push({
                    from: i,
                    to: j,
                    weight: this.weights.W2[i][j],
                    magnitude: Math.abs(this.weights.W2[i][j]),
                });
            }
        }
        connections.sort((a, b) => b.magnitude - a.magnitude);
        // Keep top 120 connections (roughly 2 per neuron pair on average)
        this.topConnections = connections.slice(0, 120);
    }

    update(hiddenActivations, outputActivations) {
        this.hiddenActivations = hiddenActivations;
        this.outputActivations = outputActivations;
    }

    render() {
        const canvas = this.canvas;
        const ctx = this.ctx;
        const dpr = window.devicePixelRatio || 1;
        const rect = canvas.getBoundingClientRect();

        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
        ctx.scale(dpr, dpr);

        const w = rect.width;
        const h = rect.height;

        ctx.fillStyle = '#0a0a0f';
        ctx.fillRect(0, 0, w, h);

        const outputY = 40;
        const hiddenY = h - 40;
        const padding = 30;

        // Draw connections first (behind neurons)
        this._drawConnections(ctx, w, h, outputY, hiddenY, padding);

        // Draw hidden layer
        this._drawHiddenLayer(ctx, w, hiddenY, padding);

        // Draw output layer
        this._drawOutputLayer(ctx, w, outputY, padding);

        // Labels
        ctx.fillStyle = '#555566';
        ctx.font = '10px "Space Grotesk", sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('OUTPUT (10 neurons, Softmax)', padding, outputY - 18);
        ctx.fillText('HIDDEN (64 neurons, ReLU)', padding, hiddenY - 18);
    }

    _drawConnections(ctx, w, h, outputY, hiddenY, padding) {
        if (!this.topConnections.length) return;

        const hiddenSpacing = (w - padding * 2) / 63;
        const outputSpacing = (w - padding * 2) / 9;

        ctx.lineWidth = 1.5;
        ctx.setLineDash([5, 3]);

        for (const conn of this.topConnections) {
            const fromX = padding + conn.from * hiddenSpacing;
            const fromY = hiddenY;
            const toX = padding + conn.to * outputSpacing;
            const toY = outputY + 20;

            const activation = this.hiddenActivations[conn.from] || 0;
            const strength = Math.min(activation * conn.magnitude * 2, 1);
            const baseAlpha = 0.05 + strength * 0.4;

            if (conn.weight > 0) {
                ctx.strokeStyle = `rgba(0, 255, 136, ${baseAlpha})`;
            } else {
                ctx.strokeStyle = `rgba(255, 0, 170, ${baseAlpha})`;
            }

            ctx.beginPath();
            ctx.moveTo(fromX, fromY);
            // Curved connection
            const midY = (fromY + toY) / 2;
            ctx.quadraticCurveTo(fromX, midY, toX, toY);
            ctx.stroke();
        }

        ctx.setLineDash([]);
    }

    _drawHiddenLayer(ctx, w, y, padding) {
        const spacing = (w - padding * 2) / 63;
        const size = Math.min(spacing * 0.7, 8);

        for (let i = 0; i < 64; i++) {
            const x = padding + i * spacing;
            const activation = this.hiddenActivations[i] || 0;
            const normalizedAct = Math.min(activation / 3, 1); // Normalize

            // Interpolate color: inactive gray -> active cyan
            const r = Math.floor(51 + (0 - 51) * normalizedAct);
            const g = Math.floor(51 + (245 - 51) * normalizedAct);
            const b = Math.floor(68 + (255 - 68) * normalizedAct);

            // Glow
            if (normalizedAct > 0.1) {
                ctx.shadowColor = `rgba(0, 245, 255, ${normalizedAct * 0.5})`;
                ctx.shadowBlur = normalizedAct * 10;
            }

            ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
            ctx.fillRect(x - size / 2, y - size / 2, size, size);

            ctx.shadowColor = 'transparent';
            ctx.shadowBlur = 0;

            // Border
            ctx.strokeStyle = `rgba(0, 245, 255, ${0.2 + normalizedAct * 0.6})`;
            ctx.lineWidth = 1;
            ctx.strokeRect(x - size / 2, y - size / 2, size, size);
        }
    }

    _drawOutputLayer(ctx, w, y, padding) {
        const spacing = (w - padding * 2) / 9;
        const radius = 16;

        let maxIdx = 0;
        let maxVal = 0;
        for (let i = 0; i < 10; i++) {
            if (this.outputActivations[i] > maxVal) {
                maxVal = this.outputActivations[i];
                maxIdx = i;
            }
        }

        for (let i = 0; i < 10; i++) {
            const x = padding + i * spacing;
            const activation = this.outputActivations[i] || 0;
            const isWinner = (i === maxIdx && maxVal > 0.1);

            // Circle
            ctx.beginPath();
            ctx.arc(x, y, radius, 0, Math.PI * 2);

            if (isWinner) {
                ctx.fillStyle = `rgba(0, 245, 255, ${0.15 + activation * 0.3})`;
                ctx.shadowColor = 'rgba(0, 245, 255, 0.5)';
                ctx.shadowBlur = 15;
            } else {
                ctx.fillStyle = `rgba(51, 51, 68, ${0.5 + activation * 0.5})`;
            }
            ctx.fill();

            ctx.shadowColor = 'transparent';
            ctx.shadowBlur = 0;

            ctx.strokeStyle = isWinner ? '#00f5ff' : `rgba(100, 100, 150, ${0.3 + activation * 0.5})`;
            ctx.lineWidth = isWinner ? 2 : 1;
            ctx.stroke();

            // Digit label
            ctx.fillStyle = isWinner ? '#00f5ff' : '#8888aa';
            ctx.font = `${isWinner ? '600' : '400'} 13px "JetBrains Mono", monospace`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(String(i), x, y);

            // Confidence arc
            if (activation > 0.01) {
                ctx.beginPath();
                ctx.arc(x, y, radius + 3, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * activation);
                ctx.strokeStyle = isWinner
                    ? `rgba(0, 245, 255, ${0.6 + activation * 0.4})`
                    : `rgba(0, 255, 136, ${activation * 0.6})`;
                ctx.lineWidth = 2;
                ctx.stroke();
            }
        }
    }
}
