/**
 * SessionTracker - Manages scanning session state
 */
class SessionTracker {
    constructor() {
        this.detectedCards = new Set();
        this.cards = [];
        this.runningTotal = 0;
        this.load();
    }

    addCard(cardData) {
        const cardId = cardData.card_id;
        if (!cardId || this.detectedCards.has(cardId)) {
            return false; // Duplicate or invalid
        }

        this.detectedCards.add(cardId);
        this.cards.push({
            id: cardId,
            name: cardData.name,
            price: cardData.price || 0,
            timestamp: new Date().toISOString()
        });
        this.runningTotal += cardData.price || 0;
        this.save();
        this.updateDisplay();
        return true; // New card added
    }

    clear() {
        this.detectedCards.clear();
        this.cards = [];
        this.runningTotal = 0;
        this.save();
        this.updateDisplay();
    }

    save() {
        try {
            localStorage.setItem('pricelens_session', JSON.stringify({
                cards: this.cards,
                total: this.runningTotal,
                count: this.cards.length
            }));
        } catch (error) {
            console.error('Failed to save session:', error);
        }
    }

    load() {
        try {
            const saved = localStorage.getItem('pricelens_session');
            if (saved) {
                const data = JSON.parse(saved);
                this.cards = data.cards || [];
                this.runningTotal = data.total || 0;
                this.detectedCards = new Set(this.cards.map(c => c.id));
                this.updateDisplay();
            }
        } catch (error) {
            console.error('Failed to load session:', error);
        }
    }

    updateDisplay() {
        const totalElement = document.getElementById('total-value');
        const countElement = document.getElementById('card-count');

        if (totalElement) {
            totalElement.textContent = `$${this.runningTotal.toFixed(2)}`;
            // Add pulse animation
            totalElement.classList.add('updating');
            setTimeout(() => totalElement.classList.remove('updating'), 300);
        }

        if (countElement) {
            const cardText = this.cards.length === 1 ? 'card' : 'cards';
            countElement.textContent = `📦 ${this.cards.length} ${cardText}`;
        }
    }
}

// Global session tracker instance
let sessionTracker = null;

/**
 * CameraManager - Handles live camera scanning
 */
class CameraManager {
    constructor() {
        this.stream = null;
        this.video = document.getElementById('camera-feed');
        this.canvas = document.getElementById('overlay-canvas');
        this.ctx = this.canvas ? this.canvas.getContext('2d') : null;
        this.isDetecting = false;
        this.detectionInterval = null;
        this.detectionInProgress = false;
        this.currentDetections = [];  // Track current detections
    }

    async startCamera() {
        try {
            const constraints = {
                video: {
                    facingMode: 'environment', // Rear camera on mobile
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                }
            };

            this.stream = await navigator.mediaDevices.getUserMedia(constraints);
            this.video.srcObject = this.stream;

            // Match canvas size to video when metadata loads
            this.video.addEventListener('loadedmetadata', () => {
                this.canvas.width = this.video.videoWidth;
                this.canvas.height = this.video.videoHeight;
                console.log(`Camera started: ${this.video.videoWidth}x${this.video.videoHeight}`);
            });

            return true;
        } catch (error) {
            console.error('Error starting camera:', error);
            alert('Could not access camera. Please grant camera permissions.');
            return false;
        }
    }

    stopCamera() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
            this.video.srcObject = null;
        }
        this.stopDetection();

        // Clear all detection overlays and reset state
        if (this.ctx) {
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        }
        this.currentDetections = [];  // Reset detection state
    }

    startDetection() {
        if (this.isDetecting) return;

        this.isDetecting = true;
        console.log('Starting live detection...');

        // Detect every 500ms (2 FPS)
        this.detectionInterval = setInterval(() => {
            if (!this.detectionInProgress) {
                this.detectFrame();
            }
        }, 500);
    }

    stopDetection() {
        this.isDetecting = false;
        if (this.detectionInterval) {
            clearInterval(this.detectionInterval);
            this.detectionInterval = null;
        }
        // Clear canvas and reset detection state
        if (this.ctx) {
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        }
        this.currentDetections = [];  // Reset detection state
    }

    async detectFrame() {
        if (!this.video || this.video.readyState !== 4) return;

        this.detectionInProgress = true;

        try {
            // Capture current frame
            const frameCanvas = document.createElement('canvas');
            frameCanvas.width = this.video.videoWidth;
            frameCanvas.height = this.video.videoHeight;
            const frameCtx = frameCanvas.getContext('2d');
            frameCtx.drawImage(this.video, 0, 0);

            // Convert to blob
            const blob = await new Promise(resolve =>
                frameCanvas.toBlob(resolve, 'image/jpeg', 0.8)
            );

            // Send to backend
            const formData = new FormData();
            formData.append('file', blob, 'frame.jpg');

            const response = await fetch('/detect-live', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (this.isDetecting && data.success && data.detections) {
                this.drawDetections(data.detections);
            } else {
                // Clear if no detections
                this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
                this.currentDetections = [];  // Reset detection state
            }
        } catch (error) {
            console.error('Detection error:', error);
        } finally {
            this.detectionInProgress = false;
        }
    }

    drawDetections(detections) {
        if (!this.ctx) return;

        // Store current detections
        this.currentDetections = detections || [];

        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Draw each detection
        detections.forEach(det => {
            const [x1, y1, x2, y2] = det.bbox;

            // Draw bounding box
            this.ctx.strokeStyle = '#38bdf8';
            this.ctx.lineWidth = 3;
            this.ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

            // Prepare label
            const price = det.price !== null && det.price !== undefined
                ? `$${det.price.toFixed(2)}`
                : 'Fetching...';
            const label = `${det.name} - ${price}`;

            // Draw label background
            this.ctx.font = 'bold 16px Outfit, sans-serif';
            const textWidth = this.ctx.measureText(label).width;

            this.ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
            this.ctx.fillRect(x1, y1 - 30, textWidth + 20, 30);

            // Draw label text
            this.ctx.fillStyle = '#4ade80';
            this.ctx.fillText(label, x1 + 10, y1 - 8);

            // Draw confidence badge
            const confText = `${Math.round(det.confidence * 100)}%`;
            this.ctx.font = '12px Outfit, sans-serif';
            this.ctx.fillStyle = '#38bdf8';
            this.ctx.fillText(confText, x2 - 40, y1 - 8);

            // Add to session tracker if price is available
            if (sessionTracker && det.price !== null && det.price !== undefined) {
                sessionTracker.addCard({
                    card_id: det.card_id,
                    name: det.name,
                    price: det.price
                });
            }
        });
    }

    async captureFrame() {
        if (!this.video || this.video.readyState !== 4) {
            alert('Camera not ready');
            return;
        }

        // Stop detection while capturing
        const wasDetecting = this.isDetecting;
        if (wasDetecting) {
            this.stopDetection();
        }

        // Capture frame
        const canvas = document.createElement('canvas');
        canvas.width = this.video.videoWidth;
        canvas.height = this.video.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(this.video, 0, 0);

        canvas.toBlob(async (blob) => {
            // Send to existing analyze endpoint
            const formData = new FormData();
            formData.append('file', blob, 'capture.jpg');

            // Show loading
            document.getElementById('loading').hidden = false;

            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (data.success) {
                    // Stop camera and show result
                    this.stopCamera();
                    document.getElementById('camera-section').hidden = true;
                    document.getElementById('loading').hidden = true;

                    // Trigger existing result display
                    if (window.showResult) {
                        window.showResult(data);
                    }
                } else {
                    alert(data.message || 'Failed to analyze capture');
                    document.getElementById('loading').hidden = true;

                    // Resume detection
                    if (wasDetecting) {
                        this.startDetection();
                    }
                }
            } catch (error) {
                console.error('Capture error:', error);
                alert('Error analyzing capture');
                document.getElementById('loading').hidden = true;

                // Resume detection
                if (wasDetecting) {
                    this.startDetection();
                }
            }
        }, 'image/jpeg', 0.95);
    }
}

// Initialize camera manager and session tracker globally
window.cameraManager = new CameraManager();
sessionTracker = new SessionTracker();

// Setup clear session button
document.addEventListener('DOMContentLoaded', () => {
    const clearBtn = document.getElementById('clear-session-btn');
    if (clearBtn) {
        clearBtn.addEventListener('click', () => {
            if (confirm('Clear current session? This will reset the total and card count.')) {
                sessionTracker.clear();
            }
        });
    }
});
