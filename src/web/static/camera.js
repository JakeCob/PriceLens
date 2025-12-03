// Audio Manager for sound effects
class AudioManager {
    constructor() {
        this.enabled = true;
        this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    }

    toggle() {
        this.enabled = !this.enabled;
        return this.enabled;
    }

    play(type) {
        if (!this.enabled) return;

        // Resume context if suspended (browser policy)
        if (this.ctx.state === 'suspended') {
            this.ctx.resume();
        }

        switch (type) {
            case 'new_card':
                this.playCoinSound();
                break;
            case 'high_value':
                this.playJackpotSound();
                break;
        }
    }

    playCoinSound() {
        const t = this.ctx.currentTime;

        // First coin hit
        this.createOscillator(1200, t, 0.1, 'sine');
        this.createOscillator(1600, t + 0.05, 0.2, 'sine');
    }

    playJackpotSound() {
        const t = this.ctx.currentTime;
        // Rapid arpeggio
        [523.25, 659.25, 783.99, 1046.50, 1318.51, 1567.98].forEach((freq, i) => {
            this.createOscillator(freq, t + i * 0.08, 0.1, 'square', 0.1);
        });
    }

    createOscillator(freq, startTime, duration, type = 'sine', volume = 0.1) {
        const osc = this.ctx.createOscillator();
        const gain = this.ctx.createGain();

        osc.type = type;
        osc.frequency.value = freq;

        gain.gain.setValueAtTime(volume, startTime);
        gain.gain.exponentialRampToValueAtTime(0.01, startTime + duration);

        osc.connect(gain);
        gain.connect(this.ctx.destination);

        osc.start(startTime);
        osc.stop(startTime + duration);
    }
}

// Currency Manager for multi-currency support
class CurrencyManager {
    constructor() {
        // Exchange rates (USD base) - Updated 2024-11-25
        this.rates = {
            'USD': { rate: 1.00, symbol: '$', name: 'US Dollar' },
            'EUR': { rate: 0.92, symbol: '€', name: 'Euro' },
            'GBP': { rate: 0.79, symbol: '£', name: 'British Pound' },
            'PHP': { rate: 56.50, symbol: '₱', name: 'Philippine Peso' },
            'JPY': { rate: 149.50, symbol: '¥', name: 'Japanese Yen' },
            'CAD': { rate: 1.39, symbol: 'C$', name: 'Canadian Dollar' }
        };

        this.currentCurrency = this.loadPreference();
    }

    loadPreference() {
        return localStorage.getItem('currency') || 'USD';
    }

    savePreference(currency) {
        localStorage.setItem('currency', currency);
        this.currentCurrency = currency;
    }

    convert(usdAmount, toCurrency = null) {
        const currency = toCurrency || this.currentCurrency;
        if (!this.rates[currency]) return usdAmount;

        const rate = this.rates[currency].rate;
        return usdAmount * rate;
    }

    format(usdAmount, toCurrency = null) {
        const currency = toCurrency || this.currentCurrency;
        if (!this.rates[currency]) return `$${usdAmount.toFixed(2)}`;

        const converted = this.convert(usdAmount, currency);
        const symbol = this.rates[currency].symbol;

        // JPY doesn't use decimals
        if (currency === 'JPY') {
            return `${symbol}${Math.round(converted).toLocaleString()}`;
        }

        return `${symbol}${converted.toFixed(2)}`;
    }

    getCurrencies() {
        return Object.keys(this.rates);
    }
}

// Initialize global managers
const audioManager = new AudioManager();
const currencyManager = new CurrencyManager();

/**
 * SessionTracker - Manages scanning session state with stability checks
 */
class SessionTracker {
    constructor() {
        this.detectedCards = new Set(); // Set of committed card IDs
        this.cards = []; // List of committed card objects
        this.runningTotal = 0;

        // Stability tracking
        this.pendingCards = new Map(); // card_id -> { firstSeen, votes: {}, lastData }
        this.cardLastSeen = new Map(); // card_name -> timestamp

        this.load();
    }

    addCard(cardId, cardData) {
        const now = Date.now();

        // 1. If already committed, just update last seen
        if (this.detectedCards.has(cardId)) {
            this.cardLastSeen.set(cardData.name, now);
            return false;
        }

        // 2. Handle Pending State
        if (!this.pendingCards.has(cardId)) {
            this.pendingCards.set(cardId, {
                firstSeen: now,
                votes: {},
                lastData: cardData
            });
        }

        const pending = this.pendingCards.get(cardId);
        pending.lastData = cardData;

        // Vote for the identified name
        const name = cardData.name;
        pending.votes[name] = (pending.votes[name] || 0) + 1;

        // Early commit: If all votes (2+) are unanimous, commit immediately
        let totalVotes = 0;
        for (const count of Object.values(pending.votes)) {
            totalVotes += count;
        }

        if (totalVotes >= 2) {
            const votes = Object.values(pending.votes);
            const maxVote = Math.max(...votes);

            // All votes agree on the same card
            if (maxVote >= 2 && maxVote === totalVotes) {
                this.commitCard(cardId, pending.lastData);
                this.pendingCards.delete(cardId);
                return true;
            }
        }

        // 3. Check for Commitment (Stability Check)
        // Require 0.5 seconds of stability
        if (now - pending.firstSeen > 500) { // Reduced from 1000ms to 500ms
            // Find most frequent name
            let bestName = null;
            let maxVotes = 0;
            let totalVotes = 0;

            for (const [n, count] of Object.entries(pending.votes)) {
                totalVotes += count;
                if (count > maxVotes) {
                    maxVotes = count;
                    bestName = n;
                }
            }

            // Require 70% consensus
            if (bestName && (maxVotes / totalVotes > 0.6)) { // Reduced from 70% to 60%
                this.commitCard(cardId, pending.lastData);
                this.pendingCards.delete(cardId);
                return true;
            } else {
                // Reset if unstable (or maybe just keep waiting?)
                // For now, let's keep waiting but cap it to avoid memory leaks
                if (now - pending.firstSeen > 5000) {
                    this.pendingCards.delete(cardId); // Give up
                }
            }
        }

        return false;
    }

    commitCard(cardId, cardData) {
        const now = Date.now();

        // 4. Re-identification / De-duplication Check
        // Check if we saw this same card name recently (< 5 seconds ago)
        const lastSeenTime = this.cardLastSeen.get(cardData.name);
        if (lastSeenTime && (now - lastSeenTime < 5000)) {
            console.log(`Duplicate prevented: ${cardData.name} (seen ${(now - lastSeenTime) / 1000}s ago)`);
            // Mark as detected so we don't process it again, but don't add to total
            this.detectedCards.add(cardId);
            this.cardLastSeen.set(cardData.name, now);
            // Optional: Play a subtle "merge" beep?
            return;
        }

        // 5. Commit new card
        this.detectedCards.add(cardId);
        this.cardLastSeen.set(cardData.name, now);

        const normalizedPrice = this._parsePrice(cardData.price);

        this.cards.push({
            id: cardId,
            name: cardData.name,
            price: normalizedPrice,
            timestamp: new Date().toISOString()
        });

        this.runningTotal += normalizedPrice;

        this.save();
        this.updateDisplay();

        // Play sound
        if (normalizedPrice > 50) {
            audioManager.play('high_value');
        } else {
            audioManager.play('new_card');
        }
    }

    isCommitted(cardId) {
        return this.detectedCards.has(cardId);
    }

    clear() {
        this.detectedCards.clear();
        this.cards = [];
        this.runningTotal = 0;
        this.pendingCards.clear();
        this.cardLastSeen.clear();
        this.save();
        this.updateDisplay();
    }

    save() {
        try {
            localStorage.setItem('session', JSON.stringify({
                cards: this.cards,
                total: this.runningTotal,
                // We don't save pending state
            }));
        } catch (error) {
            console.error('Failed to save session:', error);
        }
    }

    load() {
        try {
            const saved = localStorage.getItem('session');
            if (saved) {
                const data = JSON.parse(saved);
                this.cards = data.cards || [];
                this.runningTotal = data.total || 0;
                this.detectedCards = new Set(this.cards.map(c => c.id));

                // Restore cardLastSeen from loaded cards (approximate)
                const now = Date.now();
                this.cards.forEach(c => {
                    // Assume loaded cards were seen "long ago" so they don't block new ones?
                    // Or maybe we shouldn't restore this history if the session is old.
                    // Let's leave it empty, so reloading page allows re-scanning.
                });
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
            totalElement.textContent = currencyManager.format(this.runningTotal);
            // Add pulse animation
            totalElement.classList.add('updating');
            setTimeout(() => totalElement.classList.remove('updating'), 300);
        }

        if (countElement) {
            const cardText = this.cards.length === 1 ? 'card' : 'cards';
            countElement.textContent = `📦 ${this.cards.length} ${cardText}`;
        }
    }

    _parsePrice(priceValue) {
        if (priceValue === null || priceValue === undefined) {
            return 0;
        }

        // Handle object shapes like { market: 12.34 }
        if (typeof priceValue === 'object') {
            const candidate = priceValue.market ?? priceValue.price ?? priceValue.value;
            return this._parsePrice(candidate);
        }

        if (typeof priceValue === 'number') {
            return Number.isFinite(priceValue) ? priceValue : 0;
        }

        if (typeof priceValue === 'string') {
            const cleaned = priceValue.replace(/[^0-9.]/g, '');
            const parsed = parseFloat(cleaned);
            return Number.isFinite(parsed) ? parsed : 0;
        }

        return 0;
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
        this.consecutiveEmptyFrames = 0;  // Track frames with no detections
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
        this.consecutiveEmptyFrames = 0;
        console.log('Starting live detection...');

        // Start with fast interval
        this.scheduleNextDetection(500);
    }

    scheduleNextDetection(delay) {
        if (!this.isDetecting) return;

        this.detectionInterval = setTimeout(() => {
            if (!this.detectionInProgress) {
                this.detectFrame();
            } else {
                // If still processing, try again in 100ms
                this.scheduleNextDetection(100);
            }
        }, delay);
    }

    stopDetection() {
        this.isDetecting = false;
        if (this.detectionInterval) {
            clearTimeout(this.detectionInterval);  // Changed from clearInterval
            this.detectionInterval = null;
        }
        // Clear canvas and reset detection state
        if (this.ctx) {
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        }
        this.currentDetections = [];  // Reset detection state
        this.consecutiveEmptyFrames = 0;
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

            if (this.isDetecting && data.success && data.detections && data.detections.length > 0) {
                this.drawDetections(data.detections);
                this.consecutiveEmptyFrames = 0;
                // Fast interval when actively detecting
                this.scheduleNextDetection(500);
            } else {
                // No detections - adapt interval
                this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
                this.currentDetections = [];  // Reset detection state
                this.consecutiveEmptyFrames++;

                // Adaptive: slow down if no detections for a while
                let nextDelay = 1000;  // Default: 1 second
                if (this.consecutiveEmptyFrames > 3) {
                    nextDelay = 1500;  // Idle: 1.5 seconds
                }
                this.scheduleNextDetection(nextDelay);
            }
        } catch (error) {
            console.error('Detection error:', error);
            // On error, try again after 1 second
            if (this.isDetecting) {
                this.scheduleNextDetection(1000);
            }
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
            const isCommitted = sessionTracker ? sessionTracker.isCommitted(det.card_id) : false;

            // Draw bounding box
            this.ctx.strokeStyle = isCommitted ? '#4ade80' : '#facc15'; // Green if committed, Yellow if pending
            this.ctx.lineWidth = 3;
            this.ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

            // Prepare label
            let priceText = 'Fetching...';
            if (det.price !== null && det.price !== undefined) {
                priceText = currencyManager.format(det.price);
            }

            let label = `${det.name} - ${priceText}`;
            if (!isCommitted) {
                label = `Verifying... ${det.name}`;
            }

            // Draw label with adaptive positioning
            this.ctx.font = 'bold 16px Outfit, sans-serif';
            const textWidth = this.ctx.measureText(label).width;

            const labelHeight = 30;
            const labelPadding = 10;
            let labelY, textY;

            // Adaptive positioning: check if there's enough space above the box
            if (y1 >= 40) {
                // Enough space above - draw label above box (preferred)
                labelY = y1 - labelHeight;
                textY = y1 - 8;
            } else {
                // Not enough space - draw label inside box at the top
                labelY = y1 + 2;
                textY = y1 + 22;
            }

            // Draw label background
            this.ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
            this.ctx.fillRect(x1, labelY, textWidth + 20, labelHeight);

            // Draw label text
            this.ctx.fillStyle = isCommitted ? '#4ade80' : '#facc15';
            this.ctx.fillText(label, x1 + labelPadding, textY);

            // Draw confidence badge (aligned with label)
            const confText = `${Math.round(det.confidence * 100)}%`;
            this.ctx.font = '12px Outfit, sans-serif';
            this.ctx.fillStyle = '#38bdf8';
            this.ctx.fillText(confText, x2 - 40, textY);

            // Add to session tracker if price is available
            if (sessionTracker && det.price !== null && det.price !== undefined) {
                sessionTracker.addCard(det.card_id, {
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

    const muteBtn = document.getElementById('mute-btn');
    if (muteBtn) {
        muteBtn.addEventListener('click', () => {
            const enabled = audioManager.toggle();
            muteBtn.textContent = enabled ? '🔊' : '🔇';
            muteBtn.classList.toggle('muted', !enabled);
        });
    }

    const currencySelector = document.getElementById('currency-selector');
    if (currencySelector) {
        // Set initial value from preference
        currencySelector.value = currencyManager.currentCurrency;

        currencySelector.addEventListener('change', (e) => {
            currencyManager.savePreference(e.target.value);
            // Update session display
            if (sessionTracker) {
                sessionTracker.updateDisplay();
            }
        });
    }
});
