document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const loadingOverlay = document.getElementById('loading');
    const resultSection = document.getElementById('result-section');
    const btnReset = document.getElementById('btn-reset');

    // Mode switching
    const btnUploadMode = document.getElementById('btn-upload-mode');
    const btnCameraMode = document.getElementById('btn-camera-mode');
    const cameraSection = document.getElementById('camera-section');
    const btnStartCamera = document.getElementById('btn-start-camera');
    const btnStopCamera = document.getElementById('btn-stop-camera');
    const btnCapture = document.getElementById('btn-capture');

    // Elements to update
    const resultImage = document.getElementById('result-image');
    const cardName = document.getElementById('card-name');
    const cardSet = document.getElementById('card-set');
    const cardPrice = document.getElementById('card-price');
    const cardConfidence = document.getElementById('card-confidence');
    const cardId = document.getElementById('card-id');

    // Mode switching handlers
    btnUploadMode.addEventListener('click', () => {
        btnUploadMode.classList.add('active');
        btnCameraMode.classList.remove('active');
        dropZone.hidden = false;
        cameraSection.hidden = true;
        if (window.cameraManager) {
            window.cameraManager.stopCamera();
        }
    });

    btnCameraMode.addEventListener('click', () => {
        btnCameraMode.classList.add('active');
        btnUploadMode.classList.remove('active');
        dropZone.hidden = true;
        cameraSection.hidden = false;
        resultSection.hidden = true;
    });

    // Camera control handlers
    btnStartCamera.addEventListener('click', async () => {
        const started = await window.cameraManager.startCamera();
        if (started) {
            btnStartCamera.hidden = true;
            btnStopCamera.hidden = false;
            btnCapture.hidden = false;
            window.cameraManager.startDetection();
        }
    });

    btnStopCamera.addEventListener('click', () => {
        window.cameraManager.stopCamera();
        btnStartCamera.hidden = false;
        btnStopCamera.hidden = true;
        btnCapture.hidden = true;
    });

    btnCapture.addEventListener('click', () => {
        window.cameraManager.captureFrame();
    });

    // Drag & Drop Handlers
    dropZone.addEventListener('click', () => fileInput.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');

        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            handleFile(fileInput.files[0]);
        }
    });

    btnReset.addEventListener('click', resetUI);

    async function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file');
            return;
        }

        // Show loading
        loadingOverlay.hidden = false;

        try {
            const formData = new FormData();
            formData.append('file', file);

            // Create AbortController for timeout
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 120000); // 120s timeout

            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData,
                signal: controller.signal
            });

            clearTimeout(timeoutId);
            const data = await response.json();

            if (data.success) {
                showResult(data);
            } else {
                alert(data.message || 'Failed to analyze image');
                loadingOverlay.hidden = true;
            }
        } catch (error) {
            console.error('Error:', error);
            if (error.name === 'AbortError') {
                alert('Request timed out. The price API is very slow. The card was identified, but price could not be fetched in time.');
            } else {
                alert('An error occurred while processing the image');
            }
            loadingOverlay.hidden = true;
        }
    }

    function showResult(data) {
        // Update UI with data
        resultImage.src = data.image;
        cardName.textContent = data.card.name;
        cardSet.textContent = data.card.set;
        cardId.textContent = data.card.id;

        // Format confidence
        const confPercent = Math.round(data.card.confidence * 100);
        cardConfidence.textContent = `${confPercent}%`;

        // Format price with better feedback
        if (data.price && data.price.market !== null && data.price.market !== undefined) {
            cardPrice.textContent = `$${data.price.market.toFixed(2)}`;
            cardPrice.classList.remove('fetching');
        } else {
            // Show N/A but indicate it might be a temporary issue
            cardPrice.textContent = 'N/A';
            cardPrice.classList.remove('fetching');
            cardPrice.title = 'Price not available (API timeout or card not found)';
        }

        // Hide upload, show result
        loadingOverlay.hidden = true;
        dropZone.hidden = true;
        resultSection.hidden = false;
    }

    function resetUI() {
        fileInput.value = '';
        resultImage.src = '';
        dropZone.hidden = false;
        resultSection.hidden = true;
    }

    // Expose functions globally for camera.js
    window.showResult = showResult;
    window.resetUI = resetUI;
});
