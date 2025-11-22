document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const loadingOverlay = document.getElementById('loading');
    const resultSection = document.getElementById('result-section');
    const btnReset = document.getElementById('btn-reset');

    // Elements to update
    const resultImage = document.getElementById('result-image');
    const cardName = document.getElementById('card-name');
    const cardSet = document.getElementById('card-set');
    const cardPrice = document.getElementById('card-price');
    const cardConfidence = document.getElementById('card-confidence');
    const cardId = document.getElementById('card-id');

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

            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                showResult(data);
            } else {
                alert(data.message || 'Failed to analyze image');
                loadingOverlay.hidden = true;
            }
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while processing the image');
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

        // Format price
        if (data.price && data.price.market) {
            cardPrice.textContent = `$${data.price.market.toFixed(2)}`;
        } else {
            cardPrice.textContent = 'N/A';
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
});
