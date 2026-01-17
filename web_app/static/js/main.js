
let mediaRecorder;
let recordedChunks = [];
let stream;

const videoPreview = document.getElementById('videoPreview');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const submitBtn = document.getElementById('submitBtn');
const statusMsg = document.getElementById('statusMessage');
const loader = document.getElementById('loader');
const detectedEmotion = document.getElementById('detectedEmotion');
const metricsGrid = document.getElementById('metricsGrid');

// --- Pre-initialize Chart Data Structures (Optional, logic handles dynamic creation) ---

startBtn.addEventListener('click', async () => {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        videoPreview.srcObject = stream;
        videoPreview.muted = true; // Avoid feedback if audio was on

        mediaRecorder = new MediaRecorder(stream, { mimeType: 'video/webm' });

        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                recordedChunks.push(event.data);
            }
        };

        mediaRecorder.onstart = () => {
            recordedChunks = [];
            statusMsg.textContent = "Recording... limit processing to short clips (3-5s) for speed.";
            startBtn.disabled = true;
            stopBtn.disabled = false;
            submitBtn.disabled = true;

            // Visual indicator
            videoPreview.style.border = "2px solid var(--danger)";
        };

        mediaRecorder.onstop = () => {
            statusMsg.textContent = "Recording finished. Ready to analyze.";
            startBtn.disabled = false;
            stopBtn.disabled = true;
            submitBtn.disabled = false;
            videoPreview.style.border = "1px solid var(--glass-border)";

            // Stop stream tracks to release camera (optional, maybe keep it on for re-record)
            // stream.getTracks().forEach(track => track.stop());
        };

        mediaRecorder.start();

    } catch (error) {
        console.error("Error accessing camera:", error);
        statusMsg.textContent = "Error accessing camera. Please allow permissions.";
    }
});

stopBtn.addEventListener('click', () => {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
        // stream.getTracks().forEach(track => track.stop()); // Stop camera to indicate stop
    }
});

submitBtn.addEventListener('click', async () => {
    if (recordedChunks.length === 0) return;

    const blob = new Blob(recordedChunks, { type: 'video/webm' });
    const formData = new FormData();
    formData.append('video', blob, 'capture.webm');

    // UI Updates
    statusMsg.textContent = "Analyzing video... this may take a moment.";
    loader.classList.remove('hidden');
    submitBtn.disabled = true;
    detectedEmotion.textContent = "--";
    metricsGrid.innerHTML = ""; // Clear previous results

    try {
        const response = await fetch('/process', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Processing failed');

        const result = await response.json();
        console.log(result);

        // Update UI
        loader.classList.add('hidden');
        statusMsg.textContent = "Analysis Complete.";
        submitBtn.disabled = false;

        detectedEmotion.textContent = result.emotion;

        renderMetricsTables(result.surface_stats);

    } catch (error) {
        console.error("Error:", error);
        statusMsg.textContent = "Error during analysis.";
        loader.classList.add('hidden');
        submitBtn.disabled = false;
    }
});


function renderMetricsTables(stats) {
    // stats structure: { 'brow_left': {mean_mag: [], var: [], angle: []}, ... }

    const regions = ['Brow', 'Eye', 'Cheek', 'Mouth', 'Lips', 'Jaw'];

    // Create Layout
    regions.forEach(region => {
        // Create Card
        const card = document.createElement('div');
        card.className = 'region-card glass-panel';
        card.innerHTML = `<div class="region-title"><i class="fa-solid fa-table"></i> ${region} Region</div>`;

        // Data Preparation
        const leftKey = `${region.toLowerCase()}_left`;
        const rightKey = `${region.toLowerCase()}_right`;

        // Assume all arrays have the same length (synced by frames)
        // Get length from one available metric
        let dataLength = 0;
        if (stats[leftKey] && stats[leftKey].mean_mag) dataLength = stats[leftKey].mean_mag.length;
        else if (stats[rightKey] && stats[rightKey].mean_mag) dataLength = stats[rightKey].mean_mag.length;

        // Container for scrollable table
        const tableContainer = document.createElement('div');
        tableContainer.style.maxHeight = '300px';
        tableContainer.style.overflowY = 'auto';
        tableContainer.style.marginTop = '1rem';

        const table = document.createElement('table');
        table.className = 'metrics-table';
        table.style.width = '100%';
        table.style.borderCollapse = 'collapse';
        table.style.fontSize = '0.85rem';
        table.style.color = '#e2e8f0';

        // Header
        const thead = document.createElement('thead');
        thead.innerHTML = `
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.1); text-align: right;">
                <th style="padding: 8px; text-align: left;">Frame</th>
                <th style="padding: 8px; color: #60a5fa;">L-Mag</th>
                <th style="padding: 8px; color: #60a5fa;">L-Var</th>
                <th style="padding: 8px; color: #60a5fa;">L-Ang</th>
                <th style="padding: 8px; color: #f87171;">R-Mag</th>
                <th style="padding: 8px; color: #f87171;">R-Var</th>
                <th style="padding: 8px; color: #f87171;">R-Ang</th>
            </tr>
        `;
        table.appendChild(thead);

        // Body
        const tbody = document.createElement('tbody');

        for (let i = 0; i < dataLength; i++) {
            const row = document.createElement('tr');
            row.style.borderBottom = '1px solid rgba(255,255,255,0.05)';

            // Helper to safe get value
            const getVal = (sideKey, metric) => {
                const val = stats[sideKey] && stats[sideKey][metric] ? stats[sideKey][metric][i] : 0;
                return typeof val === 'number' ? val.toFixed(4) : val;
            };

            row.innerHTML = `
                <td style="padding: 6px; text-align: left;">${i}</td>
                <td style="padding: 6px; text-align: right;">${getVal(leftKey, 'mean_mag')}</td>
                <td style="padding: 6px; text-align: right;">${getVal(leftKey, 'var')}</td>
                <td style="padding: 6px; text-align: right;">${getVal(leftKey, 'angle')}</td>
                <td style="padding: 6px; text-align: right;">${getVal(rightKey, 'mean_mag')}</td>
                <td style="padding: 6px; text-align: right;">${getVal(rightKey, 'var')}</td>
                <td style="padding: 6px; text-align: right;">${getVal(rightKey, 'angle')}</td>
            `;
            tbody.appendChild(row);
        }
        table.appendChild(tbody);
        tableContainer.appendChild(table);
        card.appendChild(tableContainer);
        metricsGrid.appendChild(card);
    });
}
