const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const previewContainer = document.getElementById('previewContainer');
const previewImage = document.getElementById('previewImage');
const removeBtn = document.getElementById('removeBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultsSection = document.getElementById('resultsSection');

const originalImage = document.getElementById('originalImage');
const heatmapImage = document.getElementById('heatmapImage');

const defectClass = document.getElementById('defectClass');
const severityBadge = document.getElementById('severityBadge');
const severityValue = document.getElementById('severityValue');
const statusBadge = document.getElementById('statusBadge');

let selectedFile = null;

uploadArea.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;

    selectedFile = file;
    const reader = new FileReader();
    reader.onload = () => {
        previewImage.src = reader.result;
        previewContainer.style.display = 'block';
        analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
});

removeBtn.addEventListener('click', () => {
    selectedFile = null;
    previewContainer.style.display = 'none';
    analyzeBtn.disabled = true;
});

analyzeBtn.addEventListener('click', async () => {

    const formData = new FormData();
    formData.append('image', selectedFile);

    const response = await fetch('/api/analyze', {
        method: 'POST',
        body: formData
    });

    const data = await response.json();

    if (!data.success) {
        alert(data.error);
        return;
    }

    originalImage.src = data.original_url;
    heatmapImage.src = data.heatmap_url;

    defectClass.textContent = data.predicted_class;
    severityBadge.textContent = data.severity_level;
    severityValue.textContent = data.severity_score + " / 100";

    statusBadge.textContent = data.is_defective ? "DEFECT DETECTED" : "NO DEFECT";

    resultsSection.style.display = 'block';
});
