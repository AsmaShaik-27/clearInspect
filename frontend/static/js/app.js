const resultsSection = document.getElementById('resultsSection');
const defectClass = document.getElementById('defectClass');
const severityBadge = document.getElementById('severityBadge');
const severityValue = document.getElementById('severityValue');
const statusBadge = document.getElementById('statusBadge');

const liveFeed = document.getElementById('liveFeed');

let statusInterval = null;

/* ------------------------
   Start Camera
------------------------- */
function startCamera() {
    liveFeed.src = "/video_feed";
    resultsSection.style.display = "block";

    // Start polling prediction status
    statusInterval = setInterval(fetchStatus, 500);
}

/* ------------------------
   Stop Camera
------------------------- */
function stopCamera() {
    liveFeed.src = "";
    clearInterval(statusInterval);
}

/* ------------------------
   Fetch Latest Prediction
------------------------- */
async function fetchStatus() {
    try {
        const response = await fetch("/api/live_status");
        const data = await response.json();

        if (!data.success) return;

        defectClass.textContent = data.predicted_class;
        severityBadge.textContent = data.severity_level;
        severityValue.textContent = data.severity_score + " / 100";

        statusBadge.textContent = data.is_defective
            ? "DEFECT DETECTED"
            : "NO DEFECT";

        statusBadge.style.backgroundColor = data.is_defective
            ? "#e74c3c"
            : "#2ecc71";

    } catch (err) {
        console.error("Live status error:", err);
    }
}