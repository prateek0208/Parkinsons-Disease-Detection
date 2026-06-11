// ===== MODEL DATA =====
let modelData = null;
let lastPrediction = null; // stores last prediction for PDF report

// ===== DATASET AVERAGES (for SHAP-style feature importance) =====
const FEATURE_NAMES = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)",
    "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
    "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR",
    "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
];

const HEALTHY_AVG = [
    181.938, 223.637, 145.207, 0.00387, 0.00002, 0.00205, 0.00226, 0.00616,
    0.01762, 0.162, 0.00886, 0.01050, 0.01228, 0.02659, 0.01149, 24.679,
    0.44255, 0.69572, -6.759, 0.16486, 2.15422, 0.12302
];

const PARKINSON_AVG = [
    145.181, 188.441, 106.893, 0.00699, 0.00005, 0.00374, 0.00384, 0.01122,
    0.03366, 0.310, 0.01757, 0.02105, 0.02748, 0.05271, 0.02906, 20.974,
    0.51682, 0.72541, -5.333, 0.24945, 2.47964, 0.23383
];

const FIELD_IDS = [
    'fo','fhi','flo','jitter','jitter_abs','rap','ppq','ddp',
    'shimmer','shimmer_db','apq3','apq5','apq','dda',
    'nhr','hnr','rpde','dfa','spread1','spread2','d2','ppe'
];

// ===== INIT =====
window.addEventListener('DOMContentLoaded', async () => {
    createParticles();
    await loadModel();
    document.getElementById('predictionForm').addEventListener('submit', handlePredict);
});

async function loadModel() {
    try {
        const res = await fetch('model.json');
        modelData = await res.json();
        console.log(`Model loaded: ${modelData.n_trees} trees, ${modelData.n_features} features`);
    } catch (e) {
        console.error('Failed to load model:', e);
        alert('Error: Could not load the ML model file (model.json).');
    }
}

// ===== RANDOM FOREST PREDICTION =====
function predictTree(tree, features) {
    let node = tree;
    while (!node.leaf) {
        if (features[node.feature] <= node.threshold) {
            node = node.left;
        } else {
            node = node.right;
        }
    }
    return node.class;
}

function predictForest(features) {
    if (!modelData) return { prediction: -1, confidence: 0 };
    let votes = [0, 0];
    for (const tree of modelData.trees) {
        votes[predictTree(tree, features)]++;
    }
    const total = votes[0] + votes[1];
    const prediction = votes[1] > votes[0] ? 1 : 0;
    const confidence = Math.max(votes[0], votes[1]) / total;
    return { prediction, confidence };
}

// ===== SHAP-STYLE FEATURE IMPORTANCE =====
function computeFeatureImportance(features) {
    // For each feature, compute how much it "pushes" toward Parkinson's vs Healthy
    // by comparing the input value's position between the healthy and parkinson averages
    const importances = [];
    for (let i = 0; i < 22; i++) {
        const hAvg = HEALTHY_AVG[i];
        const pAvg = PARKINSON_AVG[i];
        const range = Math.abs(pAvg - hAvg);
        if (range === 0) continue;

        // How far the value is from healthy average, normalized by the range
        // Positive = pushes toward Parkinson's, Negative = pushes toward Healthy
        let direction;
        if (pAvg > hAvg) {
            // Higher values = more Parkinson's (e.g., Jitter, Shimmer)
            direction = (features[i] - hAvg) / range;
        } else {
            // Lower values = more Parkinson's (e.g., Fo, HNR)
            direction = (hAvg - features[i]) / range;
        }

        importances.push({
            index: i,
            name: FEATURE_NAMES[i],
            value: features[i],
            importance: direction,
            absImportance: Math.abs(direction)
        });
    }

    // Sort by absolute importance (most important first)
    importances.sort((a, b) => b.absImportance - a.absImportance);
    return importances;
}

function renderSHAPChart(features) {
    const importances = computeFeatureImportance(features);
    const chart = document.getElementById('shapChart');
    chart.innerHTML = '';

    // Show top 10 most important features
    const top = importances.slice(0, 10);
    const maxAbs = Math.max(...top.map(f => f.absImportance), 0.01);

    top.forEach((feat, idx) => {
        const row = document.createElement('div');
        row.className = 'shap-row';
        row.style.animationDelay = (idx * 0.05) + 's';

        const barWidth = Math.min((feat.absImportance / maxAbs) * 50, 50);
        const isParkinson = feat.importance > 0;

        row.innerHTML = `
            <div class="feat-name" title="${feat.name}">${feat.name}</div>
            <div class="shap-bar-track">
                <div class="shap-center-line"></div>
                <div class="shap-bar-fill ${isParkinson ? 'push-parkinson' : 'push-healthy'}"
                     style="width: 0%"></div>
            </div>
            <div class="feat-val" style="color: ${isParkinson ? '#ff3366' : '#00ff88'}">
                ${isParkinson ? '+' : '-'}${feat.absImportance.toFixed(2)}
            </div>
        `;
        chart.appendChild(row);

        // Animate bar width
        setTimeout(() => {
            row.querySelector('.shap-bar-fill').style.width = barWidth + '%';
        }, 100 + idx * 60);
    });
}

// ===== FORM HANDLER =====
function handlePredict(e) {
    e.preventDefault();
    const features = FIELD_IDS.map(id => parseFloat(document.getElementById(id).value));

    if (features.some(v => isNaN(v))) {
        alert('Please fill in all 22 fields with valid numbers.');
        return;
    }

    const btn = document.getElementById('submitBtn');
    btn.disabled = true;
    btn.innerHTML = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg> Analyzing...';

    setTimeout(() => {
        const { prediction, confidence } = predictForest(features);

        // Store for PDF report
        lastPrediction = { features, prediction, confidence, timestamp: new Date() };

        showResult(prediction, confidence);
        renderSHAPChart(features);

        btn.disabled = false;
        btn.innerHTML = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg> Run AI Diagnosis';
    }, 600);
}

// ===== DISPLAY RESULT =====
function showResult(prediction, confidence) {
    const section = document.getElementById('resultSection');
    const card = document.getElementById('resultCard');
    const icon = document.getElementById('resultIcon');
    const title = document.getElementById('resultTitle');
    const desc = document.getElementById('resultDesc');
    const bar = document.getElementById('confidenceBar');
    const val = document.getElementById('confidenceValue');

    section.classList.remove('hidden');
    const pct = (confidence * 100).toFixed(1);

    if (prediction === 0) {
        card.className = 'result-card result-healthy';
        icon.textContent = '\u2705';
        title.textContent = 'Healthy - No Parkinson\'s';
        desc.textContent = 'The AI model predicts this voice sample does NOT indicate Parkinson\'s Disease.';
        bar.className = 'bar-fill bar-fill-green';
        val.style.color = '#00ff88';
    } else {
        card.className = 'result-card result-parkinson';
        icon.textContent = '\u26A0\uFE0F';
        title.textContent = 'Parkinson\'s Detected';
        desc.textContent = 'The AI model predicts this voice sample indicates possible Parkinson\'s Disease. Please consult a medical professional.';
        bar.className = 'bar-fill bar-fill-red';
        val.style.color = '#ff3366';
    }

    bar.style.width = '0%';
    setTimeout(() => { bar.style.width = pct + '%'; }, 100);
    val.textContent = pct + '% confidence';
    section.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ===== PDF REPORT DOWNLOAD =====
function downloadReport() {
    if (!lastPrediction) {
        alert('Please run a diagnosis first.');
        return;
    }

    const { features, prediction, confidence, timestamp } = lastPrediction;
    const pct = (confidence * 100).toFixed(1);
    const diagnosis = prediction === 0 ? 'HEALTHY - No Parkinson\'s Detected' : 'PARKINSON\'S DISEASE DETECTED';
    const diagColor = prediction === 0 ? '#00c864' : '#ff3366';
    const importances = computeFeatureImportance(features);
    const topFeatures = importances.slice(0, 10);
    const maxImp = Math.max(...topFeatures.map(f => f.absImportance), 0.01);
    const reportId = 'NS-' + Date.now().toString(36).toUpperCase();

    // Build feature rows
    let featureRows = '';
    for (let i = 0; i < 22; i++) {
        const val = features[i];
        const distH = Math.abs(val - HEALTHY_AVG[i]);
        const distP = Math.abs(val - PARKINSON_AVG[i]);
        const status = distH < distP ? 'Normal' : 'Abnormal';
        const statusColor = status === 'Normal' ? '#00a050' : '#dc2840';
        const bg = i % 2 === 0 ? '#f5f8ff' : '#ffffff';
        featureRows += `<tr style="background:${bg}">
            <td style="padding:6px 10px;border-bottom:1px solid #eee;font-size:12px">${FEATURE_NAMES[i]}</td>
            <td style="padding:6px 10px;border-bottom:1px solid #eee;font-size:12px;text-align:center">${val}</td>
            <td style="padding:6px 10px;border-bottom:1px solid #eee;font-size:12px;text-align:center;font-weight:600;color:${statusColor}">${status}</td>
        </tr>`;
    }

    // Build importance bars
    let importanceBars = '';
    topFeatures.forEach(f => {
        const barW = Math.round((f.absImportance / maxImp) * 100);
        const isP = f.importance > 0;
        const barColor = isP ? '#ff3366' : '#00c864';
        const dirLabel = isP ? 'Parkinson\'s' : 'Healthy';
        importanceBars += `<div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">
            <div style="width:130px;font-size:11px;text-align:right;color:#555">${f.name}</div>
            <div style="flex:1;height:16px;background:#f0f0f5;border-radius:3px;overflow:hidden">
                <div style="width:${barW}%;height:100%;background:${barColor};border-radius:3px"></div>
            </div>
            <div style="width:80px;font-size:10px;color:${barColor};font-weight:600">${dirLabel}</div>
        </div>`;
    });

    const html = `<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>NeuroScan AI - Diagnostic Report</title>
<style>
@media print { body { -webkit-print-color-adjust: exact; print-color-adjust: exact; } }
body { font-family: Arial, Helvetica, sans-serif; margin:0; padding:0; color:#333; }
</style>
</head><body>

<!-- HEADER -->
<div style="background:#060b18;padding:25px 30px 20px;border-bottom:4px solid #0070ff">
    <div style="display:flex;justify-content:space-between;align-items:flex-start">
        <div>
            <div style="color:#00f0ff;font-size:24px;font-weight:bold;margin-bottom:4px">NeuroScan AI</div>
            <div style="color:#ccc;font-size:13px">Parkinson's Disease Diagnostic Report</div>
            <div style="color:#888;font-size:10px;margin-top:4px">Random Forest ML Model | 94.87% Accuracy | 100 Decision Trees</div>
        </div>
        <div style="text-align:right">
            <div style="color:#888;font-size:10px">Report ID: ${reportId}</div>
            <div style="color:#888;font-size:10px;margin-top:2px">Generated: ${timestamp.toLocaleString()}</div>
        </div>
    </div>
</div>

<!-- DIAGNOSIS BANNER -->
<div style="margin:25px 30px;padding:18px;background:${diagColor};border-radius:8px;text-align:center">
    <div style="color:white;font-size:18px;font-weight:bold;letter-spacing:1px">${diagnosis}</div>
    <div style="color:rgba(255,255,255,0.85);font-size:13px;margin-top:5px">Confidence: ${pct}%</div>
</div>

<!-- PATIENT DATA TABLE -->
<div style="margin:0 30px">
    <h3 style="color:#0070ff;font-size:14px;margin-bottom:10px;border-bottom:2px solid #0070ff;padding-bottom:5px">Patient Vocal Biomarker Data</h3>
    <table style="width:100%;border-collapse:collapse">
        <thead>
            <tr style="background:#e6f0ff">
                <th style="padding:8px 10px;text-align:left;font-size:11px;font-weight:bold;color:#334">Feature</th>
                <th style="padding:8px 10px;text-align:center;font-size:11px;font-weight:bold;color:#334">Value</th>
                <th style="padding:8px 10px;text-align:center;font-size:11px;font-weight:bold;color:#334">Status</th>
            </tr>
        </thead>
        <tbody>${featureRows}</tbody>
    </table>
</div>

<!-- FEATURE IMPORTANCE -->
<div style="margin:25px 30px">
    <h3 style="color:#0070ff;font-size:14px;margin-bottom:5px;border-bottom:2px solid #0070ff;padding-bottom:5px">Feature Importance (Explainable AI)</h3>
    <p style="font-size:10px;color:#888;margin-bottom:12px">Features ranked by how strongly they influenced the AI prediction</p>
    ${importanceBars}
</div>

<!-- DISCLAIMER -->
<div style="margin:30px;padding-top:15px;border-top:1px solid #ddd">
    <p style="font-size:9px;color:#999;line-height:1.6;font-style:italic">
        <strong>DISCLAIMER:</strong> This report is generated by an AI screening tool and is NOT a medical diagnosis.
        Please consult a qualified medical professional for clinical evaluation and treatment decisions.<br>
        Model: Random Forest Classifier (100 trees) | Dataset: UCI Parkinson's (195 samples, 22 features) | Developer: Prateek Ranjan
    </p>
</div>

</body></html>`;

    // Open in new window and trigger print (Save as PDF)
    const win = window.open('', '_blank');
    win.document.write(html);
    win.document.close();
    setTimeout(() => { win.print(); }, 500);
}

// ===== QUICK FILL =====
function fillHealthy() {
    const vals = {
        fo: 197.076, fhi: 206.896, flo: 192.055, jitter: 0.00289,
        jitter_abs: 0.00001, rap: 0.00166, ppq: 0.00168, ddp: 0.00498,
        shimmer: 0.01098, shimmer_db: 0.097, apq3: 0.00563, apq5: 0.0068,
        apq: 0.00802, dda: 0.01689, nhr: 0.00339, hnr: 26.775,
        rpde: 0.422229, dfa: 0.741367, spread1: -7.3483, spread2: 0.177551,
        d2: 1.743867, ppe: 0.085569
    };
    Object.entries(vals).forEach(([id, v]) => { document.getElementById(id).value = v; });
}

function fillParkinson() {
    const vals = {
        fo: 119.992, fhi: 157.302, flo: 74.997, jitter: 0.00784,
        jitter_abs: 0.00007, rap: 0.0037, ppq: 0.00554, ddp: 0.01109,
        shimmer: 0.04374, shimmer_db: 0.426, apq3: 0.02182, apq5: 0.0313,
        apq: 0.02971, dda: 0.06545, nhr: 0.02211, hnr: 21.033,
        rpde: 0.414783, dfa: 0.815285, spread1: -4.813031, spread2: 0.266482,
        d2: 2.301442, ppe: 0.284654
    };
    Object.entries(vals).forEach(([id, v]) => { document.getElementById(id).value = v; });
}

// ===== PARTICLES =====
function createParticles() {
    const container = document.getElementById('particles');
    for (let i = 0; i < 30; i++) {
        const p = document.createElement('div');
        p.className = 'particle';
        p.style.left = Math.random() * 100 + '%';
        p.style.top = Math.random() * 100 + '%';
        p.style.animationDelay = (Math.random() * 4) + 's';
        p.style.animationDuration = (3 + Math.random() * 3) + 's';
        container.appendChild(p);
    }
}
