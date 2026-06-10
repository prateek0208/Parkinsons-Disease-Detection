document.getElementById('prediction-form').addEventListener('submit', async (e) => {
    e.preventDefault();

    const fo = parseFloat(document.getElementById('fo').value);
    const fhi = parseFloat(document.getElementById('fhi').value);
    const flo = parseFloat(document.getElementById('flo').value);
    const jitter = parseFloat(document.getElementById('jitter').value);

    // Provide default mean values for the remaining 18 features (based on the dataset)
    // In a real medical app, the user would upload a full audio profile.
    const defaultFeatures = [
        fo, fhi, flo, jitter, 
        0.00004, 0.003, 0.004, 0.009, // Jitter Abs, RAP, PPQ, DDP
        0.029, 0.282, 0.015, 0.017, // Shimmer, Shimmer dB, APQ3, APQ5
        0.024, 0.046, 0.024, 21.8, // APQ, DDA, NHR, HNR
        0.498, 0.718, -5.68, 0.226, // RPDE, DFA, spread1, spread2
        2.38, 0.206 // D2, PPE
    ];

    const btn = document.querySelector('.btn-3d');
    const originalText = btn.innerText;
    btn.innerText = 'Analyzing...';
    btn.disabled = true;

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ features: defaultFeatures })
        });

        const data = await response.json();

        const resultContainer = document.getElementById('result-container');
        const predictionText = document.getElementById('prediction-text');

        resultContainer.classList.remove('hidden');

        if (data.prediction === 1) {
            predictionText.innerText = "Parkinson's Detected";
            predictionText.className = "result-box parkinsons";
        } else {
            predictionText.innerText = "Healthy (No Parkinson's)";
            predictionText.className = "result-box healthy";
        }

    } catch (error) {
        alert("Error connecting to the diagnostic API.");
        console.error(error);
    } finally {
        btn.innerText = originalText;
        btn.disabled = false;
    }
});
