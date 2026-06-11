import pandas as pd
import numpy as np

df = pd.read_csv('data/parkinsons.data')
df.drop(columns=['name'], inplace=True)

cols = [c for c in df.columns if c != 'status']

# Get diverse samples
healthy = df[df['status']==0]
parkinson = df[df['status']==1]

print("=" * 80)
print("DATASET OVERVIEW")
print("=" * 80)
print(f"Total Samples: {len(df)}")
print(f"Healthy (status=0): {len(healthy)} patients")
print(f"Parkinson's (status=1): {len(parkinson)} patients")
print()

# --- HEALTHY CASES ---
h_samples = [
    healthy.iloc[0],   # typical healthy
    healthy.iloc[5],   # another healthy
    healthy.iloc[10],  # edge case healthy
]

# --- PARKINSON'S CASES ---
p_samples = [
    parkinson.iloc[0],   # typical parkinson
    parkinson.iloc[20],  # moderate
    parkinson.iloc[50],  # severe markers
]

all_cases = [
    ("HEALTHY #1 - Stable Voice", h_samples[0], 0),
    ("HEALTHY #2 - Strong Harmonics", h_samples[1], 0),
    ("HEALTHY #3 - Edge Case", h_samples[2], 0),
    ("PARKINSON'S #1 - Classic Tremor", p_samples[0], 1),
    ("PARKINSON'S #2 - Moderate Symptoms", p_samples[1], 1),
    ("PARKINSON'S #3 - Severe Markers", p_samples[2], 1),
]

# Print each case
for name, sample, status in all_cases:
    print("=" * 80)
    label = "HEALTHY" if status == 0 else "PARKINSON'S"
    print(f"TEST CASE: {name}")
    print(f"Expected Result: {label}")
    print("-" * 80)
    
    # Key indicators
    fo = sample['MDVP:Fo(Hz)']
    jitter = sample['MDVP:Jitter(%)']
    shimmer = sample['MDVP:Shimmer']
    hnr = sample['HNR']
    spread1 = sample['spread1']
    ppe = sample['PPE']
    
    print(f"  KEY INDICATORS:")
    print(f"    Vocal Frequency (Fo):  {fo:.3f} Hz  {'(Normal: >150 Hz)' if fo > 150 else '(Low: vocal instability)'}")
    print(f"    Jitter:                {jitter:.5f}   {'(Normal: <0.005)' if jitter < 0.005 else '(High: frequency tremor)'}")
    print(f"    Shimmer:               {shimmer:.5f}   {'(Normal: <0.02)' if shimmer < 0.02 else '(High: amplitude tremor)'}")
    print(f"    HNR:                   {hnr:.3f}     {'(Good: >24 dB)' if hnr > 24 else '(Poor: noisy voice)'}")
    print(f"    spread1:               {spread1:.4f}  {'(Normal: < -6)' if spread1 < -6 else '(Abnormal: > -6)'}")
    print(f"    PPE:                   {ppe:.5f}   {'(Normal: <0.15)' if ppe < 0.15 else '(High: pitch instability)'}")
    print()
    print(f"  ALL 22 VALUES (copy-paste ready):")
    
    ids = ['fo','fhi','flo','jitter','jitter_abs','rap','ppq','ddp',
           'shimmer','shimmer_db','apq3','apq5','apq','dda',
           'nhr','hnr','rpde','dfa','spread1','spread2','d2','ppe']
    
    for i, c in enumerate(cols):
        print(f"    {ids[i]:12s} = {sample[c]}")
    print()

# Also print the averages for comparison
print("=" * 80)
print("AVERAGE VALUES FOR REFERENCE")
print("=" * 80)
print(f"{'Feature':<25} {'Healthy Avg':>15} {'Parkinson Avg':>15} {'Interpretation'}")
print("-" * 80)
key_features = ['MDVP:Fo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Shimmer', 'HNR', 'spread1', 'PPE', 'DFA', 'RPDE']
interpretations = [
    'Higher = healthier voice',
    'Lower = less tremor',
    'Lower = stable amplitude',
    'Higher = cleaner voice',
    'More negative = healthier',
    'Lower = stable pitch',
    'Lower = healthier signal',
    'Lower = more regular'
]
for feat, interp in zip(key_features, interpretations):
    h_avg = healthy[feat].mean()
    p_avg = parkinson[feat].mean()
    print(f"  {feat:<23} {h_avg:>15.5f} {p_avg:>15.5f}   {interp}")
