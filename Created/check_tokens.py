#!/usr/bin/env python3

import pickle
import sys
sys.path.append('/home/bryan/Desktop/Allen/UniAD')

# Load results
with open('output/results_10.pkl', 'rb') as f:
    results = pickle.load(f)

print("Sample tokens in results:")
for i, result in enumerate(results):
    if 'token' in result:
        print(f"Sample {i}: {result['token']}")
    else:
        print(f"Sample {i}: No token found")
        print(f"Available keys: {list(result.keys())}")
        break