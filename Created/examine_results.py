#!/usr/bin/env python3

import pickle
import numpy as np

# Load the results
with open('output/results_10.pkl', 'rb') as f:
    results = pickle.load(f)

print(f"Results type: {type(results)}")
print(f"Number of results: {len(results)}")

# Examine the structure of the first result
if len(results) > 0:
    first_result = results[0]
    print(f"\nFirst result type: {type(first_result)}")
    
    if isinstance(first_result, dict):
        print("\nFirst result keys:")
        for key, value in first_result.items():
            print(f"  {key}: {type(value)}")
            if hasattr(value, 'shape'):
                print(f"    shape: {value.shape}")
            elif isinstance(value, (list, tuple)):
                print(f"    length: {len(value)}")
                if len(value) > 0:
                    print(f"    first element type: {type(value[0])}")
                    if hasattr(value[0], 'shape'):
                        print(f"    first element shape: {value[0].shape}")
    elif isinstance(first_result, (list, tuple)):
        print(f"First result length: {len(first_result)}")
        if len(first_result) > 0:
            print(f"First element type: {type(first_result[0])}")
    elif hasattr(first_result, 'shape'):
        print(f"First result shape: {first_result.shape}")
    
    print(f"\nFirst result: {first_result}")