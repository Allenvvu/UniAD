# NetworkX compatibility patch for Python 3.9
# Place this file in your UniAD root directory and run before evaluation

import sys
import math

# Add gcd back to fractions module for NetworkX 2.2 compatibility
try:
    import fractions
    if not hasattr(fractions, 'gcd'):
        fractions.gcd = math.gcd
        print("Applied NetworkX compatibility patch for Python 3.9")
except ImportError:
    pass

# Also patch networkx directly if needed
try:
    import networkx.algorithms.dag
    if hasattr(networkx.algorithms.dag, 'fractions'):
        networkx.algorithms.dag.fractions.gcd = math.gcd
except (ImportError, AttributeError):
    pass