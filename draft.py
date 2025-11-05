"""
Which labelling scheme is more
challenging to predict from the given data:
target_human or target_asv
"""

import pandas as pd

orig = pd.read_csv("system_Original.csv")
print(orig.columns)
