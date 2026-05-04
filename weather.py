import pandas as pd
import numpy as np
from math import log2

# Dataset
data = [
["Sunny","Hot","High","Weak","No"],
["Sunny","Hot","High","Strong","No"],
["Overcast","Hot","High","Weak","Yes"],
["Rain","Mild","High","Weak","Yes"],
["Rain","Cool","Normal","Weak","Yes"],
["Rain","Cool","Normal","Strong","No"],
["Overcast","Cool","Normal","Strong","Yes"],
["Sunny","Mild","High","Weak","No"],
["Sunny","Cool","Normal","Weak","Yes"],
["Rain","Mild","Normal","Weak","Yes"],
["Sunny","Mild","Normal","Strong","Yes"],
["Overcast","Mild","High","Strong","Yes"],
["Overcast","Hot","Normal","Weak","Yes"],
["Rain","Mild","High","Strong","No"]
]

df = pd.DataFrame(data, columns=["Outlook","Temperature","Humidity","Wind","Play"])

# Entropy function
def entropy(target):
    values, counts = np.unique(target, return_counts=True)
    probs = counts / counts.sum()
    return -sum(p * log2(p) for p in probs)

# Information Gain function
def information_gain(df, feature, target="Play"):
    total_entropy = entropy(df[target])
    
    values, counts = np.unique(df[feature], return_counts=True)
    
    weighted_entropy = sum(
        (counts[i] / sum(counts)) *
        entropy(df[df[feature] == values[i]][target])
        for i in range(len(values))
    )
    
    return total_entropy - weighted_entropy

# Compute for each feature
for feature in ["Outlook", "Temperature", "Humidity", "Wind"]:
    print(f"{feature}: {round(information_gain(df, feature), 3)}")

