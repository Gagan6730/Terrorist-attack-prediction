import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Read dataset
dataset = pd.read_csv('dataset.csv')
X=dataset['region_txt'].value_counts()
print(X)