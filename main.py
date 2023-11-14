import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set random seeds
np.random.seed(2023)
random_seed = 2023

# Load the data
data_fluid_overload_imputed = pd.read_csv("./data_fluid.overload_imputed.csv")

# Calculate the number of imputed datasets
total_datasets = len(data_fluid_overload_imputed) // 991

# Split the DataFrame into individual datasets
datasets = [data_fluid_overload_imputed.iloc[i*991:(i+1)*991] for i in range(total_datasets)]

# Randomly select 10 datasets
selected_datasets = np.random.choice(datasets, size=10, replace=False)

# Combine the selected datasets into a single DataFrame
data_fluid_overload = pd.concat(selected_datasets)
