import fynesse.assess as assess
import numpy as np
import matplotlib.pyplot as plt

# load pandas dataframe
df = assess.data()

# visualizing timeseries correlations
fig, axes = plt.subplots(4, 2, figsize=(16,16))

# randomly sample from dataframe
n = 8
samples = assess.sample_from(df, n)
sample = df.iloc[samples[0]]

array = np.array([sample['date_of_transfer'], sample['date_of_transfer']])
array - np.min(array)

for count, i in enumerate(samples):
    sample = df.iloc[i]
    ax = axes[count//2][count%2]
    
    dataset = df.loc[assess.bbox((df['latitude'], df['longitude']), (sample['latitude'], sample['longitude']), dist=500)]
    
    assess.plot_correlation(ax, dataset['price'], dataset['date_of_transfer'], regression=True)

plt.show()
