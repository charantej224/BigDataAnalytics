import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

## Reading the dataframe from the CSV file
df = pd.read_csv('Data/Data.csv')

'''Data Cleaning
Step1: drop the unique values as it doesnt mean of significance when understanding co-relation.
Step : the target values are considered as Malignent or Benign cancer which are replaced as numbers rather than characters.
'''
df = df.drop("id", axis=1)
target_dict = {
    "M": 0,
    "B": 1
}
df["diagnosis"] = df["diagnosis"].apply(lambda x: target_dict[x])

'''
Understanding co-relation 
'''
corr = df.corr()
ax = sns.heatmap(
    corr,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)
plt.savefig("plots/correlation_features.png", bbox_inches='tight')
plt.show()
