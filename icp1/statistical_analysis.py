import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Data/Data.csv')
df = df.drop("id", axis=1)
target_dict = {
    "M": 0,
    "B": 1
}
df["diagnosis"] = df["diagnosis"].apply(lambda x: target_dict[x])

mean_df = df.groupby(["diagnosis"]).mean().reset_index()
m_df = mean_df.iloc[0]
b_df = mean_df.iloc[1]
index = mean_df.columns
plot_df = pd.DataFrame({'malignant': m_df,
                        'benign': b_df}, index=index)
ax = plot_df.plot.bar(rot=90)
plt.savefig("plots/mean_graph.png",bbox_inches='tight', dpi=960)
plt.show()

