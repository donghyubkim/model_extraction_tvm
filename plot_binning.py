import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times'], 'size': 14})
rc('figure', **{'figsize': (6, 4)})


df1 = pd.read_csv('profiling_result/googlenet_1.csv')
df2 = pd.read_csv('profiling_result/vgg16.csv')
df3 = pd.read_csv('profiling_result/resnet18_1.csv')

# count how many durations are larger than 4000 and how many are lower than 4000
count_df1 = df1['Duration (us)'].apply(lambda x: x > 4000).value_counts()
count_df2 = df2['Duration (us)'].apply(lambda x: x > 4000).value_counts()
count_df3 = df3['Duration (us)'].apply(lambda x: x > 4000).value_counts()

# create bar graph
fig, (ax1, ax2) = plt.subplots(ncols=2, sharex=True)
ax1.bar(['googlenet', 'vgg16', 'resnet18'], [count_df1[True], count_df2[True], count_df3[True]], color=['red', 'green', 'blue'])
ax2.bar(['googlenet', 'vgg16', 'resnet18'], [count_df1[False], count_df2[False], count_df3[False]], color=['red', 'green', 'blue'])
ax1.set_ylabel('Count')
ax1.set_xlabel('Duration ≥ 4000us')
ax2.set_xlabel('Duration < 4000us')

plt.tight_layout()
plt.savefig("binned_3models.pdf",dpi = 600)