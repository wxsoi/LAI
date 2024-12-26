#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("./data/political_leaning_en.csv")
# df.rename(columns={'auhtor_ID': 'author_ID'}, inplace=True)
df.info()


print(df['political_leaning'].value_counts(normalize=True))


post_counts = df.groupby('author_ID').size().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
post_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Number of Posts per User')
plt.xlabel('Author ID')
plt.ylabel('Number of Posts')
plt.xticks(rotation=0)
plt.show()

percentile_95 = np.percentile(post_counts, 95)
print(f"95th Percentile of Posts per User: {percentile_95}")

print(f"Median of Posts per User: {np.median(post_counts)}")

print(f"Mean of Posts per User: {np.mean(post_counts)}")

# Resample users with more than 95th percentile posts
max_posts = int(percentile_95)
df_resampled = df.groupby('author_ID').apply(
    lambda x: x.sample(n=max_posts, random_state=42) if len(x) > max_posts else x
).reset_index(drop=True)

print(f"New dataset size: {len(df_resampled)}")


df_resampled.head()


grouped = df_resampled.groupby('political_leaning')
min_size = min(grouped.size())

# Sample equal number of posts from each group (33% each)
balanced_df = grouped.apply(lambda x: x.sample(n=min_size, random_state=42)).reset_index(drop=True)

# Check distribution
print(balanced_df['political_leaning'].value_counts(normalize=True))
print(f"Balanced Dataset Size: {len(balanced_df)}")

pd.DataFrame(balanced_df).to_csv('.data/political_leaning_sample_en.csv')

# dfs = np.array_split(balanced_df, 14)
# counter = 1
# for df in dfs:
#     pd.DataFrame(df).to_csv(f'df_part_{counter}.csv')
#     counter += 1

