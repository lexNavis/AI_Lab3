from sklearn.metrics import adjusted_rand_score
import pandas as pd

# Метки из C++
cpp_labels = pd.read_csv('../../results/part_b/kmeans_cpp.csv')['cluster'].values

# Метки из Python (сохранены в rfm_clustered.csv)
py_labels = pd.read_csv('../../results/part_b/rfm_clustered.csv')['Cluster'].values

ari = adjusted_rand_score(py_labels, cpp_labels)
print(f"ARI between C++ and Python K-Means: {ari:.4f}")
