import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Загружаем данные (с метками K-Means)
rfm = pd.read_csv('../../results/part_b/rfm_clustered.csv', index_col=0)
X = rfm[['Recency', 'Frequency', 'Monetary', 'UniqueProducts', 'AvgBasket']].copy()
X = np.log1p(X)
X_scaled = StandardScaler().fit_transform(X)
labels = rfm['Cluster'].values

# --- PCA ---
print("=== PCA ===")
pca = PCA()
pca.fit(X_scaled)

# График объяснённой дисперсии
plt.figure(figsize=(8, 4))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.axhline(y=0.90, color='r', linestyle='--', label='90% дисперсии')
plt.axhline(y=0.95, color='g', linestyle='--', label='95% дисперсии')
plt.xlabel('Число компонент')
plt.ylabel('Накопленная объяснённая дисперсия')
plt.title('PCA: выбор числа компонент')
plt.legend()
plt.grid(True)
plt.savefig('../../results/figures/pca_variance.png')
print("График сохранён в results/figures/pca_variance.png")

# Находим минимальное число компонент для 90% и 95%
cumvar = np.cumsum(pca.explained_variance_ratio_)
n_90 = np.argmax(cumvar >= 0.90) + 1
n_95 = np.argmax(cumvar >= 0.95) + 1
print(f"Компонент для 90% дисперсии: {n_90}")
print(f"Компонент для 95% дисперсии: {n_95}")

# Ошибка восстановления для разного числа компонент
for n in range(1, X_scaled.shape[1] + 1):
    pca_n = PCA(n_components=n)
    X_reduced = pca_n.fit_transform(X_scaled)
    X_reconstructed = pca_n.inverse_transform(X_reduced)
    mse = np.mean((X_scaled - X_reconstructed) ** 2)
    print(f"n_components={n}: MSE восстановления = {mse:.6f}")

# --- t-SNE ---
print("\n=== t-SNE ===")
for perplexity in [5, 30]:
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='pca')
    X_tsne = tsne.fit_transform(X_scaled)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis', s=10, alpha=0.7)
    plt.colorbar(scatter, label='Кластер')
    plt.title(f't-SNE (perplexity={perplexity})')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.savefig(f'../../results/figures/tsne_perplexity_{perplexity}.png')
    print(f"График t-SNE (perplexity={perplexity}) сохранён в results/figures/tsne_perplexity_{perplexity}.png")
