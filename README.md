# Лабораторная работа: Машинное обучение (Bank Marketing & Online Retail)

## Структура проекта
- `data/` — исходные данные (не включены в репозиторий из-за размера).
- `src/part_a/` — код для задачи классификации (Bank Marketing).
- `src/part_b/` — код для задачи кластеризации (Online Retail).
- `cpp/` — собственная реализация LogReg и K-Means на C++ (Armadillo).
- `results/part_a/` — метрики и таблицы части А.
- `results/part_b/` — результаты кластеризации части Б.
- `results/figures/` — все графики и визуализации.

## Установка и зависимости
Для Python:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn torch
```
Для C++ (Ubuntu):
```bash
sudo apt install libarmadillo-dev cmake g++
```

## Порядок запуска (Часть А)
```bash
python3 src/part_a/data_loader.py — загрузка данных
python3 src/part_a/data_audit.py — аудит данных
python3 src/part_a/eda_stats.py — статистика и графики
python3 src/part_a/data_preparation.py — подготовка признаков
python3 src/part_a/baseline_logreg.py — библиотечная LogReg
python3 src/part_a/train_classic_models.py — 8 моделей
python3 src/part_a/train_svm.py — SVM 
python3 src/part_a/train_neural_network.py — нейросеть
python3 src/part_a/own_logreg.py — своя LogReg
python3 src/part_a/compare_implementations.py — сравнение
```
## Порядок запуска (Часть Б)
```bash
python3 src/part_b/prepare_retail_data.py — RFM-агрегация
python3 src/part_b/remove_outliers.py — вынос оптовиков
python3 src/part_b/clustering_kmeans.py — K-Means
python3 src/part_b/clustering_dbscan.py — DBSCAN
python3 src/part_b/clustering_gmm_agglo.py — GMM и Agglomerative
python3 src/part_b/dimensionality_reduction.py — PCA и t-SNE
python3 src/part_b/stability_analysis.py
```
## C++ реализация
```bash
cd cpp
mkdir build && cd build
cmake ..
make
./kmeans
```