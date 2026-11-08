import pandas as pd

# Загружаем данные
df = pd.read_csv('../../data/train_prepared.csv')

# Берем только те непрерывные числовые колонки, которые были ДО One-Hot Encoding
# (age, campaign, pdays, previous, emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed, was_contacted_before)
check_cols = [
    'age', 'campaign', 'pdays', 'previous', 
    'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 
    'euribor3m', 'nr.employed', 'was_contacted_before'
]

# Оставляем только те, которые реально есть в датасете
cols_to_check = [c for c in check_cols if c in df.columns]

print("=== Проверка масштабирования числовых признаков ===")
print(df[cols_to_check].agg(['mean', 'std']).T)