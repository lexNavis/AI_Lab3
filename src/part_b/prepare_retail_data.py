import pandas as pd
import numpy as np

print("Загрузка данных...")
df = pd.read_csv('../../data/online_retail.csv')
total_before = len(df)

# 1. Удаление строк без CustomerID. 
# Без идентификатора клиента невозможно провести агрегацию (сгруппировать покупки по человеку).
mask = df['CustomerID'].isna()
print(f"Удалено без CustomerID: {mask.sum()}")
df = df[~mask]

# 2. Удаление отмен. 
# В этом датасете отмены помечаются префиксом 'C' в номере чека. 
# Нам нужны только подтвержденные покупки для корректного расчета RFM.
mask = df['InvoiceNo'].astype(str).str.startswith('C')
print(f"Удалено отмен (InvoiceNo с 'C'): {mask.sum()}")
df = df[~mask]

# 3. Удаление неположительных Quantity. 
# Отрицательные значения — это возвраты, а нулевые — корректировки. 
# Для оценки объема покупок (Monetary) они не подходят.
mask = df['Quantity'] <= 0
print(f"Удалено Quantity <= 0: {mask.sum()}")
df = df[~mask]

# 4. Удаление неположительных UnitPrice. 
# Нулевые или отрицательные цены — это служебные строки, которые искажают сумму покупки.
mask = df['UnitPrice'] <= 0
print(f"Удалено UnitPrice <= 0: {mask.sum()}")
df = df[~mask]

# 5. Удаление StockCode без цифр. 
# Технические коды (POST, D, M, BANK CHARGES) не являются реальными товарами.
# Если их оставить, они исказят подсчет числа уникальных товаров (UniqueProducts).
mask = ~df['StockCode'].astype(str).str.contains(r'\d')
print(f"Удалено StockCode без цифр: {mask.sum()}")
df = df[~mask]

print(f"Осталось строк после очистки: {len(df)} (из {total_before})")

# Расчет суммарной стоимости каждой позиции в чеке.
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# Определение даты среза. 
# Это день, следующий за последней покупкой в датасете. 
# Нужен для расчета Recency (сколько дней прошло от последней покупки клиента до этой даты).
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

# Агрегация по клиентам (CustomerID). 
# На выходе получаем одну строку на клиента с признаками для RFM-анализа.
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days, # Recency: дней с последней покупки
    'InvoiceNo': 'nunique',                                  # Frequency: число уникальных чеков
    'TotalPrice': 'sum',                                     # Monetary: общая сумма покупок
    'StockCode': 'nunique'                                   # UniqueProducts: число уникальных товаров
})

# Переименование колонок для удобства.
rfm.rename(columns={
    'InvoiceDate': 'Recency',
    'InvoiceNo': 'Frequency',
    'TotalPrice': 'Monetary',
    'StockCode': 'UniqueProducts'
}, inplace=True)

# Добавление признака "средний чек". 
# Рассчитывается как отношение общей суммы к числу заказов.
rfm['AvgBasket'] = rfm['Monetary'] / rfm['Frequency']

# Сохранение агрегированной таблицы для кластеризации.
rfm.to_csv('../../results/part_b/rfm.csv')
print(f"Агрегировано клиентов: {len(rfm)}")
print(rfm.head())
