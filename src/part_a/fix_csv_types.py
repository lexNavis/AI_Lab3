import pandas as pd

for name in ['train_prepared', 'test_prepared']:
    path = f'../../data/{name}.csv'
    df = pd.read_csv(path)
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].replace({'True': 1, 'False': 0, 'true': 1, 'false': 0})
    df = df.astype(float)
    df.to_csv(path, index=False)
    print(f'{name} fixed')
    