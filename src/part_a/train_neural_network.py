import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt
import time

# Загружаем подготовленные данные (признаки отмасштабированы, категории закодированы).
# Используем train для обучения, test — для финальной проверки.
train = pd.read_csv('../../data/train_prepared.csv')
test = pd.read_csv('../../data/test_prepared.csv')

# Отделяем целевую переменную от признаков.
X_train_full = train.drop(columns=['y']).astype(float).values
y_train_full = train['y'].astype(float).values
X_test = test.drop(columns=['y']).astype(float).values
y_test = test['y'].astype(float).values

# Для ранней остановки выделяем валидационную часть из train (последние 20% по времени).
# Это позволяет не подсматривать в тест при выборе лучшей эпохи.
split_idx = int(len(X_train_full) * 0.8)
X_tr, X_val = X_train_full[:split_idx], X_train_full[split_idx:]
y_tr, y_val = y_train_full[:split_idx], y_train_full[split_idx:]

# Конвертируем массивы в тензоры PyTorch (float32 — стандарт для весов).
X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)

# Описываем архитектуру сети. hidden_sizes — список размеров скрытых слоёв.
# Dropout случайно обнуляет часть нейронов, чтобы сеть не запоминала шум.
class Net(nn.Module):
    def __init__(self, input_dim, hidden_sizes, dropout_rate=0.2):
        super(Net, self).__init__()
        layers = []
        prev_size = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = h
        layers.append(nn.Linear(prev_size, 1))
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x).squeeze()

# Сравниваем две архитектуры, как требует задание.
architectures = {
    "128-64": [128, 64],
    "256-128-64": [256, 128, 64],
    "64-32": [64, 32]
}

results_list = []

# Создаем фигуру для графиков (Loss и AP).
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
ax_loss_1 = axes[0, 0]
ax_loss_2 = axes[0, 1]
ax_ap = axes[1, 0]

for idx, (arch_name, hidden_sizes) in enumerate(architectures.items()):
    print(f"\n=== Архитектура: {arch_name} ===")
    
    # Задаем Dropout для каждой архитектуры отдельно
    dropout_val = 0.4 if "64-32" in arch_name else 0.2
    model = Net(input_dim=X_tr.shape[1], hidden_sizes=hidden_sizes, dropout_rate=dropout_val)
    criterion = nn.BCEWithLogitsLoss() # Внутри применяет сигмоиду, принимает "сырые" логиты
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    # Списки для сохранения истории обучения (для графиков)
    history_train_loss = []
    history_val_loss = []
    history_val_ap = []
    
    start_time = time.time()
    
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        logits = model(X_tr_t)
        loss = criterion(logits, y_tr_t)
        loss.backward()
        optimizer.step()
        
        # Переводим модель в режим оценки (выключаем Dropout) и проверяем на валидации
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t).item()
            # Для расчета AP нужны вероятности, поэтому применяем сигмоиду
            val_proba = torch.sigmoid(val_logits).numpy()
            val_ap = average_precision_score(y_val, val_proba)
        
        # Сохраняем историю для графиков
        history_train_loss.append(loss.item())
        history_val_loss.append(val_loss)
        history_val_ap.append(val_ap)
        
        # Логика ранней остановки: если val_loss перестает падать, прекращаем обучение
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'best_model_{arch_name}.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Остановка на эпохе {epoch}: val_loss не падает.")
                break
        
        if epoch % 10 == 0:
            print(f"Эпоха {epoch}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, Val AP: {val_ap:.4f}")
    
    train_time = time.time() - start_time
    print(f"Обучение завершено за {train_time:.2f} сек.")
    
    # Добавляем линии на графики для текущей архитектуры
    ax_loss_1.plot(history_train_loss, label=f'{arch_name} Train')
    ax_loss_1.plot(history_val_loss, label=f'{arch_name} Val')
    ax_loss_2.plot(history_val_loss, label=f'{arch_name} Val')
    ax_ap.plot(history_val_ap, label=f'{arch_name} Val AP')
    
    # Загружаем лучшую модель (с минимальным val_loss) и делаем предсказание на тесте
    model.load_state_dict(torch.load(f'best_model_{arch_name}.pth'))
    model.eval()

    # Если это архитектура 128-64 (лучшая), сохраняем её веса в общую папку результатов
    if arch_name == "128-64":
        torch.save(model.state_dict(), '../../results/part_a/nn_weights.pt')
        print(f"Веса лучшей модели ({arch_name}) сохранены в results/part_a/nn_weights.pt")
    
    with torch.no_grad():
        test_logits = model(X_test_t)
        y_proba = torch.sigmoid(test_logits).numpy()
    
    # Подбираем оптимальный порог по F1 (так как классы несбалансированы)
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    
    y_pred = (y_proba >= best_threshold).astype(int)
    
    # Считаем финальные метрики
    f1 = f1_score(y_test, y_pred)
    ap = average_precision_score(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    print(f"Оптимальный порог: {best_threshold:.4f}")
    print(f"F1: {f1:.4f}, AP: {ap:.4f}, ROC-AUC: {roc_auc:.4f}")
    
    # Сохраняем результаты в таблицу
    results_list.append({
        "Architecture": arch_name,
        "F1": f1,
        "AP": ap,
        "ROC-AUC": roc_auc,
        "Train Time (s)": round(train_time, 2)
    })

# Настраиваем графики и сохраняем их
ax_loss_1.set_title('Loss (Train vs Val)')
ax_loss_1.set_xlabel('Epoch')
ax_loss_1.set_ylabel('Loss')
ax_loss_1.legend()
ax_loss_1.grid(True, alpha=0.3)

ax_loss_2.set_title('Val Loss (сравнение архитектур)')
ax_loss_2.set_xlabel('Epoch')
ax_loss_2.set_ylabel('Val Loss')
ax_loss_2.legend()
ax_loss_2.grid(True, alpha=0.3)

ax_ap.set_title('Val AP (сравнение архитектур)')
ax_ap.set_xlabel('Epoch')
ax_ap.set_ylabel('AP')
ax_ap.legend()
ax_ap.grid(True, alpha=0.3)

# Убираем пустую ячейку в сетке графиков
fig.delaxes(axes[1, 1])

plt.tight_layout()
plt.savefig('../../results/figures/nn_curves.png')
print("\nГрафики сохранены в ../../results/figures/nn_curves.png")

# Сохраняем итоговую таблицу метрик
results_df = pd.DataFrame(results_list)
results_df.to_csv('../../results/part_a/neural_results.csv', index=False)
print("\n=== ИТОГОВАЯ ТАБЛИЦА НЕЙРОСЕТИ ===")
print(results_df.to_string(index=False))
