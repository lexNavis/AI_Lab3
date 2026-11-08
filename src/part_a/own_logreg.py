import pandas as pd
import numpy as np
import joblib

# ==========================================
# 1. Собственная реализация (класс)
# ==========================================
class MyLogisticRegression:
    def __init__(self, C=1.0, lr=0.1, max_iter=2000, tol=1e-6):
        self.C = C
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.w = None
        self.b = None
        self.loss_history = []

    @staticmethod
    def _sigmoid(z):
        # Численно устойчивая сигмоида, чтобы не было переполнения
        out = np.empty_like(z, dtype=np.float64)
        pos = z >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        expz = np.exp(z[~pos])
        out[~pos] = expz / (1.0 + expz)
        return out

    def _loss_and_grads(self, X, y):
        n, d = X.shape
        z = X @ self.w + self.b
        p = self._sigmoid(z)
        eps = 1e-12
        
        # Кросс-энтропия + L2-регуляризация
        loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        loss += 0.5 / self.C * np.sum(self.w ** 2)
        
        # Градиенты
        diff = (p - y) / n
        grad_w = X.T @ diff + self.w / self.C
        grad_b = np.sum(diff)
        return loss, grad_w, grad_b

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        self.loss_history = []
        
        loss, gw, gb = self._loss_and_grads(X, y)
        self.loss_history.append(loss)

        for it in range(self.max_iter):
            # Backtracking: если шаг слишком большой и loss растет, уменьшаем lr
            lr = self.lr
            improved = False
            for _ in range(30):
                w_new = self.w - lr * gw
                b_new = self.b - lr * gb
                
                if not (np.all(np.isfinite(w_new)) and np.isfinite(b_new)):
                    lr *= 0.5
                    continue
                
                # Временно подставляем веса, чтобы проверить loss
                old_w, old_b = self.w, self.b
                self.w, self.b = w_new, b_new
                new_loss, gw_new, gb_new = self._loss_and_grads(X, y)
                self.w, self.b = old_w, old_b
                
                if np.isfinite(new_loss) and new_loss < loss:
                    self.w, self.b = w_new, b_new
                    gw, gb = gw_new, gb_new
                    improved = True
                    break
                lr *= 0.5
                
            if not improved:
                break
                
            prev_loss = loss
            loss = new_loss
            self.loss_history.append(loss)
            if abs(prev_loss - loss) < self.tol:
                break
        return self

    def predict_proba(self, X):
        z = np.asarray(X) @ self.w + self.b
        return self._sigmoid(z)

# ==========================================
# 2. Загрузка данных и обучение
# ==========================================
if __name__ == "__main__":
    print("Загрузка данных...")
    train = pd.read_csv('../../data/train_prepared.csv')
    test = pd.read_csv('../../data/test_prepared.csv')

    X_train = train.drop(columns=['y']).astype(float).values
    y_train = train['y'].astype(float).values
    X_test = test.drop(columns=['y']).astype(float).values
    y_test = test['y'].astype(float).values

    # Добавляем bias (колонку единиц), как в нашем C++ коде
    X_train = np.column_stack([np.ones(X_train.shape[0]), X_train])
    X_test = np.column_stack([np.ones(X_test.shape[0]), X_test])

    print(f"Размер train: {X_train.shape}, test: {X_test.shape}")

    # Обучаем модель
    print("Обучение своей LogisticRegression...")
    model = MyLogisticRegression(C=100.0, lr=0.1, max_iter=5000, tol=1e-6)
    model.fit(X_train, y_train)

    joblib.dump({'w': model.w, 'b': model.b}, '../../results/part_a/model_own_logreg.joblib')
    print("Веса собственной LogReg сохранены в results/part_a/model_own_logreg.joblib")

    # Сохраняем предсказания для сравнения
    y_proba = model.predict_proba(X_test)
    np.savetxt('../../results/part_a/logreg_preds_own.csv', y_proba, delimiter='\n')
    
    print(f"Обучение завершено. Финальный loss: {model.loss_history[-1]:.6f}")
    print(f"Вероятности сохранены в ../../results/part_a/logreg_preds_own.csv")
