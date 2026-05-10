from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .analyzer import summaries_to_frame
from .objective import add_normalized_scores, score_summary
from .params import ParameterSpec, denormalize_vector, normalize_config, repair_config


@dataclass
class LocalGradientResult:
    config: dict[str, Any]
    predicted_score: float
    steps: list[dict[str, Any]]


class NeuralSurrogate:
    """Нейросетевая суррогатная модель для локального уточнения конфигурации.

    Используется MLPRegressor со слоями 128-64-32. Для градиентного шага берется
    численная оценка градиента по нормализованному вектору параметров. Такой
    вариант не требует PyTorch/TensorFlow и остается легким для дипломного стенда.
    """

    def __init__(self, specs: list[ParameterSpec], top_params: list[str], random_state: int = 42):
        self.specs = specs
        self.top_params = top_params
        self.scaler = StandardScaler()
        self.model = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=700,
            random_state=random_state,
            early_stopping=True,
            n_iter_no_change=30,
        )
        self._trained = False
        self._y_min: float = 0.0
        self._y_max: float = 1.0

    def fit(self, summaries: list[dict[str, Any]], objective_params: dict[str, Any]) -> bool:
        if len(summaries) < 5:  # минимум 5 точек для обучения суррогата
            return False
        df = summaries_to_frame(summaries, self.specs)
        feature_cols = [name for name in self.top_params if name in df.columns]
        if len(feature_cols) < 2:
            return False
        records = df.to_dict("records")
        # Используем нормализованный score [0..1] — иначе суррогат обучается
        # на значениях ~1400 (raw QPS), градиент взрывается и предсказания бессмысленны
        scored_rows = add_normalized_scores(records, objective_params)
        y = np.asarray([r["score"] for r in scored_rows], dtype=float)
        if len(np.unique(np.round(y, 8))) <= 1:
            return False
        x_norm = []
        for row in records:
            x_norm.append(normalize_config(row, self.specs, self.top_params))
        X = np.asarray(x_norm, dtype=float)
        Xs = self.scaler.fit_transform(X)

        # early_stopping требует validation split — при малом числе точек (<25)
        # в validation попадает <2 сэмпла и sklearn бросает UndefinedMetricWarning на R².
        # При n < 25 отключаем early_stopping и полагаемся на max_iter + n_iter_no_change.
        n = len(summaries)
        self.model.set_params(early_stopping=n >= 25)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            self.model.fit(Xs, y)

        self._y_min = float(y.min())
        self._y_max = float(y.max())
        self._trained = True
        return True

    def predict_vector(self, vector: np.ndarray) -> float:
        if not self._trained:
            raise RuntimeError("NeuralSurrogate is not trained")
        vector = np.clip(vector.astype(float), 0.0, 1.0).reshape(1, -1)
        raw = float(self.model.predict(self.scaler.transform(vector))[0])
        # Ограничиваем предсказание диапазоном обучающих данных ±20%,
        # чтобы суррогат не взрывался при экстраполяции за пределы LHS-данных
        lo = float(self._y_min - 0.2 * abs(self._y_min))
        hi = float(self._y_max + 0.2 * abs(self._y_max))
        return float(np.clip(raw, lo, hi))

    def improve(
        self,
        start_config: dict[str, Any],
        learning_rate: float = 0.08,
        steps: int = 12,
        eps: float = 1e-3,
    ) -> LocalGradientResult:
        if not self._trained:
            return LocalGradientResult(start_config, float("nan"), [])
        x = np.asarray(normalize_config(start_config, self.specs, self.top_params), dtype=float)
        history: list[dict[str, Any]] = []
        best_x = x.copy()
        best_pred = self.predict_vector(best_x)
        for step in range(steps):
            grad = np.zeros_like(x)
            for i in range(len(x)):
                xp = x.copy(); xp[i] = min(1.0, xp[i] + eps)
                xm = x.copy(); xm[i] = max(0.0, xm[i] - eps)
                grad[i] = (self.predict_vector(xp) - self.predict_vector(xm)) / max(eps, xp[i] - xm[i])
            norm = float(np.linalg.norm(grad))
            if norm > 0:
                grad = grad / norm
            x = np.clip(x + learning_rate * grad, 0.0, 1.0)
            pred = self.predict_vector(x)
            history.append({"step": step, "predicted_score": pred, "gradient_norm": norm})
            if pred > best_pred:
                best_pred = pred
                best_x = x.copy()
            else:
                learning_rate *= 0.5
        cfg = dict(start_config)
        cfg.update(denormalize_vector(best_x.tolist(), self.specs, self.top_params))
        return LocalGradientResult(repair_config(cfg), best_pred, history)