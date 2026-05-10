from __future__ import annotations

import random
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

from .analyzer import summaries_to_frame
from .lhs import latin_hypercube_configs
from .objective import add_normalized_scores, score_summary
from .params import ParameterSpec, random_config, repair_config


def choose_initial_population_by_surrogate(
    summaries: list[dict[str, Any]],
    specs: list[ParameterSpec],
    objective_params: dict[str, Any],
    rng: random.Random,
    candidates: int = 1000,
    top_k: int = 12,
    only_params: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Первый этап: RF-суррогат + LHS-кандидаты + отбор top-K стартовых конфигураций."""
    top_k = max(1, int(top_k))
    selected = set(only_params) if only_params else None
    fallback = latin_hypercube_configs(specs, rng, top_k, only_params=only_params)

    if len(summaries) < 5:
        return fallback

    df = summaries_to_frame(summaries, specs)
    feature_cols = [s.name for s in specs if s.name in df.columns and (selected is None or s.name in selected)]
    if not feature_cols:
        return fallback

    norm_rows = add_normalized_scores(df.to_dict("records"), objective_params)
    y_arr = np.asarray([score_summary(row, objective_params) for row in norm_rows], dtype=float)
    if len(y_arr) < 5 or len(set(np.round(y_arr, 6))) <= 1:
        return fallback

    model = RandomForestRegressor(n_estimators=300, random_state=rng.randint(1, 10_000), n_jobs=-1)
    model.fit(df[feature_cols].fillna(0.0), y_arr)

    candidate_cfgs = latin_hypercube_configs(specs, rng, candidates, only_params=only_params)
    scored: list[tuple[float, dict[str, Any]]] = []
    for cfg in candidate_cfgs:
        x = pd.DataFrame(
            [[float(cfg.get(name, 0.0)) for name in feature_cols]],
            columns=feature_cols,
        )
        pred = float(model.predict(x)[0])
        scored.append((pred, repair_config(cfg)))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [cfg for _, cfg in scored[:top_k]]


def choose_initial_population_with_scores(
    summaries: list[dict[str, Any]],
    specs: list[ParameterSpec],
    objective_params: dict[str, Any],
    rng: random.Random,
    candidates: int = 1000,
    top_k: int = 12,
    only_params: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[float], int]:
    """
    То же что choose_initial_population_by_surrogate, но дополнительно возвращает
    (population, predicted_scores, candidates_count) — для отчёта и сохранения в JSON.
    """
    top_k = max(1, int(top_k))
    selected = set(only_params) if only_params else None
    fallback = latin_hypercube_configs(specs, rng, top_k, only_params=only_params)

    if len(summaries) < 5:
        return fallback, [], candidates

    df = summaries_to_frame(summaries, specs)
    feature_cols = [s.name for s in specs if s.name in df.columns and (selected is None or s.name in selected)]
    if not feature_cols:
        return fallback, [], candidates

    norm_rows = add_normalized_scores(df.to_dict("records"), objective_params)
    y_arr = np.asarray([score_summary(row, objective_params) for row in norm_rows], dtype=float)
    if len(y_arr) < 5 or len(set(np.round(y_arr, 6))) <= 1:
        return fallback, [], candidates

    model = RandomForestRegressor(n_estimators=300, random_state=rng.randint(1, 10_000), n_jobs=-1)
    model.fit(df[feature_cols].fillna(0.0), y_arr)

    candidate_cfgs = latin_hypercube_configs(specs, rng, candidates, only_params=only_params)
    scored: list[tuple[float, dict[str, Any]]] = []
    for cfg in candidate_cfgs:
        x = pd.DataFrame(
            [[float(cfg.get(name, 0.0)) for name in feature_cols]],
            columns=feature_cols,
        )
        pred = float(model.predict(x)[0])
        scored.append((pred, repair_config(cfg)))
    scored.sort(key=lambda item: item[0], reverse=True)
    top = scored[:top_k]
    return (
        [cfg for _, cfg in top],
        [round(s, 6) for s, _ in top],
        candidates,
    )


def choose_initial_config_by_surrogate(
    summaries: list[dict[str, Any]],
    specs: list[ParameterSpec],
    objective_params: dict[str, Any],
    rng: random.Random,
    candidates: int = 500,
    only_params: list[str] | None = None,
) -> dict[str, Any]:
    population = choose_initial_population_by_surrogate(
        summaries=summaries,
        specs=specs,
        objective_params=objective_params,
        rng=rng,
        candidates=candidates,
        top_k=1,
        only_params=only_params,
    )
    return population[0] if population else repair_config(random_config(specs, rng, set(only_params) if only_params else None))
