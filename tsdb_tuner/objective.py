from __future__ import annotations

from typing import Any


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def score_summary(summary: dict[str, Any], objective_params: dict[str, Any]) -> float:
    """Скоринговая функция для одного эксперимента.

    Поддерживает два режима:
    - simple: qps_weight * QPS - p95_weight * p95 - p99_weight * p99;
    - normalized: если в summary уже переданы нормализованные поля, используется
      формула из специального раздела: wQ*Q + w95*L95 + w99*L99 + w50*L50.
    """
    if all(k in summary for k in ("q_norm", "l95_norm", "l99_norm", "l50_norm")):
        return (
            safe_float(objective_params.get("w_q"), 0.30) * safe_float(summary.get("q_norm"))
            + safe_float(objective_params.get("w95"), 0.30) * safe_float(summary.get("l95_norm"))
            + safe_float(objective_params.get("w99"), 0.20) * safe_float(summary.get("l99_norm"))
            + safe_float(objective_params.get("w50"), 0.20) * safe_float(summary.get("l50_norm"))
        )

    qps_weight = safe_float(objective_params.get("qps_weight"), 1.0)
    p95_weight = safe_float(objective_params.get("p95_weight"), 0.10)
    p99_weight = safe_float(objective_params.get("p99_weight"), 0.02)
    p50_weight = safe_float(objective_params.get("p50_weight"), 0.00)

    qps = safe_float(summary.get("avg_rate_qps"))
    p95 = safe_float(summary.get("p95_q95_ms"), safe_float(summary.get("avg_q95_ms")))
    p99 = safe_float(summary.get("p99_q99_ms"), safe_float(summary.get("avg_q99_ms")))
    p50 = safe_float(summary.get("median_q50_ms"))
    return qps_weight * qps - p95_weight * p95 - p99_weight * p99 - p50_weight * p50


def add_normalized_scores(rows: list[dict[str, Any]], objective_params: dict[str, Any]) -> list[dict[str, Any]]:
    """Добавляет score по формуле диплома с min-max нормализацией по истории экспериментов."""
    if not rows:
        return []

    q_values = [safe_float(r.get("avg_rate_qps")) for r in rows]
    l50_values = [safe_float(r.get("median_q50_ms")) for r in rows]
    l95_values = [safe_float(r.get("p95_q95_ms"), safe_float(r.get("avg_q95_ms"))) for r in rows]
    l99_values = [safe_float(r.get("p99_q99_ms"), safe_float(r.get("avg_q99_ms"))) for r in rows]

    def norm_more_better(v: float, values: list[float]) -> float:
        lo, hi = min(values), max(values)
        if hi == lo:
            return 1.0
        return (v - lo) / (hi - lo)

    def norm_less_better(v: float, values: list[float]) -> float:
        lo, hi = min(values), max(values)
        if hi == lo:
            return 1.0
        return 1.0 - (v - lo) / (hi - lo)

    out: list[dict[str, Any]] = []
    for row in rows:
        r = dict(row)
        r["q_norm"] = norm_more_better(safe_float(r.get("avg_rate_qps")), q_values)
        r["l50_norm"] = norm_less_better(safe_float(r.get("median_q50_ms")), l50_values)
        r["l95_norm"] = norm_less_better(safe_float(r.get("p95_q95_ms"), safe_float(r.get("avg_q95_ms"))), l95_values)
        r["l99_norm"] = norm_less_better(safe_float(r.get("p99_q99_ms"), safe_float(r.get("avg_q99_ms"))), l99_values)
        r["score"] = score_summary(r, objective_params)
        out.append(r)
    return out