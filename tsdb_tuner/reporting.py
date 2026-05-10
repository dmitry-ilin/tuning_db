"""
reporting.py — форматированный вывод результатов оптимизации в консоль.

Показывает:
  1. Базовые метрики (первый эксперимент в рамках сессии random-search)
  2. Прогресс по поколениям ГА (Q50, Q99, QPS, score)
  3. Итоговое сравнение: начальное → конечное с Δ и % улучшения
  4. Метрики нагрузки на контейнеры (CPU, RAM, IO)
  5. Внутренние метрики СУБД (cache hit ratio, bgwriter, соединения)
"""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.text import Text

console = Console()


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _safe(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _fmt_ms(ms: float) -> str:
    if ms == 0:
        return "—"
    return f"{ms:.1f} ms"


def _fmt_qps(q: float) -> str:
    if q == 0:
        return "—"
    return f"{q:.1f}"


def _delta_style(delta: float, lower_is_better: bool = False) -> tuple[str, str]:
    """Возвращает (formatted_delta, rich_style)."""
    if delta == 0:
        return "±0.0%", "dim"
    pct = delta  # уже в %
    if lower_is_better:
        style = "bold green" if delta < 0 else "bold red"
    else:
        style = "bold green" if delta > 0 else "bold red"
    sign = "+" if delta > 0 else ""
    return f"{sign}{abs(pct):.1f}%", style


def _pct_change(old: float, new: float) -> float:
    """Процент изменения (new-old)/old*100. 0 если old=0."""
    if old == 0:
        return 0.0
    return (new - old) / old * 100.0


# ---------------------------------------------------------------------------
# Таблица: прогресс оптимизации по поколениям
# ---------------------------------------------------------------------------

def print_generation_progress(
    generation_metrics: list[dict[str, Any]],
    title: str = "Прогресс оптимизации (по поколениям ГА)",
) -> None:
    """
    generation_metrics: список dict с полями
        generation, best_score, best_qps, best_q50_ms, best_q99_ms
    """
    if not generation_metrics:
        return

    table = Table(title=title, box=box.ROUNDED, show_lines=True)
    table.add_column("Поколение", justify="center", style="cyan", no_wrap=True)
    table.add_column("QPS", justify="right")
    table.add_column("Q50 (ms)", justify="right")
    table.add_column("Q99 (ms)", justify="right")
    table.add_column("Score", justify="right", style="bold")
    table.add_column("Δ Score", justify="right")

    prev_score: float | None = None
    for row in generation_metrics:
        gen = str(row.get("generation", "?"))
        qps = _fmt_qps(_safe(row.get("best_qps")))
        q50 = _fmt_ms(_safe(row.get("best_q50_ms")))
        q99 = _fmt_ms(_safe(row.get("best_q99_ms")))
        score = _safe(row.get("best_score"))
        score_str = f"{score:.4f}"

        if prev_score is not None and prev_score != 0:
            delta = score - prev_score
            delta_str = f"{delta:+.4f}"
            delta_style = "green" if delta > 0 else ("red" if delta < 0 else "dim")
        else:
            delta_str = "—"
            delta_style = "dim"

        table.add_row(gen, qps, q50, q99, score_str, Text(delta_str, style=delta_style))
        prev_score = score

    console.print(table)


# ---------------------------------------------------------------------------
# Таблица: итоговое сравнение до/после
# ---------------------------------------------------------------------------

def print_before_after_comparison(
    baseline: dict[str, Any],
    final: dict[str, Any],
    label_baseline: str = "Начальная (LHS baseline)",
    label_final: str = "Финальная (ГА + градиент)",
) -> None:
    """
    Печатает сравнительную таблицу ключевых метрик.
    baseline / final — dict из v_experiment_summary.
    """
    metrics = [
        # (label, key_in_summary, lower_is_better)
        ("QPS (запросов/сек)", "avg_rate_qps", False),
        ("Q50 / Median (ms)", "median_q50_ms", True),
        ("Q95 (ms)", "p95_q95_ms", True),
        ("Q99 (ms)", "p99_q99_ms", True),
        ("Score (целевая ф-ция)", "score", False),
    ]

    table = Table(
        title="📊  Сравнение: начальная конфигурация → оптимальная конфигурация",
        box=box.ROUNDED,
        show_lines=True,
    )
    table.add_column("Метрика", style="bold white", min_width=24)
    table.add_column(label_baseline, justify="right", style="dim")
    table.add_column(label_final, justify="right")
    table.add_column("Изменение", justify="right")
    table.add_column("Оценка", justify="center")

    for label, key, lower_is_better in metrics:
        old_val = _safe(baseline.get(key))
        new_val = _safe(final.get(key))

        old_str = (_fmt_ms(old_val) if "ms" in label else _fmt_qps(old_val)) if key != "score" else f"{old_val:.4f}"
        new_str = (_fmt_ms(new_val) if "ms" in label else _fmt_qps(new_val)) if key != "score" else f"{new_val:.4f}"

        pct = _pct_change(old_val, new_val)
        delta_str, style = _delta_style(pct, lower_is_better=lower_is_better)

        # Эмодзи-оценка
        if lower_is_better:
            emoji = "✅" if pct < -5 else ("⚠️" if pct > 5 else "➡️")
        else:
            emoji = "✅" if pct > 5 else ("⚠️" if pct < -5 else "➡️")

        table.add_row(label, old_str, new_str, Text(delta_str, style=style), emoji)

    console.print(table)


# ---------------------------------------------------------------------------
# Таблица: метрики контейнеров
# ---------------------------------------------------------------------------

def print_container_stats(
    baseline_stats: dict[str, Any],
    final_stats: dict[str, Any],
) -> None:
    """
    baseline_stats / final_stats — dict[container_name, ContainerStats].
    """
    all_names = sorted(set(list(baseline_stats.keys()) + list(final_stats.keys())))
    if not all_names:
        return

    table = Table(
        title="🐳  Нагрузка на контейнеры (до → после оптимизации)",
        box=box.ROUNDED,
        show_lines=True,
    )
    table.add_column("Контейнер", style="cyan")
    table.add_column("CPU avg %\n(до / после)", justify="right")
    table.add_column("CPU max %\n(до / после)", justify="right")
    table.add_column("RAM avg MiB\n(до / после)", justify="right")
    table.add_column("RAM max MiB\n(до / после)", justify="right")
    table.add_column("Disk R MiB\n(до / после)", justify="right")
    table.add_column("Disk W MiB\n(до / после)", justify="right")
    table.add_column("Net RX MiB\n(до / после)", justify="right")

    for name in all_names:
        b = baseline_stats.get(name)
        f = final_stats.get(name)

        def fmt_pair(bv: float | None, fv: float | None, decimals: int = 1) -> str:
            bs = f"{bv:.{decimals}f}" if bv is not None else "—"
            fs = f"{fv:.{decimals}f}" if fv is not None else "—"
            return f"{bs} / {fs}"

        def get(stats: Any, attr: str) -> float | None:
            if stats is None:
                return None
            v = getattr(stats, attr, None)
            return float(v) if v is not None else None

        table.add_row(
            name,
            fmt_pair(get(b, "cpu_pct_avg"), get(f, "cpu_pct_avg")),
            fmt_pair(get(b, "cpu_pct_max"), get(f, "cpu_pct_max")),
            fmt_pair(get(b, "mem_used_mb_avg"), get(f, "mem_used_mb_avg"), 0),
            fmt_pair(get(b, "mem_used_mb_max"), get(f, "mem_used_mb_max"), 0),
            fmt_pair(get(b, "blk_read_delta_mb"), get(f, "blk_read_delta_mb")),
            fmt_pair(get(b, "blk_write_delta_mb"), get(f, "blk_write_delta_mb")),
            fmt_pair(get(b, "net_rx_delta_mb"), get(f, "net_rx_delta_mb")),
        )

    console.print(table)


# ---------------------------------------------------------------------------
# Таблица: внутренние метрики СУБД
# ---------------------------------------------------------------------------

def print_pg_stats_comparison(
    baseline_pg: dict[str, Any],
    final_pg: dict[str, Any],
) -> None:
    if not baseline_pg and not final_pg:
        return

    table = Table(
        title="🗄️  Внутренние метрики TimescaleDB/PostgreSQL (до → после)",
        box=box.ROUNDED,
        show_lines=True,
    )
    table.add_column("Метрика", style="bold white", min_width=30)
    table.add_column("До оптимизации", justify="right", style="dim")
    table.add_column("После оптимизации", justify="right")

    def nested(d: dict, *keys: str) -> Any:
        for k in keys:
            if not isinstance(d, dict):
                return None
            d = d.get(k)  # type: ignore[assignment]
        return d

    rows_def = [
        ("Cache hit ratio", ("db", "cache_hit_ratio"), "{:.4f}"),
        ("Блоки: прочитано (blks_read)", ("db", "blks_read"), "{:.0f}"),
        ("Блоки: попало в кеш (blks_hit)", ("db", "blks_hit"), "{:.0f}"),
        ("Транзакции commit", ("db", "xact_commit"), "{:.0f}"),
        ("Транзакции rollback", ("db", "xact_rollback"), "{:.0f}"),
        ("Deadlocks", ("db", "deadlocks"), "{:.0f}"),
        ("Temp files", ("db", "temp_files"), "{:.0f}"),
        ("Temp bytes (bytes)", ("db", "temp_bytes"), "{:.0f}"),
        ("Bgwriter: checkpoints_req", ("bgwriter", "checkpoints_req"), "{:.0f}"),
        ("Bgwriter: buffers_backend", ("bgwriter", "buffers_backend"), "{:.0f}"),
        ("Bgwriter: buffers_alloc", ("bgwriter", "buffers_alloc"), "{:.0f}"),
        ("Активных соединений", ("connections", "active"), "{:.0f}"),
        ("Размер БД (MiB)", ("db_size_mb",), "{:.1f}"),
    ]

    for label, key_path, fmt in rows_def:
        if len(key_path) == 1:
            bv = baseline_pg.get(key_path[0])
            fv = final_pg.get(key_path[0])
        else:
            bv = nested(baseline_pg, *key_path)
            fv = nested(final_pg, *key_path)

        b_str = fmt.format(float(bv)) if bv is not None else "—"
        f_str = fmt.format(float(fv)) if fv is not None else "—"
        table.add_row(label, b_str, f_str)

    console.print(table)


# ---------------------------------------------------------------------------
# Таблица: лучшие конфигурации найденные за всю сессию
# ---------------------------------------------------------------------------

def print_best_configs_summary(
    trials: list[dict[str, Any]],
    top_n: int = 5,
) -> None:
    """trials — список dict из optimization_trials, отсортированных по score DESC."""
    if not trials:
        return

    table = Table(
        title=f"🏆  Топ-{top_n} конфигураций за сессию ГА",
        box=box.ROUNDED,
        show_lines=True,
    )
    table.add_column("#", justify="center", style="bold cyan")
    table.add_column("Config ID", justify="center")
    table.add_column("Поколение", justify="center")
    table.add_column("Score", justify="right", style="bold green")
    table.add_column("QPS", justify="right")
    table.add_column("Q50 (ms)", justify="right")
    table.add_column("Q99 (ms)", justify="right")
    table.add_column("Stage", justify="center", style="dim")

    for i, trial in enumerate(trials[:top_n], start=1):
        metrics = trial.get("metrics") or {}
        if isinstance(metrics, str):
            import json
            try:
                metrics = json.loads(metrics)
            except Exception:
                metrics = {}
        table.add_row(
            str(i),
            str(trial.get("config_id", "?")),
            str(trial.get("generation", "?")),
            f"{_safe(trial.get('score')):.4f}",
            _fmt_qps(_safe(metrics.get("avg_rate_qps"))),
            _fmt_ms(_safe(metrics.get("median_q50_ms"))),
            _fmt_ms(_safe(metrics.get("p99_q99_ms", metrics.get("avg_q99_ms")))),
            str(trial.get("stage") or trial.get("status") or "ga"),
        )

    console.print(table)


# ---------------------------------------------------------------------------
# Итоговый баннер
# ---------------------------------------------------------------------------

def print_optimization_summary_banner(
    baseline_score: float,
    final_score: float,
    total_experiments: int,
    elapsed_sec: float,
    top_params: list[str],
) -> None:
    pct = _pct_change(baseline_score, final_score)
    sign = "+" if pct >= 0 else ""
    color = "green" if pct >= 0 else "red"

    lines = [
        f"[bold]Score:[/bold]  {baseline_score:.4f}  →  [bold {color}]{final_score:.4f}[/bold {color}]"
        f"  ([{color}]{sign}{abs(pct):.1f}%[/{color}])",
        f"[bold]Экспериментов:[/bold] {total_experiments}",
        f"[bold]Время:[/bold]  {elapsed_sec / 60:.1f} мин",
        "",
        "[bold]Оптимизировавшиеся параметры:[/bold]",
        "  " + ", ".join(top_params[:10]),
    ]

    panel = Panel(
        "\n".join(lines),
        title="[bold white]✨  Оптимизация завершена[/bold white]",
        border_style="green" if pct >= 0 else "red",
        expand=False,
    )
    console.print(panel)
