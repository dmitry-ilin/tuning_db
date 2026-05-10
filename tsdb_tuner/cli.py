from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .ai_initial import choose_initial_population_by_surrogate, choose_initial_population_with_scores
from .analyzer import fit_importances, union_top_params
from .benchmark import BenchmarkService
from .config_apply import ConfigApplier
from .db import Db
from .lhs import latin_hypercube_configs
from .neural_surrogate import NeuralSurrogate
from .optimizer_ga import GeneticOptimizer
from .params import load_param_space, random_config, repair_config
from .objective import score_summary as _score_summary
from .reporting import (
    print_before_after_comparison,
    print_best_configs_summary,
    print_container_stats,
    print_generation_progress,
    print_optimization_summary_banner,
    print_pg_stats_comparison,
)
from .repository import ResultsRepository
from .settings import load_settings
from .state import save_last_scope, load_last_scope

app = typer.Typer(add_completion=False, help="CLI для автоматизированного подбора параметров TimescaleDB/PostgreSQL")
console = Console()


def build_services(config: str | None):
    settings = load_settings(config)
    specs = load_param_space(settings.param_space_path)
    results_db = Db(settings.results_db_dsn)
    target_db = Db(settings.target_db_dsn)
    repo = ResultsRepository(results_db)
    applier = ConfigApplier(target_db, specs, settings.apply)
    benchmark = BenchmarkService(
        repo, applier, settings.benchmark, settings.objective,
        target_db_dsn=settings.target_db_dsn,
    )
    return settings, specs, repo, benchmark


@app.command("init-db")
def init_db(
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    results_sql: Path = typer.Option(Path("sql/001_extend_results_schema.sql")),
):
    """Безопасно добавить недостающие таблицы и поля без удаления старых данных."""
    settings, specs, repo, _ = build_services(config)
    base = Path(config).resolve().parent.parent if config else Path.cwd()
    if not results_sql.is_absolute():
        results_sql = base / results_sql
    # Основной файл уже содержит DDL таблиц мониторинга и представлений ГА
    Db(settings.results_db_dsn).execute_sql_file(results_sql)
    repo.upsert_parameter_space(specs)
    console.print("[green]OK:[/green] схема БД обновлена (включая таблицы мониторинга и представления ГА).")


@app.command("random-search")
def random_search(
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    count: int = typer.Option(10, "--count"),
    seed: int = typer.Option(42, "--seed"),
    apply_config: bool = typer.Option(True, "--apply/--no-apply"),
    lhs: bool = typer.Option(True, "--lhs/--random"),
):
    """Первичный сбор данных: LHS/random-конфигурации -> применение -> TSBS -> БД."""
    _, specs, repo, benchmark = build_services(config)
    rng = random.Random(seed)
    workload_id = repo.get_or_create_workload(benchmark.benchmark_settings.get("workload_name", "tsbs-devops"), tool="tsbs")
    session_id = repo.create_session("initial-sampling", "random", workload_id, benchmark.objective_settings, metadata={"lhs": lhs})
    configs = latin_hypercube_configs(specs, rng, count) if lhs else [repair_config(random_config(specs, rng)) for _ in range(count)]
    best_score = -float("inf")
    best_config_id = None
    try:
        for i, cfg in enumerate(configs):
            try:
                ev = benchmark.evaluate(cfg, source="lhs" if lhs else "random", stage="initial_sampling", generation=0, candidate_index=i, apply_config=apply_config)
                repo.insert_trial(session_id, 0, i, ev.config_id, ev.experiment_id, ev.run_id, ev.metrics, ev.score, "finished")
                console.print(f"[{i+1}/{count}] config_id={ev.config_id} score={ev.score:.3f}")
                if ev.score > best_score:
                    best_score = ev.score
                    best_config_id = ev.config_id
            except Exception as exc:
                failed_config_id = repo.get_or_create_config(cfg, source="sampling_failed", generation=0, candidate_index=i)
                repo.insert_trial(session_id, 0, i, failed_config_id, None, None, {}, None, "failed", str(exc))
                console.print(f"[red][{i+1}/{count}] failed:[/red] {exc}")
        repo.finish_session(session_id, best_config_id, best_score if best_config_id else None)

        latest = repo.latest_summaries(limit=count)
        experiment_ids = [int(row["experiment_id"]) for row in latest]
        save_last_scope(experiment_ids, count=count)
        print(f"Последняя рабочая выборка: {experiment_ids}")

    except KeyboardInterrupt:
        repo.finish_session(session_id, best_config_id, best_score if best_config_id else None, status="interrupted")
        raise typer.Exit(130)


@app.command("analyze")
def analyze(
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    top_n: int = typer.Option(10, "--top-n"),
    min_runs: int = typer.Option(1, "--min-runs"),
    save: bool = typer.Option(True, "--save/--no-save"),
):
    """RandomForest: ранжирование параметров по QPS, p50, p95, p99."""
    _, specs, repo, _ = build_services(config)
    # summaries = repo.all_summaries(min_runs=min_runs)
    scope_ids = load_last_scope()
    if scope_ids:
        summaries = repo.summaries_by_experiment_ids(scope_ids)
    else:
        summaries = repo.all_summaries(min_runs=min_runs)
    importances = fit_importances(summaries, specs, top_n=top_n)
    if not importances:
        console.print("[yellow]Недостаточно данных для анализа.[/yellow]")
        return
    session_id = repo.create_session("knob-selection", "random", None, {}, metadata={"stage": "analysis"}) if save else None
    for metric, rows in importances.items():
        table = Table(title=f"Топ-{top_n} параметров для {metric}")
        table.add_column("#", justify="right")
        table.add_column("Параметр")
        table.add_column("Важность", justify="right")
        for idx, (name, importance) in enumerate(rows, start=1):
            table.add_row(str(idx), name, f"{importance:.5f}")
        console.print(table)
        if save:
            repo.save_importances(session_id, metric, rows)
    top_params = union_top_params(importances, top_n)
    console.print("[bold]Итоговый набор:[/bold] " + ", ".join(top_params))


@app.command("ai-initial")
def ai_initial(
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    top_n: int = typer.Option(10, "--top-n"),
    candidates: int = typer.Option(1000, "--candidates"),
    seed: int = typer.Option(42, "--seed"),
    evaluate: bool = typer.Option(False, "--evaluate/--no-evaluate"),
    output: Optional[Path] = typer.Option(Path("best_ai_config.json"), "--output"),
):
    """Первый этап: RF-суррогат + LHS-кандидаты -> стартовая популяция для ГА."""
    settings, specs, repo, benchmark = build_services(config)
    rng = random.Random(seed)
    # summaries = repo.all_summaries(min_runs=1)
    scope_ids = load_last_scope()
    if scope_ids:
        summaries = repo.summaries_by_experiment_ids(scope_ids)
    else:
        summaries = repo.all_summaries(min_runs=1)
    importances = fit_importances(summaries, specs, top_n=top_n, random_state=seed)
    top_params = union_top_params(importances, top_n) if importances else [s.name for s in specs[:top_n]]
    population_size = int(settings.optimizer.get("ga_population", 12))
    population, pred_scores, n_candidates = choose_initial_population_with_scores(
        summaries, specs, settings.objective, rng, candidates, population_size, top_params
    )
    payload = {
        "best_config": population[0],
        "initial_population": population,
        "top_params": top_params,
        "predicted_scores": pred_scores,
        "candidates_evaluated": n_candidates,
    }
    if output:
        output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        console.print(f"[green]Сохранено:[/green] {output}")

    # Сохраняем RF-суррогат в таблицу surrogate_models
    try:
        from .ai_initial import choose_initial_population_with_scores as _cwps
        _rf_score = getattr(_cwps, "_last_rf_train_score", None)
        _rf_feats = getattr(_cwps, "_last_rf_feature_names", top_params)
        _rf_hp    = getattr(_cwps, "_last_rf_hyperparams", {"n_estimators": 300})
        ai_session_id = repo.create_session(
            "ai-initial", "rf_initial", None, settings.objective, top_params,
            metadata={"candidates": candidates, "population_size": population_size},
        )
        repo.save_surrogate_model(
            session_id=ai_session_id,
            model_type="random_forest",
            target_metric="qps_latency_score",
            train_rows=len(summaries),
            feature_names=_rf_feats,
            hyperparams=_rf_hp,
            train_score=_rf_score,
        )
        repo.finish_session(ai_session_id, None, None)
        if _rf_score is not None:
            console.print(f"[dim]RF-суррогат сохранён в surrogate_models (R²={_rf_score:.3f}, обучен на {len(summaries)} точках)[/dim]")
    except Exception as _exc:
        console.print(f"[yellow]Предупреждение: не удалось сохранить RF-суррогат в БД: {_exc}[/yellow]")

    # Таблица: RF выбрал эти конфигурации
    from rich import box as _box
    ai_table = Table(
        title=f"🤖  Этап 1 (AI/RF): топ-{len(population)} конфигураций, отобранных из {n_candidates} кандидатов",
        box=_box.ROUNDED,
    )
    ai_table.add_column("#", justify="center", style="cyan")
    ai_table.add_column("RF predicted score", justify="right", style="bold green")
    for param in (top_params or [])[:6]:
        ai_table.add_column(param, justify="right", style="dim")
    for i, (cfg, ps) in enumerate(zip(population, pred_scores or [None]*len(population)), start=1):
        row = [str(i), f"{ps:.4f}" if ps is not None else "—"]
        for param in (top_params or [])[:6]:
            v = cfg.get(param)
            row.append(f"{v:g}" if isinstance(v, float) else str(v) if v is not None else "—")
        ai_table.add_row(*row)
    console.print(ai_table)
    console.print_json(json.dumps({"initial_population_size": len(population), "top_params": top_params}, ensure_ascii=False))
    if evaluate:
        workload_id = repo.get_or_create_workload(settings.benchmark.get("workload_name", "tsbs-devops"), tool="tsbs")
        session_id = repo.create_session("ai-initial", "rf_initial", workload_id, settings.objective, top_params)
        best_ev = None
        for idx, candidate in enumerate(population):
            ev = benchmark.evaluate(candidate, source="rf_initial", stage="rf_initial", generation=0, candidate_index=idx)
            repo.insert_trial(session_id, 0, idx, ev.config_id, ev.experiment_id, ev.run_id, ev.metrics, ev.score, "finished")
            if best_ev is None or ev.score > best_ev.score:
                best_ev = ev
        repo.finish_session(session_id, best_ev.config_id if best_ev else None, best_ev.score if best_ev else None)


@app.command("ga-optimize")
def ga_optimize(
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    initial_config: Optional[Path] = typer.Option(None, "--initial-config"),
    top_n: int = typer.Option(10, "--top-n"),
    seed: int = typer.Option(42, "--seed"),
    population: Optional[int] = typer.Option(None, "--population"),
    generations: Optional[int] = typer.Option(None, "--generations"),
):
    """Второй этап: генетический алгоритм + опциональное нейросетевое локальное уточнение."""
    settings, specs, repo, benchmark = build_services(config)
    rng = random.Random(seed)
    scope_ids = load_last_scope()
    if scope_ids:
        summaries = repo.summaries_by_experiment_ids(scope_ids)
    else:
        summaries = repo.all_summaries(min_runs=1)
    importances = fit_importances(summaries, specs, top_n=top_n, random_state=seed)
    top_params = union_top_params(importances, top_n) if importances else [s.name for s in specs[:top_n]]
    base_config = None
    initial_population = None
    ai_population_scores: list[float] = []   # predicted RF-scores для отчёта

    # ── Этап 1: загружаем или генерируем стартовую популяцию через RF-суррогат ──
    if initial_config and not initial_config.exists():
        console.print(
            f"[yellow]⚠️  Файл {initial_config} не найден — запускаю ai-initial автоматически...[/yellow]"
        )
        initial_config = None  # сбросим, пересчитаем ниже

    if initial_config and initial_config.exists():
        loaded = json.loads(initial_config.read_text(encoding="utf-8"))
        if isinstance(loaded, dict) and "initial_population" in loaded:
            initial_population = loaded.get("initial_population") or []
            base_config = loaded.get("best_config") or (initial_population[0] if initial_population else None)
            top_params = loaded.get("top_params") or top_params
            ai_population_scores = loaded.get("predicted_scores") or []
            console.print(
                f"[green]✓ Этап 1 (AI/RF):[/green] загружена стартовая популяция "
                f"из {initial_config} — {len(initial_population)} конфигураций, "
                f"отобранных RF-суррогатом из {loaded.get('candidates_evaluated', '?')} кандидатов."
            )
        else:
            base_config = loaded
    else:
        # ai-initial не был запущен — делаем его прямо сейчас
        console.print("[cyan]▶ Этап 1 (AI/RF): генерирую стартовую популяцию через RF-суррогат...[/cyan]")
        n_candidates = int(settings.optimizer.get("random_candidates_for_ai", 500))
        population_size_ai = population or int(settings.optimizer.get("ga_population", 12))
        initial_population, ai_population_scores, _ = choose_initial_population_with_scores(
            summaries, specs, settings.objective, rng, n_candidates, population_size_ai, top_params
        )
        base_config = initial_population[0] if initial_population else None
        console.print(
            f"[green]✓ Этап 1 (AI/RF):[/green] RF-суррогат отобрал {len(initial_population)} "
            f"конфигураций из {n_candidates} LHS-кандидатов "
            f"(лучший predicted score: {ai_population_scores[0]:.4f})."
        )

    workload_id = repo.get_or_create_workload(settings.benchmark.get("workload_name", "tsbs-devops"), tool="tsbs")
    session_id = repo.create_session("ga-optimize", "ga", workload_id, settings.objective, top_params)

    # ── Запоминаем baseline (лучший из LHS) перед запуском ГА ──────────────
    baseline_summary = repo.get_lhs_baseline_summary(scope_ids=scope_ids or [])

    optimizer = GeneticOptimizer(
        specs=specs,
        benchmark=benchmark,
        rng=rng,
        top_params=top_params,
        population_size=population or int(settings.optimizer.get("ga_population", 12)),
        generations=generations or int(settings.optimizer.get("ga_generations", 5)),
        mutation_probability=float(settings.optimizer.get("mutation_probability", 0.08)),
        crossover_probability=float(settings.optimizer.get("crossover_probability", 0.8)),
        elite_count=int(settings.optimizer.get("elite_count", 2)),
        tournament_size=int(settings.optimizer.get("tournament_size", 3)),
        local_gradient_steps=int(settings.optimizer.get("local_gradient_steps", 0)),
        local_learning_rate=float(settings.optimizer.get("local_learning_rate", 0.08)),
    )
    start_ts = time.time()
    try:
        result = optimizer.optimize(session_id, base_config=base_config, initial_population=initial_population)
        elapsed = time.time() - start_ts

        best_config_id = result.best_evaluation.config_id if result.best_evaluation else None
        repo.finish_session(session_id, best_config_id, result.best_score)

        runtime_dir = Path("/app/runtime")
        runtime_dir.mkdir(parents=True, exist_ok=True)
        best_config_path = runtime_dir / "best_ga_config.json"
        best_config_path.write_text(
            json.dumps(result.best_config, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # ══════════════════════════════════════════════════════════════════════
        # ОТЧЁТ: Этап 1 — что RF-суррогат отобрал как стартовую популяцию
        # ══════════════════════════════════════════════════════════════════════
        if initial_population:
            from rich import box as _rbox
            console.print()
            ai_tbl = Table(
                title=f"🤖  Этап 1 (AI/RF): стартовая популяция ГА — {len(initial_population)} конфигураций",
                box=_rbox.ROUNDED,
                show_lines=True,
            )
            ai_tbl.add_column("#", justify="center", style="cyan")
            ai_tbl.add_column("RF predicted score", justify="right", style="bold")
            for param in top_params[:5]:
                ai_tbl.add_column(param, justify="right", style="dim")
            for i, cfg in enumerate(initial_population, start=1):
                ps = ai_population_scores[i - 1] if i - 1 < len(ai_population_scores) else None
                row = [str(i), f"{ps:.4f}" if ps is not None else "—"]
                for param in top_params[:5]:
                    v = cfg.get(param)
                    row.append(f"{v:g}" if isinstance(v, float) else str(v) if v is not None else "—")
                ai_tbl.add_row(*row)
            console.print(ai_tbl)
            console.print(
                "[dim]RF-суррогат обучен на данных random-search (LHS) и отобрал "
                "эти конфигурации как наиболее перспективные стартовые точки для ГА.[/dim]"
            )

        # ══════════════════════════════════════════════════════════════════════
        # ОТЧЁТ: Этап 2 — прогресс ГА по поколениям
        # ══════════════════════════════════════════════════════════════════════
        console.print()
        gen_metrics = repo.get_generation_best_metrics(session_id)
        if gen_metrics:
            print_generation_progress(
                [dict(r) for r in gen_metrics],
                title="📈  Этап 2 (ГА + градиентный спуск): прогресс по поколениям",
            )

        # ══════════════════════════════════════════════════════════════════════
        # ОТЧЁТ: топ-5 конфигураций за сессию
        # ══════════════════════════════════════════════════════════════════════
        trials = repo.get_session_trials_ordered(session_id)
        print_best_configs_summary([dict(t) for t in trials], top_n=5)

        # ══════════════════════════════════════════════════════════════════════
        # ОТЧЁТ: сравнение до/после по метрикам производительности
        # ══════════════════════════════════════════════════════════════════════
        final_summary = dict(result.best_evaluation.metrics) if result.best_evaluation else {}

        if baseline_summary:
            baseline_dict = dict(baseline_summary)
            # Нормализуем ОБЕ точки вместе — тогда score сопоставим между ними.
            # Нормализация по двум точкам даёт 0.0/1.0, поэтому берём всю историю.
            from .objective import add_normalized_scores as _norm_scores
            # all_history = repo.all_summaries(min_runs=1)
            # # Добавляем обе точки в общий пул для нормализации
            # scored_all = _norm_scores(all_history, settings.objective)
            if scope_ids:
                run_history = repo.summaries_by_experiment_ids(scope_ids)
            else:
                run_history = repo.all_summaries(min_runs=1)
            # Добавляем GA-эксперименты текущей сессии если их нет в scope_ids
            ga_exp_ids = [
                t["experiment_id"] for t in trials
                if t.get("experiment_id") and t["experiment_id"] not in (scope_ids or [])
            ]
            if ga_exp_ids:
                ga_history = repo.summaries_by_experiment_ids(ga_exp_ids)
                run_history = run_history + [r for r in ga_history if r not in run_history]
            scored_all = _norm_scores(run_history, settings.objective)
            # Находим score baseline и final в нормализованном пространстве всей истории
            def _find_score(target: dict, pool: list) -> float:
                exp_id = target.get("experiment_id")
                for r in pool:
                    if r.get("experiment_id") == exp_id:
                        return float(r.get("score") or 0.0)
                # fallback: нормализуем вместе с пулом
                scored = _norm_scores(run_history + [target], settings.objective)
                return float(scored[-1].get("score") or 0.0)

            baseline_dict["score"] = _find_score(baseline_dict, scored_all)
            final_exp_id = result.best_evaluation.experiment_id if result.best_evaluation else None
            if final_exp_id:
                for r in scored_all:
                    if r.get("experiment_id") == final_exp_id:
                        final_summary["score"] = float(r.get("score") or result.best_score)
                        break
                else:
                    final_summary["score"] = result.best_score
            else:
                final_summary["score"] = result.best_score

            print_before_after_comparison(
                baseline=baseline_dict,
                final=final_summary,
                label_baseline="LHS baseline (случайный поиск)",
                label_final="ГА + градиентный спуск",
            )

        # ══════════════════════════════════════════════════════════════════════
        # ОТЧЁТ: метрики контейнеров (если включён мониторинг)
        # ══════════════════════════════════════════════════════════════════════
        if result.best_evaluation and result.best_evaluation.container_stats:
            first_exp = repo.get_first_finished_experiment_for_session(session_id)
            baseline_container_stats: dict = {}
            if first_exp:
                cont_rows = repo.get_experiment_container_stats(int(first_exp["experiment_id"]))
                baseline_container_stats = {r["container_name"]: _RowStats(r) for r in cont_rows}
            final_container_stats = {
                name: _DictStats(s)
                for name, s in result.best_evaluation.container_stats.items()
            }
            if baseline_container_stats or final_container_stats:
                print_container_stats(baseline_container_stats, final_container_stats)

        # ══════════════════════════════════════════════════════════════════════
        # ОТЧЁТ: внутренние метрики СУБД (только если реально собраны данные)
        # ══════════════════════════════════════════════════════════════════════
        if result.best_evaluation:
            import json as _json
            baseline_pg: dict = {}
            final_pg: dict = result.best_evaluation.pg_stats_post or {}

            # pg_stats_pre лучшего — или из первого эксперимента сессии
            if result.best_evaluation.pg_stats_pre:
                baseline_pg = result.best_evaluation.pg_stats_pre
            else:
                first_exp = repo.get_first_finished_experiment_for_session(session_id)
                if first_exp:
                    pg_row = repo.get_experiment_pg_stats(int(first_exp["experiment_id"]), "pre_run")
                    if pg_row:
                        raw = pg_row.get("stats_json") or {}
                        baseline_pg = _json.loads(raw) if isinstance(raw, str) else dict(raw)

            # Показываем таблицу если хотя бы одна сторона содержит данные без ошибки
            baseline_ok = baseline_pg and "error" not in baseline_pg
            final_ok    = final_pg    and "error" not in final_pg
            if baseline_ok or final_ok:
                print_pg_stats_comparison(
                    baseline_pg if baseline_ok else {},
                    final_pg    if final_ok    else {},
                )
            elif baseline_pg.get("error") or final_pg.get("error"):
                err = baseline_pg.get("error") or final_pg.get("error")
                console.print(f"[yellow]⚠️  Метрики СУБД недоступны: {err}[/yellow]")

        # ══════════════════════════════════════════════════════════════════════
        # ОТЧЁТ: финальная конфигурация параметров
        # ══════════════════════════════════════════════════════════════════════
        _print_final_config(result.best_config, specs, top_params)

        # ══════════════════════════════════════════════════════════════════════
        # ИТОГОВЫЙ БАННЕР
        # ══════════════════════════════════════════════════════════════════════
        # Score для баннера — нормализованный по всей истории (LHS + GA вместе)
        # чтобы baseline и final были в одной шкале [0..1]
        baseline_score = float(baseline_dict.get("score") or 0.0) if baseline_summary else 0.0
        final_score_banner = float(final_summary.get("score") or result.best_score)
        total_experiments = len(trials)
        print_optimization_summary_banner(
            baseline_score=baseline_score,
            final_score=final_score_banner,
            total_experiments=total_experiments,
            elapsed_sec=elapsed,
            top_params=top_params,
        )

        console.print(f"[dim]Конфигурация сохранена: {best_config_path}[/dim]\n")

    except KeyboardInterrupt:
        repo.finish_session(session_id, None, None, status="interrupted")
        raise typer.Exit(130)


def _print_final_config(
    best_config: dict,
    specs: list,
    top_params: list[str],
) -> None:
    """Выводит таблицу финальных значений оптимизированных параметров."""
    from rich import box as rich_box
    spec_map = {s.name: s for s in specs}

    table = Table(
        title="⚙️   Итоговая конфигурация параметров TimescaleDB/PostgreSQL",
        box=rich_box.ROUNDED,
        show_lines=True,
    )
    table.add_column("Параметр", style="cyan", min_width=35)
    table.add_column("Значение", justify="right", style="bold yellow", min_width=16)
    table.add_column("Единица", justify="center", style="dim")
    table.add_column("Группа", justify="center", style="dim")
    table.add_column("Оптимизировался", justify="center")

    # Сначала выводим оптимизированные параметры, потом остальные
    optimized = [(k, v) for k, v in sorted(best_config.items()) if k in top_params]
    others    = [(k, v) for k, v in sorted(best_config.items()) if k not in top_params]

    for key, val in optimized + others:
        spec = spec_map.get(key)
        unit  = spec.unit  if spec and spec.unit  and spec.unit  != "none" else ""
        group = spec.group if spec and spec.group else ""
        # Форматируем значение: bool → on/off, float без лишних нулей
        if isinstance(val, bool):
            val_str = "on" if val else "off"
        elif isinstance(val, float):
            val_str = f"{val:g}"
        else:
            val_str = str(val)
        is_opt = "✅" if key in top_params else ""
        table.add_row(key, val_str, unit, group, is_opt)

    console.print(table)


class _RowStats:
    """Обёртка над dict из БД для совместимости с print_container_stats."""
    def __init__(self, d: dict):
        for k, v in d.items():
            setattr(self, k, v)


class _DictStats:
    """Обёртка над dict из EvaluationResult.container_stats."""
    def __init__(self, d: dict):
        for k, v in d.items():
            setattr(self, k, v)


@app.command("nn-local-optimize")
def nn_local_optimize(
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    initial_config: Path = typer.Option(..., "--initial-config"),
    top_n: int = typer.Option(10, "--top-n"),
    seed: int = typer.Option(42, "--seed"),
    steps: int = typer.Option(12, "--steps"),
):
    """Отдельный запуск локального градиентного уточнения по нейросетевой суррогатной модели."""
    settings, specs, repo, benchmark = build_services(config)
    rng = random.Random(seed)
    loaded = json.loads(initial_config.read_text(encoding="utf-8"))
    init_cfg = repair_config(loaded.get("best_config", loaded) if isinstance(loaded, dict) else loaded)
    # summaries = repo.all_summaries(min_runs=1)
    scope_ids = load_last_scope()
    if scope_ids:
        summaries = repo.summaries_by_experiment_ids(scope_ids)
    else:
        summaries = repo.all_summaries(min_runs=1)
    importances = fit_importances(summaries, specs, top_n=top_n, random_state=seed)
    top_params = union_top_params(importances, top_n) if importances else [s.name for s in specs[:top_n]]
    surrogate = NeuralSurrogate(specs, top_params, rng.randint(1, 10_000))
    if not surrogate.fit(summaries, settings.objective):
        console.print("[yellow]Недостаточно данных для обучения нейросетевого суррогата.[/yellow]")
        raise typer.Exit(1)
    improved = surrogate.improve(init_cfg, steps=steps)
    ev = benchmark.evaluate(improved.config, source="nn_gradient", stage="local_gradient", generation=0, candidate_index=0)
    Path("best_nn_local_config.json").write_text(json.dumps(improved.config, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"[bold green]score={ev.score:.3f}. Файл: best_nn_local_config.json[/bold green]")


@app.command("show-best")
def show_best(config: Optional[str] = typer.Option(None, "--config", "-c"), limit: int = typer.Option(10, "--limit")):
    _, _, repo, _ = build_services(config)
    rows = repo.db.fetch_all(
        """
        SELECT e.id AS experiment_id, e.score, e.stage, e.created_at, c.id AS config_id, c.params
        FROM public.experiments e
        JOIN public.configs c ON c.id = e.config_id
        WHERE e.score IS NOT NULL
        ORDER BY e.score DESC
        LIMIT %s
        """,
        (limit,),
    )
    table = Table(title="Лучшие конфигурации")
    table.add_column("#")
    table.add_column("score", justify="right")
    table.add_column("stage")
    table.add_column("experiment")
    table.add_column("config")
    for i, row in enumerate(rows, start=1):
        table.add_row(str(i), f"{row['score']:.3f}", str(row["stage"]), str(row["experiment_id"]), str(row["config_id"]))
    console.print(table)



# def show_progress(
#     config: Optional[str] = typer.Option(None, "--config", "-c"),
#     session_id: Optional[int] = typer.Option(None, "--session-id", "-s", help="ID сессии оптимизации (по умолчанию — последняя)"),
#     top_n: int = typer.Option(5, "--top-n"),
# ):
#     """Показывает прогресс оптимизации: прогресс по поколениям ГА, топ конфигурации и сравнение до/после."""
#     _, _, repo, _ = build_services(config)

#     # Находим нужную сессию
#     if session_id is None:
#         row = repo.db.fetch_one(
#             "SELECT id FROM public.optimization_sessions WHERE algorithm = 'ga' ORDER BY id DESC LIMIT 1"
#         )
#         if not row:
#             console.print("[yellow]Нет сессий ГА. Сначала запустите ga-optimize.[/yellow]")
#             raise typer.Exit(1)
#         session_id = int(row["id"])

#     console.print(f"\n[bold]Сессия ГА #{session_id}[/bold]")

#     gen_metrics = repo.get_generation_best_metrics(session_id)
#     if gen_metrics:
#         print_generation_progress([dict(r) for r in gen_metrics])

#     trials = repo.get_session_trials_ordered(session_id)
#     print_best_configs_summary([dict(t) for t in trials], top_n=top_n)

#     # baseline vs best
    
#     _scope_ids = load_last_scope()  # читает last_scope.json
#     baseline = repo.get_lhs_baseline_summary(scope_ids=_scope_ids or [])
#     if baseline and trials:
#         best_trial = dict(trials[0])
#         import json as _json
#         metrics = best_trial.get("metrics") or {}
#         if isinstance(metrics, str):
#             try:
#                 metrics = _json.loads(metrics)
#             except Exception:
#                 metrics = {}
#         metrics["score"] = best_trial.get("score", 0.0)
#         print_before_after_comparison(
#             baseline=dict(baseline),
#             final=metrics,
#             label_baseline="LHS baseline",
#             label_final=f"Лучший результат ГА (config #{best_trial.get('config_id')})",
#         )
@app.command("show-progress")
def show_progress(
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    session_id: Optional[int] = typer.Option(None, "--session-id", "-s", help="ID сессии оптимизации (по умолчанию — последняя)"),
    top_n: int = typer.Option(5, "--top-n"),
):
    """Показывает прогресс оптимизации: прогресс по поколениям ГА, топ конфигурации и сравнение до/после."""
    _, _, repo, _ = build_services(config)

    # Находим нужную сессию
    if session_id is None:
        row = repo.db.fetch_one(
            "SELECT id FROM public.optimization_sessions WHERE algorithm = 'ga' ORDER BY id DESC LIMIT 1"
        )
        if not row:
            console.print("[yellow]Нет сессий ГА. Сначала запустите ga-optimize.[/yellow]")
            raise typer.Exit(1)
        session_id = int(row["id"])

    console.print(f"\n[bold]Сессия ГА #{session_id}[/bold]")

    gen_metrics = repo.get_generation_best_metrics(session_id)
    if gen_metrics:
        print_generation_progress([dict(r) for r in gen_metrics])

    trials = repo.get_session_trials_ordered(session_id)
    print_best_configs_summary([dict(t) for t in trials], top_n=top_n)

    # baseline vs best — пересчитываем score по всему scope запуска
    import json as _json
    from .objective import add_normalized_scores as _norm_scores
    from .state import load_last_scope as _load_scope
    _scope_ids = _load_scope()
    baseline = repo.get_lhs_baseline_summary(scope_ids=_scope_ids or [])
    if baseline and trials:
        best_trial = dict(trials[0])
        metrics = best_trial.get("metrics") or {}
        if isinstance(metrics, str):
            try:
                metrics = _json.loads(metrics)
            except Exception:
                metrics = {}

        # Пересчитываем score по всему scope (LHS + GA) — как в ga-optimize
        _settings_sp = load_settings(config)
        _ga_ids = [t["experiment_id"] for t in trials if t.get("experiment_id")]
        _all_ids = sorted(set((_scope_ids or []) + _ga_ids))
        _all_summaries = repo.summaries_by_experiment_ids(_all_ids) if _all_ids else []
        _scored = _norm_scores(_all_summaries, _settings_sp.objective) if _all_summaries else []

        baseline_dict = dict(baseline)
        final_dict    = dict(metrics)
        for r in _scored:
            exp_id = r.get("experiment_id")
            if exp_id == baseline_dict.get("experiment_id"):
                baseline_dict["score"] = float(r.get("score") or 0.0)
            if exp_id == best_trial.get("experiment_id"):
                final_dict["score"] = float(r.get("score") or 0.0)

        print_before_after_comparison(
            baseline=baseline_dict,
            final=final_dict,
            label_baseline="LHS baseline (последний random-search)",
            label_final=f"Лучший результат ГА (config #{best_trial.get('config_id')})",
        )


@app.command("show-monitoring")
def show_monitoring(
    config: Optional[str] = typer.Option(None, "--config", "-c"),
    experiment_id: int = typer.Argument(..., help="ID эксперимента для просмотра метрик"),
):
    """Показывает сохранённые метрики мониторинга контейнеров и СУБД для эксперимента."""
    import json as _json
    _, _, repo, _ = build_services(config)

    # Контейнеры
    rows = repo.get_experiment_container_stats(experiment_id)
    if rows:
        table = Table(title=f"🐳  Метрики контейнеров: эксперимент #{experiment_id}", box=__import__("rich.box", fromlist=["ROUNDED"]).ROUNDED)
        table.add_column("Контейнер")
        table.add_column("CPU avg %", justify="right")
        table.add_column("CPU max %", justify="right")
        table.add_column("RAM avg MiB", justify="right")
        table.add_column("RAM max MiB", justify="right")
        table.add_column("Disk R MiB", justify="right")
        table.add_column("Disk W MiB", justify="right")
        table.add_column("Длительность", justify="right")
        for r in rows:
            table.add_row(
                str(r["container_name"]),
                f"{r['cpu_pct_avg'] or 0:.1f}",
                f"{r['cpu_pct_max'] or 0:.1f}",
                f"{r['mem_used_mb_avg'] or 0:.0f}",
                f"{r['mem_used_mb_max'] or 0:.0f}",
                f"{r['blk_read_delta_mb'] or 0:.2f}",
                f"{r['blk_write_delta_mb'] or 0:.2f}",
                f"{r['duration_sec'] or 0:.0f}s",
            )
        console.print(table)
    else:
        console.print("[dim]Метрики контейнеров отсутствуют. Убедитесь, что в tuner.yml задан benchmark.monitor_containers.[/dim]")

    # Метрики СУБД
    for snap_type in ("pre_run", "post_run"):
        pg_row = repo.get_experiment_pg_stats(experiment_id, snap_type)
        if pg_row:
            raw = pg_row.get("stats_json") or {}
            pg_data = _json.loads(raw) if isinstance(raw, str) else dict(raw)
            label = "ДО бенчмарка" if snap_type == "pre_run" else "ПОСЛЕ бенчмарка"
            console.print(f"\n[bold]Метрики СУБД ({label}):[/bold]")
            console.print_json(_json.dumps(pg_data, ensure_ascii=False, default=str))


if __name__ == "__main__":
    app()