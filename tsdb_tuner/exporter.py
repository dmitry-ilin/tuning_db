"""
exporter.py — Prometheus-экспортёр метрик оптимизатора TimescaleDB.
Показывает данные только по ПОСЛЕДНЕЙ GA-сессии.
"""

from __future__ import annotations

import argparse
import os
import time

from prometheus_client import start_http_server, CollectorRegistry, REGISTRY
from prometheus_client.core import GaugeMetricFamily, CounterMetricFamily

import psycopg2
import psycopg2.extras


class TunerMetricsCollector:
    """
    Custom collector — пересоздаёт метрики при каждом scrape.
    Это гарантирует что старые лейблы не накапливаются.
    """

    def __init__(self, dsn: str):
        self.dsn = dsn

    def _connect(self):
        conn = psycopg2.connect(self.dsn)
        conn.set_session(autocommit=True)
        return conn

    def _get_last_session(self, cur) -> int | None:
        """Возвращает id последней GA-сессии."""
        cur.execute("""
            SELECT id FROM public.optimization_sessions
            ORDER BY id DESC LIMIT 1
        """)
        row = cur.fetchone()
        return int(row["id"]) if row else None

    # def _get_run_scope(self, cur, session_id: int) -> list[int]:
    #     """
    #     Возвращает эксперименты текущего запуска:
    #     последний LHS (если был недавно, прямо перед текущей GA) + текущая GA-сессия.

    #     "Недавно" = LHS-сессия с id непосредственно перед текущей GA-сессией
    #     (нет других GA-сессий между ними).
    #     """
    #     # Ищем LHS-сессию с id меньше текущей GA И без GA-сессий между ними
    #     cur.execute("""
    #         SELECT id, started_at
    #         FROM public.optimization_sessions
    #         WHERE algorithm IN ('random', 'lhs', 'random_search', 'initial_sampling')
    #           AND id < %s
    #           AND NOT EXISTS (
    #               SELECT 1 FROM public.optimization_sessions gap
    #               WHERE gap.algorithm NOT IN ('random', 'lhs', 'random_search',
    #                                           'initial_sampling', 'rf_initial')
    #                 AND gap.id > (
    #                     SELECT COALESCE(MAX(s2.id), 0)
    #                     FROM public.optimization_sessions s2
    #                     WHERE s2.algorithm IN ('random', 'lhs', 'random_search', 'initial_sampling')
    #                       AND s2.id < %s
    #                 )
    #                 AND gap.id < %s
    #           )
    #         ORDER BY id DESC LIMIT 1
    #     """, (session_id, session_id, session_id))
    #     lhs = cur.fetchone()

    #     # Эксперименты текущей GA-сессии
    #     cur.execute("""
    #         SELECT DISTINCT experiment_id
    #         FROM public.optimization_trials
    #         WHERE session_id = %s AND experiment_id IS NOT NULL
    #         ORDER BY experiment_id
    #     """, (session_id,))
    #     ga_exp_ids = [r["experiment_id"] for r in cur.fetchall()]

    #     if not lhs:
    #         return ga_exp_ids

    #     # Добавляем LHS-эксперименты
    #     cur.execute("""
    #         SELECT id FROM public.experiments
    #         WHERE created_at >= %s
    #           AND created_at < (
    #               SELECT started_at FROM public.optimization_sessions WHERE id = %s
    #           )
    #         ORDER BY id
    #     """, (lhs["started_at"], session_id))
    #     lhs_exp_ids = [r["id"] for r in cur.fetchall()]

    #     return sorted(set(ga_exp_ids + lhs_exp_ids))

    def _get_run_scope(self, cur, session_id: int) -> list[int]:
        """
        Возвращает эксперименты текущего запуска.
        Читает last_scope.json (LHS-эксперименты) и добавляет GA-эксперименты сессии.
        Это точно совпадает с логикой cli.py.
        """
        import json as _json

        # Читаем LHS-эксперименты из last_scope.json
        lhs_exp_ids: list[int] = []
        scope_path = "/app/runtime/last_scope.json"
        try:
            with open(scope_path) as f:
                data = _json.load(f)
                lhs_exp_ids = [int(x) for x in data.get("experiment_ids", [])]
        except (FileNotFoundError, Exception):
            pass

        # GA-эксперименты текущей сессии
        cur.execute("""
            SELECT DISTINCT experiment_id
            FROM public.optimization_trials
            WHERE session_id = %s AND experiment_id IS NOT NULL
            ORDER BY experiment_id
        """, (session_id,))
        ga_exp_ids = [r["experiment_id"] for r in cur.fetchall()]

        # Объединяем LHS + GA
        all_ids = sorted(set(lhs_exp_ids + ga_exp_ids))
        return all_ids if all_ids else ga_exp_ids

    def _get_session_experiments(self, cur, session_id: int) -> list[int]:
        """Эксперименты последней сессии (только GA — для container/pg метрик)."""
        cur.execute("""
            SELECT DISTINCT experiment_id
            FROM public.optimization_trials
            WHERE session_id = %s AND experiment_id IS NOT NULL
            ORDER BY experiment_id
        """, (session_id,))
        return [r["experiment_id"] for r in cur.fetchall()]

    def describe(self):
        return []

    def collect(self):
        try:
            conn = self._connect()
            try:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    session_id = self._get_last_session(cur)
                    if not session_id:
                        return
                    # run_scope — все эксперименты запуска (LHS + GA) для нормализации score
                    run_scope = self._get_run_scope(cur, session_id)
                    # session_exp — только GA-эксперименты для container/pg метрик
                    session_exp = self._get_session_experiments(cur, session_id)
                    if not session_exp:
                        return

                    yield from self._yield_experiment_metrics(cur, run_scope, session_id)
                    yield from self._yield_session_metrics(cur, session_id)
                    yield from self._yield_generation_metrics(cur, session_id)
                    yield from self._yield_container_metrics(cur, session_exp)
                    yield from self._yield_pg_metrics(cur, session_exp)
                    yield from self._yield_surrogate_metrics(cur, session_id)
            finally:
                conn.close()
        except Exception as exc:
            print(f"[exporter] collect error: {exc}")

    def _yield_experiment_metrics(self, cur, exp_ids, session_id):
        cur.execute("""
            SELECT vs.experiment_id, vs.stage,
                   vs.avg_rate_qps, vs.median_q50_ms, vs.p95_q95_ms, vs.p99_q99_ms
            FROM public.v_experiment_summary vs
            WHERE vs.experiment_id = ANY(%s) AND vs.avg_rate_qps IS NOT NULL
            ORDER BY vs.experiment_id
        """, (exp_ids,))
        rows = cur.fetchall()

        # Нормализуем score только по экспериментам текущего запуска
        # (те же exp_ids что уже отфильтрованы по последней сессии)
        cur.execute("""
            SELECT
                MIN(avg_rate_qps) AS qps_min, MAX(avg_rate_qps) AS qps_max,
                MIN(p95_q95_ms)   AS p95_min, MAX(p95_q95_ms)   AS p95_max,
                MIN(p99_q99_ms)   AS p99_min, MAX(p99_q99_ms)   AS p99_max,
                MIN(median_q50_ms)AS p50_min, MAX(median_q50_ms)AS p50_max
            FROM public.v_experiment_summary
            WHERE experiment_id = ANY(%s) AND avg_rate_qps IS NOT NULL
        """, (exp_ids,))
        bounds = cur.fetchone() or {}

        def norm(val, vmin, vmax, higher_is_better=True):
            if val is None or vmin is None or vmax is None:
                return None
            r = vmax - vmin
            if r == 0:
                return 1.0
            n = (float(val) - float(vmin)) / float(r)
            return n if higher_is_better else 1.0 - n

        g_qps  = GaugeMetricFamily("tsdb_tuner_experiment_qps",   "QPS",    labels=["experiment_id", "stage"])
        g_q50  = GaugeMetricFamily("tsdb_tuner_experiment_q50_ms","Q50 ms", labels=["experiment_id", "stage"])
        g_q95  = GaugeMetricFamily("tsdb_tuner_experiment_q95_ms","Q95 ms", labels=["experiment_id", "stage"])
        g_q99  = GaugeMetricFamily("tsdb_tuner_experiment_q99_ms","Q99 ms", labels=["experiment_id", "stage"])
        g_sc   = GaugeMetricFamily("tsdb_tuner_experiment_score",  "Score (нормализованный по истории)", labels=["experiment_id", "stage"])

        for r in rows:
            eid   = str(r["experiment_id"])
            stage = str(r["stage"] or "unknown")
            if r["avg_rate_qps"]:  g_qps.add_metric([eid, stage], float(r["avg_rate_qps"]))
            if r["median_q50_ms"]: g_q50.add_metric([eid, stage], float(r["median_q50_ms"]))
            if r["p95_q95_ms"]:    g_q95.add_metric([eid, stage], float(r["p95_q95_ms"]))
            if r["p99_q99_ms"]:    g_q99.add_metric([eid, stage], float(r["p99_q99_ms"]))
            # Score = 0.3*QPS_norm + 0.3*L95_norm + 0.2*L99_norm + 0.2*L50_norm
            qps_n = norm(r["avg_rate_qps"], bounds.get("qps_min"), bounds.get("qps_max"), True)
            p95_n = norm(r["p95_q95_ms"],   bounds.get("p95_min"), bounds.get("p95_max"), False)
            p99_n = norm(r["p99_q99_ms"],   bounds.get("p99_min"), bounds.get("p99_max"), False)
            p50_n = norm(r["median_q50_ms"],bounds.get("p50_min"), bounds.get("p50_max"), False)
            if all(v is not None for v in [qps_n, p95_n, p99_n, p50_n]):
                score = 0.3 * qps_n + 0.3 * p95_n + 0.2 * p99_n + 0.2 * p50_n
                g_sc.add_metric([eid, stage], round(score, 4))

        yield g_qps; yield g_q50; yield g_q95; yield g_q99; yield g_sc

    def _yield_session_metrics(self, cur, session_id):
        cur.execute("""
            SELECT
                os.id::text AS sid,
                os.algorithm,
                os.name,
                -- best_score из сессии или текущий максимум из trials (если сессия ещё идёт)
                COALESCE(
                    os.best_score,
                    (SELECT MAX(t.score) FROM public.optimization_trials t
                     WHERE t.session_id = os.id AND t.score IS NOT NULL)
                ) AS best_score,
                os.started_at,
                os.finished_at,
                (SELECT count(*) FROM public.optimization_trials t WHERE t.session_id = os.id) AS exp_count,
                -- Лучший QPS из экспериментов сессии
                (SELECT MAX((t.metrics->>'avg_rate_qps')::double precision)
                 FROM public.optimization_trials t
                 WHERE t.session_id = os.id AND t.score IS NOT NULL) AS best_qps,
                -- Лучший Q99 из экспериментов сессии
                (SELECT MIN((t.metrics->>'p99_q99_ms')::double precision)
                 FROM public.optimization_trials t
                 WHERE t.session_id = os.id AND t.score IS NOT NULL) AS best_q99_ms
            FROM public.optimization_sessions os
            WHERE os.id = %s
        """, (session_id,))
        rows = cur.fetchall()

        g_best  = GaugeMetricFamily("tsdb_tuner_session_best_score",        "Best score",  labels=["session_id", "algorithm"])
        g_count = GaugeMetricFamily("tsdb_tuner_session_experiments_total",  "Exp count",   labels=["session_id", "algorithm"])
        g_dur   = GaugeMetricFamily("tsdb_tuner_session_duration_seconds",   "Duration s",  labels=["session_id", "algorithm"])
        g_qps   = GaugeMetricFamily("tsdb_tuner_session_best_qps",           "Best QPS",    labels=["session_id", "algorithm"])
        g_q99   = GaugeMetricFamily("tsdb_tuner_session_best_q99_ms",        "Best Q99 ms", labels=["session_id", "algorithm"])

        for r in rows:
            sid  = str(r["sid"])
            algo = str(r["algorithm"] or "ga")
            if r["best_score"] is not None: g_best.add_metric([sid, algo], float(r["best_score"]))
            if r["exp_count"]:              g_count.add_metric([sid, algo], int(r["exp_count"]))
            if r["best_qps"] is not None:   g_qps.add_metric([sid, algo],  float(r["best_qps"]))
            if r["best_q99_ms"] is not None:g_q99.add_metric([sid, algo],  float(r["best_q99_ms"]))
            if r["started_at"]:
                elapsed = (r["finished_at"] or __import__('datetime').datetime.now(r["started_at"].tzinfo)) - r["started_at"]
                g_dur.add_metric([sid, algo], elapsed.total_seconds())

        yield g_best; yield g_count; yield g_dur; yield g_qps; yield g_q99

    def _yield_generation_metrics(self, cur, session_id):
        cur.execute("""
            SELECT session_id::text AS sid, generation::text AS gen,
                   best_score, best_qps, best_q99_ms
            FROM public.v_ga_generation_progress
            WHERE session_id = %s
            ORDER BY generation
        """, (session_id,))
        rows = cur.fetchall()

        g_sc  = GaugeMetricFamily("tsdb_tuner_ga_generation_best_score",  "Best score", labels=["generation"])
        g_qps = GaugeMetricFamily("tsdb_tuner_ga_generation_best_qps",    "Best QPS",   labels=["generation"])
        g_q99 = GaugeMetricFamily("tsdb_tuner_ga_generation_best_q99_ms", "Best Q99",   labels=["generation"])

        for r in rows:
            gen = str(r["gen"])
            if r["best_score"] is not None: g_sc.add_metric([gen],  float(r["best_score"]))
            if r["best_qps"]   is not None: g_qps.add_metric([gen], float(r["best_qps"]))
            if r["best_q99_ms"]is not None: g_q99.add_metric([gen], float(r["best_q99_ms"]))

        yield g_sc; yield g_qps; yield g_q99

    def _yield_container_metrics(self, cur, exp_ids):
        cur.execute("""
            SELECT experiment_id::text AS eid, container_name,
                   cpu_pct_avg, cpu_pct_max, mem_used_mb_avg, mem_used_mb_max,
                   blk_read_delta_mb, blk_write_delta_mb, net_rx_delta_mb, net_tx_delta_mb
            FROM public.experiment_container_stats
            WHERE experiment_id = ANY(%s)
            ORDER BY experiment_id, container_name
        """, (exp_ids,))
        rows = cur.fetchall()

        metrics = {
            "cpu_pct_avg":       GaugeMetricFamily("tsdb_tuner_container_cpu_pct_avg",     "CPU avg %",      labels=["experiment_id", "container"]),
            "cpu_pct_max":       GaugeMetricFamily("tsdb_tuner_container_cpu_pct_max",     "CPU max %",      labels=["experiment_id", "container"]),
            "mem_used_mb_avg":   GaugeMetricFamily("tsdb_tuner_container_mem_used_mb_avg", "RAM avg MiB",    labels=["experiment_id", "container"]),
            "mem_used_mb_max":   GaugeMetricFamily("tsdb_tuner_container_mem_used_mb_max", "RAM max MiB",    labels=["experiment_id", "container"]),
            "blk_read_delta_mb": GaugeMetricFamily("tsdb_tuner_container_blk_read_mb",    "Disk read MiB",  labels=["experiment_id", "container"]),
            "blk_write_delta_mb":GaugeMetricFamily("tsdb_tuner_container_blk_write_mb",   "Disk write MiB", labels=["experiment_id", "container"]),
            "net_rx_delta_mb":   GaugeMetricFamily("tsdb_tuner_container_net_rx_mb",      "Net RX MiB",     labels=["experiment_id", "container"]),
            "net_tx_delta_mb":   GaugeMetricFamily("tsdb_tuner_container_net_tx_mb",      "Net TX MiB",     labels=["experiment_id", "container"]),
        }

        for r in rows:
            eid  = str(r["eid"])
            name = str(r["container_name"])
            for field, gauge in metrics.items():
                if r[field] is not None:
                    gauge.add_metric([eid, name], float(r[field]))

        yield from metrics.values()

    def _yield_pg_metrics(self, cur, exp_ids):
        cur.execute("""
            SELECT experiment_id::text AS eid, snapshot_type,
                   cache_hit_ratio, active_connections, db_size_mb,
                   (stats_json->'bgwriter'->>'checkpoints_req')::double precision AS ckpt_req,
                   (stats_json->'bgwriter'->>'buffers_alloc')::double precision   AS buf_alloc
            FROM public.experiment_pg_stats
            WHERE experiment_id = ANY(%s)
            ORDER BY experiment_id, snapshot_type
        """, (exp_ids,))
        rows = cur.fetchall()

        g_hit  = GaugeMetricFamily("tsdb_tuner_pg_cache_hit_ratio",      "Cache hit ratio",    labels=["experiment_id", "snapshot"])
        g_conn = GaugeMetricFamily("tsdb_tuner_pg_active_connections",    "Active connections", labels=["experiment_id", "snapshot"])
        g_size = GaugeMetricFamily("tsdb_tuner_pg_db_size_mb",            "DB size MiB",        labels=["experiment_id", "snapshot"])
        g_ckpt = GaugeMetricFamily("tsdb_tuner_pg_checkpoints_req_total", "Checkpoints req",    labels=["experiment_id", "snapshot"])
        g_bufs = GaugeMetricFamily("tsdb_tuner_pg_buffers_alloc_total",   "Buffers alloc",      labels=["experiment_id", "snapshot"])

        for r in rows:
            eid  = str(r["eid"])
            snap = str(r["snapshot_type"])
            if r["cache_hit_ratio"]    is not None: g_hit.add_metric([eid, snap],  float(r["cache_hit_ratio"]))
            if r["active_connections"] is not None: g_conn.add_metric([eid, snap], float(r["active_connections"]))
            if r["db_size_mb"]         is not None: g_size.add_metric([eid, snap], float(r["db_size_mb"]))
            if r["ckpt_req"]           is not None: g_ckpt.add_metric([eid, snap], float(r["ckpt_req"]))
            if r["buf_alloc"]          is not None: g_bufs.add_metric([eid, snap], float(r["buf_alloc"]))

        yield g_hit; yield g_conn; yield g_size; yield g_ckpt; yield g_bufs

    def _yield_surrogate_metrics(self, cur, session_id):
        cur.execute("""
            SELECT session_id::text AS sid, model_type, train_rows, train_score,
                   ROW_NUMBER() OVER (ORDER BY id) AS gen_num
            FROM public.surrogate_models
            WHERE session_id = %s
            ORDER BY id
        """, (session_id,))
        rows = cur.fetchall()

        g_rows = GaugeMetricFamily("tsdb_tuner_surrogate_train_rows",     "Train rows", labels=["model_type", "generation"])
        g_r2   = GaugeMetricFamily("tsdb_tuner_surrogate_train_score_r2", "R² score",   labels=["model_type", "generation"])

        for r in rows:
            mtype = str(r["model_type"])
            gen   = str(r["gen_num"])
            if r["train_rows"]  is not None: g_rows.add_metric([mtype, gen], int(r["train_rows"]))
            if r["train_score"] is not None: g_r2.add_metric([mtype, gen],   float(r["train_score"]))

        yield g_rows; yield g_r2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",     type=int,   default=int(os.getenv("EXPORTER_PORT", "9091")))
    parser.add_argument("--interval", type=float, default=float(os.getenv("SCRAPE_INTERVAL", "15")))
    parser.add_argument("--dsn",      default=os.getenv("RESULTS_DB_DSN", ""))
    args = parser.parse_args()

    if not args.dsn:
        raise SystemExit("Укажите RESULTS_DB_DSN или --dsn")

    REGISTRY.register(TunerMetricsCollector(dsn=args.dsn))

    print(f"[exporter] Запуск на порту {args.port}")
    start_http_server(args.port)

    while True:
        time.sleep(args.interval)


if __name__ == "__main__":
    main()