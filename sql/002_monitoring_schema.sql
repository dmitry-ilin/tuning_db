-- Таблицы мониторинга для хранения метрик контейнеров и СУБД.
-- Применяется автоматически командой: python -m tsdb_tuner init-db

-- Снимки метрик контейнеров (CPU, RAM, IO), привязанные к эксперименту
CREATE TABLE IF NOT EXISTS public.experiment_container_stats
(
    id                 serial PRIMARY KEY,
    experiment_id      integer REFERENCES public.experiments(id) ON DELETE CASCADE,
    container_name     text    NOT NULL,
    samples            integer NOT NULL DEFAULT 0,
    cpu_pct_avg        double precision,
    cpu_pct_max        double precision,
    mem_used_mb_avg    double precision,
    mem_used_mb_max    double precision,
    mem_pct_avg        double precision,
    net_rx_delta_mb    double precision,
    net_tx_delta_mb    double precision,
    blk_read_delta_mb  double precision,
    blk_write_delta_mb double precision,
    duration_sec       double precision,
    created_at         timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_exp_container_stats_exp
    ON public.experiment_container_stats(experiment_id);

-- Снимки внутренних метрик СУБД (pg_stat_bgwriter, cache hit ratio, соединения и т.д.)
CREATE TABLE IF NOT EXISTS public.experiment_pg_stats
(
    id                  serial PRIMARY KEY,
    experiment_id       integer REFERENCES public.experiments(id) ON DELETE CASCADE,
    snapshot_type       text NOT NULL DEFAULT 'post_run',  -- 'pre_run' | 'post_run'
    stats_json          jsonb NOT NULL DEFAULT '{}',
    db_size_mb          double precision,
    cache_hit_ratio     double precision,
    active_connections  integer,
    created_at          timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_exp_pg_stats_exp
    ON public.experiment_pg_stats(experiment_id);

-- Удобное представление: прогресс ГА по поколениям с лучшими метриками
CREATE OR REPLACE VIEW public.v_ga_generation_progress AS
SELECT
    ot.session_id,
    ot.generation,
    COUNT(*)                                                              AS trials_count,
    MAX(ot.score)                                                        AS best_score,
    AVG(ot.score)                                                        AS avg_score,
    MAX((ot.metrics->>'avg_rate_qps')::double precision)                AS best_qps,
    MIN((ot.metrics->>'median_q50_ms')::double precision)               AS best_q50_ms,
    MIN(COALESCE(
        (ot.metrics->>'p99_q99_ms')::double precision,
        (ot.metrics->>'avg_q99_ms')::double precision
    ))                                                                   AS best_q99_ms,
    MIN(COALESCE(
        (ot.metrics->>'p95_q95_ms')::double precision,
        (ot.metrics->>'avg_q95_ms')::double precision
    ))                                                                   AS best_q95_ms
FROM public.optimization_trials ot
WHERE ot.score IS NOT NULL
GROUP BY ot.session_id, ot.generation
ORDER BY ot.session_id, ot.generation;

-- Сводное представление: лучший эксперимент каждой сессии
CREATE OR REPLACE VIEW public.v_session_best AS
SELECT
    os.id          AS session_id,
    os.name        AS session_name,
    os.algorithm,
    os.started_at,
    os.finished_at,
    os.best_score,
    os.status,
    vs.avg_rate_qps,
    vs.median_q50_ms,
    vs.p95_q95_ms,
    vs.p99_q99_ms,
    vs.stage
FROM public.optimization_sessions os
LEFT JOIN public.configs c ON c.id = os.best_config_id
LEFT JOIN public.v_experiment_summary vs
    ON vs.config_id = os.best_config_id
    AND vs.experiment_id = (
        SELECT ot2.experiment_id
        FROM public.optimization_trials ot2
        WHERE ot2.session_id = os.id AND ot2.score = os.best_score
        ORDER BY ot2.id DESC LIMIT 1
    )
ORDER BY os.id DESC;
