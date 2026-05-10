-- Расширение существующей БД benchmark_res без удаления старых данных.
-- Скрипт безопасно создает базовые таблицы, если база результатов пустая,
-- а затем добавляет служебные таблицы для автоматического подбора параметров.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS public.configs
(
    id serial PRIMARY KEY,
    params jsonb
);

CREATE TABLE IF NOT EXISTS public.experiments
(
    id serial PRIMARY KEY,
    name text NOT NULL,
    description text,
    created_at timestamptz DEFAULT now(),
    config_id integer REFERENCES public.configs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS public.runs
(
    id serial PRIMARY KEY,
    experiment_id integer REFERENCES public.experiments(id) ON DELETE CASCADE,
    query_file text NOT NULL,
    workers integer,
    limit_rps integer,
    burn_in integer,
    prewarm_queries boolean,
    duration_ms integer,
    start_time timestamptz,
    end_time timestamptz,
    raw_results jsonb,
    created_at timestamptz DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.run_metrics
(
    id serial PRIMARY KEY,
    run_id integer REFERENCES public.runs(id) ON DELETE CASCADE,
    query_name text NOT NULL,
    q50_ms double precision,
    q95_ms double precision,
    q99_ms double precision,
    q999_ms double precision,
    q100_ms double precision,
    q0_ms double precision,
    rate_qps double precision,
    UNIQUE (run_id, query_name)
);

ALTER TABLE public.configs
    ADD COLUMN IF NOT EXISTS params_hash text,
    ADD COLUMN IF NOT EXISTS source text DEFAULT 'unknown',
    ADD COLUMN IF NOT EXISTS parent_config_id integer REFERENCES public.configs(id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS generation integer,
    ADD COLUMN IF NOT EXISTS candidate_index integer,
    ADD COLUMN IF NOT EXISTS created_at timestamptz DEFAULT now(),
    ADD COLUMN IF NOT EXISTS comment text;

-- Заполняем хеши для уже существующих конфигураций.
UPDATE public.configs
SET params_hash = encode(digest(coalesce(params::text, '{}'), 'sha256'), 'hex')
WHERE params_hash IS NULL;

CREATE INDEX IF NOT EXISTS ix_configs_params_hash
    ON public.configs(params_hash);

ALTER TABLE public.experiments
    ADD COLUMN IF NOT EXISTS workload_id integer,
    ADD COLUMN IF NOT EXISTS stage text DEFAULT 'manual',
    ADD COLUMN IF NOT EXISTS status text DEFAULT 'created',
    ADD COLUMN IF NOT EXISTS objective_name text DEFAULT 'qps_latency_score',
    ADD COLUMN IF NOT EXISTS score double precision,
    ADD COLUMN IF NOT EXISTS metadata jsonb DEFAULT '{}'::jsonb;

ALTER TABLE public.runs
    ADD COLUMN IF NOT EXISTS workload_id integer,
    ADD COLUMN IF NOT EXISTS status text DEFAULT 'created',
    ADD COLUMN IF NOT EXISTS exit_code integer,
    ADD COLUMN IF NOT EXISTS stdout text,
    ADD COLUMN IF NOT EXISTS stderr text,
    ADD COLUMN IF NOT EXISTS error_text text,
    ADD COLUMN IF NOT EXISTS system_snapshot jsonb DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS db_settings_snapshot jsonb DEFAULT '{}'::jsonb;

CREATE TABLE IF NOT EXISTS public.workloads
(
    id serial PRIMARY KEY,
    name text NOT NULL UNIQUE,
    tool text NOT NULL DEFAULT 'tsbs',
    description text,
    dataset_params jsonb NOT NULL DEFAULT '{}'::jsonb,
    query_params jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE public.experiments
    DROP CONSTRAINT IF EXISTS experiments_workload_id_fkey;
ALTER TABLE public.experiments
    ADD CONSTRAINT experiments_workload_id_fkey
    FOREIGN KEY (workload_id) REFERENCES public.workloads(id) ON DELETE SET NULL;

ALTER TABLE public.runs
    DROP CONSTRAINT IF EXISTS runs_workload_id_fkey;
ALTER TABLE public.runs
    ADD CONSTRAINT runs_workload_id_fkey
    FOREIGN KEY (workload_id) REFERENCES public.workloads(id) ON DELETE SET NULL;

CREATE TABLE IF NOT EXISTS public.parameter_space
(
    name text PRIMARY KEY,
    value_type text NOT NULL CHECK (value_type IN ('int', 'float', 'bool', 'enum')),
    min_value double precision,
    max_value double precision,
    enum_values jsonb,
    unit text DEFAULT 'none',
    requires_restart boolean NOT NULL DEFAULT false,
    parameter_group text,
    is_timescaledb boolean NOT NULL DEFAULT false,
    enabled boolean NOT NULL DEFAULT true,
    description text
);

CREATE TABLE IF NOT EXISTS public.optimization_sessions
(
    id serial PRIMARY KEY,
    name text NOT NULL,
    algorithm text NOT NULL CHECK (algorithm IN ('random', 'rf_initial', 'ga', 'local_search', 'hybrid')),
    workload_id integer REFERENCES public.workloads(id) ON DELETE SET NULL,
    objective_name text NOT NULL DEFAULT 'qps_latency_score',
    objective_params jsonb NOT NULL DEFAULT '{}'::jsonb,
    top_params jsonb NOT NULL DEFAULT '[]'::jsonb,
    status text NOT NULL DEFAULT 'created',
    best_config_id integer REFERENCES public.configs(id) ON DELETE SET NULL,
    best_score double precision,
    started_at timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS public.optimization_trials
(
    id serial PRIMARY KEY,
    session_id integer NOT NULL REFERENCES public.optimization_sessions(id) ON DELETE CASCADE,
    generation integer NOT NULL DEFAULT 0,
    candidate_index integer NOT NULL DEFAULT 0,
    config_id integer NOT NULL REFERENCES public.configs(id) ON DELETE CASCADE,
    experiment_id integer REFERENCES public.experiments(id) ON DELETE SET NULL,
    run_id integer REFERENCES public.runs(id) ON DELETE SET NULL,
    metrics jsonb NOT NULL DEFAULT '{}'::jsonb,
    score double precision,
    status text NOT NULL DEFAULT 'created',
    error_text text,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(session_id, generation, candidate_index)
);

CREATE TABLE IF NOT EXISTS public.parameter_importances
(
    id serial PRIMARY KEY,
    session_id integer REFERENCES public.optimization_sessions(id) ON DELETE CASCADE,
    metric_name text NOT NULL,
    parameter_name text NOT NULL,
    importance double precision NOT NULL,
    rank integer NOT NULL,
    model_name text NOT NULL DEFAULT 'RandomForestRegressor',
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(session_id, metric_name, parameter_name)
);

-- Агрегированное представление: одна строка = один эксперимент и одна конфигурация.
CREATE OR REPLACE VIEW public.v_experiment_summary AS
SELECT
    e.id AS experiment_id,
    e.name,
    e.stage,
    e.status,
    e.created_at,
    e.config_id,
    c.params,
    c.source AS config_source,
    COUNT(DISTINCT r.id) AS runs_count,
    AVG(rm.rate_qps) AS avg_rate_qps,
    PERCENTILE_CONT(0.5)  WITHIN GROUP (ORDER BY rm.q50_ms)  AS median_q50_ms,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY rm.q95_ms) AS p95_q95_ms,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY rm.q99_ms) AS p99_q99_ms,
    AVG(rm.q95_ms) AS avg_q95_ms,
    AVG(rm.q99_ms) AS avg_q99_ms
FROM public.experiments e
JOIN public.configs c ON c.id = e.config_id
LEFT JOIN public.runs r ON r.experiment_id = e.id
LEFT JOIN public.run_metrics rm ON rm.run_id = r.id
GROUP BY e.id, e.name, e.stage, e.status, e.created_at, e.config_id, c.params, c.source;

CREATE INDEX IF NOT EXISTS ix_experiments_config_id ON public.experiments(config_id);
CREATE INDEX IF NOT EXISTS ix_runs_experiment_id ON public.runs(experiment_id);
CREATE INDEX IF NOT EXISTS ix_run_metrics_run_id ON public.run_metrics(run_id);
CREATE INDEX IF NOT EXISTS ix_optimization_trials_session_id ON public.optimization_trials(session_id);

-- Дополнительные таблицы под фактический функционал специального раздела:
-- LHS-выборка, суррогатные модели и шаги локального градиентного уточнения.
ALTER TABLE public.optimization_sessions
    DROP CONSTRAINT IF EXISTS optimization_sessions_algorithm_check;
ALTER TABLE public.optimization_sessions
    ADD CONSTRAINT optimization_sessions_algorithm_check
    CHECK (algorithm IN ('random', 'rf_initial', 'ga', 'local_search', 'hybrid', 'nn_gradient'));

CREATE TABLE IF NOT EXISTS public.sampling_designs
(
    id serial PRIMARY KEY,
    session_id integer REFERENCES public.optimization_sessions(id) ON DELETE CASCADE,
    method text NOT NULL CHECK (method IN ('random', 'lhs')),
    samples_count integer NOT NULL,
    parameter_count integer NOT NULL,
    seed integer,
    params jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.surrogate_models
(
    id serial PRIMARY KEY,
    session_id integer REFERENCES public.optimization_sessions(id) ON DELETE SET NULL,
    model_type text NOT NULL CHECK (model_type IN ('random_forest', 'mlp_regressor')),
    target_metric text NOT NULL DEFAULT 'qps_latency_score',
    train_rows integer NOT NULL DEFAULT 0,
    feature_names jsonb NOT NULL DEFAULT '[]'::jsonb,
    hyperparams jsonb NOT NULL DEFAULT '{}'::jsonb,
    train_score double precision,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS public.local_search_steps
(
    id serial PRIMARY KEY,
    session_id integer REFERENCES public.optimization_sessions(id) ON DELETE CASCADE,
    source_trial_id integer REFERENCES public.optimization_trials(id) ON DELETE SET NULL,
    step_no integer NOT NULL,
    config_id integer REFERENCES public.configs(id) ON DELETE SET NULL,
    predicted_score double precision,
    actual_score double precision,
    gradient_norm double precision,
    accepted boolean DEFAULT false,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_surrogate_models_session_id ON public.surrogate_models(session_id);
CREATE INDEX IF NOT EXISTS ix_local_search_steps_session_id ON public.local_search_steps(session_id);

-- ─────────────────────────────────────────────────────────────────────────────
-- Таблицы мониторинга контейнеров и СУБД (добавлены в рамках расширения сервиса)
-- ─────────────────────────────────────────────────────────────────────────────

-- Агрегированные метрики Docker-контейнеров (CPU, RAM, сетевой и дисковый I/O)
-- за время выполнения одного эксперимента
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
-- snapshot_type: 'pre_run' — до бенчмарка, 'post_run' — после
CREATE TABLE IF NOT EXISTS public.experiment_pg_stats
(
    id                  serial PRIMARY KEY,
    experiment_id       integer REFERENCES public.experiments(id) ON DELETE CASCADE,
    snapshot_type       text NOT NULL DEFAULT 'post_run',
    stats_json          jsonb NOT NULL DEFAULT '{}',
    db_size_mb          double precision,
    cache_hit_ratio     double precision,
    active_connections  integer,
    created_at          timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_exp_pg_stats_exp
    ON public.experiment_pg_stats(experiment_id);

-- Представление: прогресс ГА по поколениям (лучшие метрики в каждом поколении)
CREATE OR REPLACE VIEW public.v_ga_generation_progress AS
SELECT
    ot.session_id,
    ot.generation,
    COUNT(*)                                                             AS trials_count,
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

-- Представление: лучший результат каждой сессии оптимизации
CREATE OR REPLACE VIEW public.v_session_best AS
SELECT
    os.id            AS session_id,
    os.name          AS session_name,
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
LEFT JOIN public.v_experiment_summary vs
    ON vs.config_id = os.best_config_id
    AND vs.experiment_id = (
        SELECT ot2.experiment_id
        FROM public.optimization_trials ot2
        WHERE ot2.session_id = os.id
          AND ot2.score = os.best_score
        ORDER BY ot2.id DESC LIMIT 1
    )
ORDER BY os.id DESC;