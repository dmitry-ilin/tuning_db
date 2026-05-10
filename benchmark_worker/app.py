from __future__ import annotations

import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import psycopg2
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from benchmark.benchmark_db_save import TSBSRunner
from config.config_reader import TS_Config
from tsdb_tuner.params import load_param_space, repair_config

try:
    import docker as docker_sdk
except Exception:  # pragma: no cover
    docker_sdk = None

load_dotenv()

app = FastAPI(title="TSDB Tuner Benchmark Worker", version="1.0.0")

_ALLOWED_PARAM = re.compile(r"^[A-Za-z0-9_.]+$")


class RunRequest(BaseModel):
    experiment_id: int = Field(gt=0)
    config_id: int = Field(gt=0)
    config: dict[str, Any]
    run_number: int = 1


def _dsn_from_env() -> tuple[str, str]:
    target = os.getenv("TARGET_DB_DSN")
    results = os.getenv("RESULTS_DB_DSN")
    return target, results


def _load_project_config() -> dict[str, Any]:
    path = Path(os.getenv("PROJECT_CONFIG", "config/config.yml"))
    if not path.exists():
        raise RuntimeError(f"Не найден config/config.yml: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _format_for_postgres(config: dict[str, Any]) -> dict[str, str]:
    param_space = Path(os.getenv("PARAM_SPACE", "config/param_space.yml"))
    specs = {spec.name: spec for spec in load_param_space(param_space)}
    formatted: dict[str, str] = {}
    for name, value in repair_config(config).items():
        spec = specs.get(name)
        formatted[name] = spec.format_for_postgres(value) if spec else str(value)
    return formatted


def _apply_alter_system(pg_config: dict[str, str], target_dsn: str) -> bool:
    """Применение конфигурации через ALTER SYSTEM. Возвращает True, если нужен restart."""
    specs = {spec.name: spec for spec in load_param_space(os.getenv("PARAM_SPACE", "config/param_space.yml"))}
    need_restart = False
    conn = psycopg2.connect(target_dsn)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            for name, value in pg_config.items():
                if not _ALLOWED_PARAM.match(name):
                    raise RuntimeError(f"Недопустимое имя параметра: {name}")
                cur.execute(f"ALTER SYSTEM SET {name} = %s", (value,))
                need_restart = need_restart or bool(specs.get(name) and specs[name].restart)
            cur.execute("SELECT pg_reload_conf();")
    finally:
        conn.close()
    return need_restart


def _wait_target_db(target_dsn: str, timeout: int = 90) -> None:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            conn = psycopg2.connect(target_dsn)
            conn.close()
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(2)
    raise RuntimeError(f"TimescaleDB не стала доступна за {timeout} секунд: {last_error}")


def _restart_target_container() -> None:
    """Перезапуск TimescaleDB через Docker socket. Нужен только для restart-параметров."""
    container_name = os.getenv("TARGET_CONTAINER_NAME", "tsdb-timescaledb")
    if docker_sdk is None:
        raise RuntimeError("Python-пакет docker недоступен; нельзя перезапустить контейнер")
    client = docker_sdk.from_env()
    container = client.containers.get(container_name)
    container.restart(timeout=30)


def _apply_config(pg_config: dict[str, str], target_dsn: str) -> None:
    """
    Режимы:
      BENCHMARK_APPLY_MODE=alter_system  — универсальный режим для Docker-стенда;
      BENCHMARK_APPLY_MODE=ts_config     — использовать твой TS_Config.update_postgresql_conf();
      BENCHMARK_APPLY_MODE=none          — ничего не применять.
    """
    mode = os.getenv("BENCHMARK_APPLY_MODE", "alter_system").strip().lower()
    restart_mode = os.getenv("BENCHMARK_RESTART_MODE", "docker").strip().lower()

    if mode == "none":
        return

    if mode == "ts_config":
        ts_config = TS_Config()
        ts_config.update_postgresql_conf(pg_config)
        need_restart = True
    elif mode == "alter_system":
        need_restart = _apply_alter_system(pg_config, target_dsn)
    else:
        raise RuntimeError(f"Неизвестный BENCHMARK_APPLY_MODE={mode}")

    if need_restart and restart_mode == "docker":
        _restart_target_container()
        _wait_target_db(target_dsn)
    elif need_restart and restart_mode == "reload":
        # Не все параметры применятся через reload, но этот режим удобен для быстрой отладки.
        conn = psycopg2.connect(target_dsn)
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT pg_reload_conf();")
        finally:
            conn.close()
    else:
        time.sleep(int(os.getenv("BENCHMARK_SETTLE_SECONDS", "2")))


def _portable_run_query_benchmark(runner: TSBSRunner, db_config_params: dict[str, str], run_number: int) -> dict[str, Any]:
    """
    Docker-aware версия run_query_benchmark().
    В твоем исходном файле host/port БД были зашиты как localhost:5433.
    Здесь они берутся из переменных окружения, поэтому метод работает внутри compose-сети.
    """
    all_metrics: dict[str, Any] = {}
    queries_file = runner.generate_queries()

    target_host = os.getenv("TARGET_DB_HOST")
    target_port = os.getenv("TARGET_DB_PORT")
    target_user = os.getenv("TARGET_DB_USER")
    target_password = os.getenv("TARGET_DB_PASSWORD")
    target_name = os.getenv("TARGET_DB_NAME")
    timeout_seconds = int(os.getenv("TSBS_COMMAND_TIMEOUT"))

    Path(runner.results_dir).mkdir(parents=True, exist_ok=True)
    Path(runner.queries_dir).mkdir(parents=True, exist_ok=True)

    for query_type, qfile in queries_file.items():
        results_file = Path(runner.results_dir) / f"run_{run_number}_{query_type}_{int(time.time())}.json"
        run_cmd = [
            str(Path(runner.bin_path) / "tsbs_run_queries_timescaledb"),
            "--hosts", target_host,
            "--port", str(target_port),
            "--user", target_user,
            "--pass", target_password,
            "--db-name", target_name,
            "--workers", str(runner.workers),
            "--print-interval", "0",
            "--results-file", str(results_file),
            "--file", str(qfile),
        ]
        try:
            completed = subprocess.run(run_cmd, check=True, capture_output=True, text=True, timeout=timeout_seconds)
            if not results_file.exists():
                raise RuntimeError(f"TSBS не создал файл результатов: {results_file}\nSTDOUT={completed.stdout}\nSTDERR={completed.stderr}")

            run_id = runner.save_run_results(
                query_type=query_type,
                results_file=results_file,
                run_number=run_number,
                db_config_params=db_config_params,
            )
            metrics = runner._parse_json_results(results_file, query_type)
            metrics["run_id"] = run_id
            all_metrics[query_type] = metrics

            if os.getenv("KEEP_TSBS_RESULTS", "0") != "1":
                try:
                    results_file.unlink()
                except Exception:
                    pass
        except Exception as exc:  # noqa: BLE001
            all_metrics[query_type] = {"error": str(exc)}
            raise
    return all_metrics


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "tsbs-runner"}


# @app.post("/run")
# def run(req: RunRequest) -> dict[str, Any]:
#     started_at = datetime.now().isoformat()

#     try:
#         runtime_dir = Path(os.getenv("RUNTIME_DIR", "/app/runtime"))
#         runtime_dir.mkdir(parents=True, exist_ok=True)

#         config_file = runtime_dir / f"config_exp{req.experiment_id}_cfg{req.config_id}.json"
#         config_file.write_text(
#             json.dumps(req.config, ensure_ascii=False, indent=2),
#             encoding="utf-8",
#         )

#         cmd = [
#             "python",
#             "-m",
#             "benchmark.run_tsbs",
#             "--experiment-id", str(req.experiment_id),
#             "--config-id", str(req.config_id),
#             "--config-json-file", str(config_file),
#             "--project-config", os.getenv("PROJECT_CONFIG", "/app/config/config.yml"),
#             "--param-space", os.getenv("PARAM_SPACE", "/app/config/param_space.yml"),
#             "--results-dsn", os.getenv("RESULTS_DB_DSN"),
#             "--run-number", str(req.run_number),
#             "--apply-with-ts-config",
#         ]

#         if os.getenv("BENCHMARK_RESTART", "1") == "1":
#             cmd.append("--restart")
#         elif os.getenv("BENCHMARK_RELOAD", "0") == "1":
#             cmd.append("--reload")

#         completed = subprocess.run(
#             cmd,
#             cwd="/app",
#             capture_output=True,
#             text=True,
#             timeout=int(os.getenv("BENCHMARK_TIMEOUT", "1800")),
#         )

#         if completed.returncode != 0:
#             raise RuntimeError(
#                 f"benchmark.run_tsbs failed code={completed.returncode}\n"
#                 f"STDOUT:\n{completed.stdout}\n"
#                 f"STDERR:\n{completed.stderr}"
#             )

#         return {
#             "status": "ok",
#             "started_at": started_at,
#             "finished_at": datetime.now().isoformat(),
#             "experiment_id": req.experiment_id,
#             "config_id": req.config_id,
#             "stdout": completed.stdout,
#             "stderr": completed.stderr,
#         }

#     except Exception as exc:
#         raise HTTPException(
#             status_code=500,
#             detail={
#                 "status": "failed",
#                 "started_at": started_at,
#                 "finished_at": datetime.now().isoformat(),
#                 "experiment_id": req.experiment_id,
#                 "config_id": req.config_id,
#                 "error": str(exc),
#             },
#         ) from exc
@app.post("/run")
def run(req: RunRequest) -> dict[str, Any]:
    target_dsn, results_dsn = _dsn_from_env()
    started_at = datetime.now().isoformat()
    try:
        pg_config = _format_for_postgres(req.config)
        _apply_config(pg_config, target_dsn)
        _wait_target_db(target_dsn)

        project_cfg = _load_project_config()
        runner = TSBSRunner(project_cfg)

        Path(runner.results_dir).mkdir(parents=True, exist_ok=True)
        Path(runner.queries_dir).mkdir(parents=True, exist_ok=True)

        runner.connect_results_db(results_dsn)
        runner.current_experiment_id = req.experiment_id
        runner.config_id = req.config_id

        metrics = _portable_run_query_benchmark(runner, pg_config, req.run_number)
        return {
            "status": "ok",
            "started_at": started_at,
            "finished_at": datetime.now().isoformat(),
            "experiment_id": req.experiment_id,
            "config_id": req.config_id,
            "applied_config": pg_config,
            "metrics": metrics,
        }
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail={
            "status": "failed",
            "started_at": started_at,
            "finished_at": datetime.now().isoformat(),
            "experiment_id": req.experiment_id,
            "config_id": req.config_id,
            "error": str(exc),
        }) from exc


