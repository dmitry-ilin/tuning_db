FROM grafana/otel-lgtm:latest

# Копируем Prometheus конфиг
COPY monitoring/prometheus-lgtm.yml /otel-lgtm/prometheus.yaml

# Копируем Grafana provisioning — datasource и dashboards
COPY monitoring/grafana/provisioning/datasources/prometheus.yml \
     /otel-lgtm/grafana/conf/provisioning/datasources/tuner-prometheus.yml

COPY monitoring/grafana/provisioning/dashboards/dashboards.yml \
     /otel-lgtm/grafana/conf/provisioning/dashboards/tuner-dashboards.yml

# Копируем дашборды в /data — но /data это volume, поэтому копируем в образ
# и при старте копируем в /data через entrypoint
COPY monitoring/grafana/dashboards/ /otel-lgtm/tuner-dashboards/

# Entrypoint: копируем дашборды в /data и запускаем оригинальный run-all.sh
COPY docker/monitoring/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
