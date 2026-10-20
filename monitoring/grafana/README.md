# Grafana dashboard setup

1. Start Prometheus using `monitoring/prometheus/prometheus.yml`.
2. Open Grafana and add Prometheus as a data source.
3. Create panels using:
   - Request count rate: `sum(rate(heart_api_requests_total[5m])) by (status)`
   - Request rate by endpoint: `sum(rate(heart_api_requests_total[5m])) by (endpoint)`
   - Approx. p95 latency: `histogram_quantile(0.95, sum(rate(heart_api_request_duration_seconds_bucket[5m])) by (le, endpoint))`
4. Export the dashboard JSON and save it in this folder for submission.
5. Capture screenshots of dashboard panels after generating traffic.

The assignment permits a simple API metrics/logs dashboard; a full Grafana setup is optional if you document and demonstrate another monitoring view.
