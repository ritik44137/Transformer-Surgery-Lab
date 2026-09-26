# Dashboard

Streamlit page over `runs/`. Loss curves, throughput and latency bars, and a card per selected run.

```bash
make dashboard
# http://localhost:8501
```

Same thing without the compose wrapper:

```bash
docker compose -f docker/docker-compose.yml --profile dashboard up dashboard
```

Host Python, after `pip install -r requirements.txt`:

```bash
PYTHONPATH=src streamlit run dashboard/app.py
```

A run shows up if it has `summary.json` or `config_resolved.yaml`. The charts want `metrics_train.jsonl` and, for the bars, `benchmark.json` from `make benchmark RUN_DIR=runs/<run_name>`. `metrics_eval.jsonl` and `samples.json` are optional.
