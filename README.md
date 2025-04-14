# Kaze's MLOps stack

This is an evolving stack of tool I uses for my ML workflow. It is going to be very opinionated, not necessarily always up to date (considering the pace of the field, I think this is fair), and not always the best choice for your use case. I choose this stack because it fits my philosophy and my workflow.

The main purpose of this repo is to serve as a reference to retrace my steps whenever I need, instead of a template which I just copy and deploy to the next project. I have no intention of making this a full fledge library.

## Stack

- **Machine learning framework**: [Jax](https://jax.readthedocs.io/en/latest/) (with [Flax](https://flax.readthedocs.io/en/latest/) for neural networks)
- **Hyperparameter tuning** : [Optax](https://optax.readthedocs.io/en/latest/) (for optimizers) + [Optuna](https://optuna.org/) (for hyperparameter tuning)
- **Object storage**: [MinIO](https://min.io/) (for storing data and models)
- **Database**: [PostgreSQL](https://www.postgresql.org/) (for storing metadata and results)
- **Experiment tracking**: [MLflow](https://mlflow.org/) (for tracking experiments and models)
- **Data versioning**: [DVC](https://dvc.org/) (for data versioning and pipelines)
- **Orchestration**: [Dagster](https://dagster.io/) (for orchestrating the pipeline)
- **Deployment**: [BentoML](https://docs.bentoml.org/en/latest/) (for deploying the model as a service)
- **Monitoring**: [Prometheus](https://prometheus.io/) (for monitoring the service) + [Grafana](https://grafana.com/) (for visualizing the metrics)