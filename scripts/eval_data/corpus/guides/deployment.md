# Deployment Guide

## Kubernetes

Apply the manifests with kustomize:

```bash
kubectl apply -k deploy/overlays/production
kubectl rollout status deployment/helios
```

```yaml
resources:
  requests:
    cpu: "2"
    memory: 4Gi
```

## Secrets Management

API tokens and database passwords are injected from Vault by the agent sidecar. Rotate
them every thirty days; the process re-reads the mounted files on `SIGHUP`, so rotation
needs no restart.

## Rolling Back

Run `helios migrate down --steps 1`, then redeploy the previous image tag. Migrations are
backwards compatible for exactly one release.

## Observability

Dashboards live in Grafana under *Helios / Ingest*. Alerts fire on consumer lag, flush
latency and error-budget burn rate.
