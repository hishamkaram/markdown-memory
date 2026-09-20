[![CI](https://img.shields.io/badge/ci-passing-green.svg)](https://example.com/ci)

Orbit Gateway is a small reverse proxy. This summary precedes the first heading.

# Orbit Gateway

Orbit Gateway routes HTTP traffic to upstream services and enforces per-tenant limits.

## Installation

Install the binary with the package manager of your choice.

```bash
# this comment is inside a fence and is not a heading
curl -fsSL https://example.com/install.sh | sh
orbit --version
```

### From Source

Clone the repository and build it:

```bash
git clone https://example.com/orbit.git
make build
```

## Configuration

Orbit reads `orbit.toml` first and then environment variables.

### Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `ORBIT_LISTEN_ADDR` | `0.0.0.0:8080` | Address the proxy binds to |
| `ORBIT_UPSTREAM_TIMEOUT_MS` | `3000` | Upstream request deadline |
| `ORBIT_MAX_INFLIGHT` | `512` | Concurrent requests per tenant |

### Command Line Flags

Flags override the file and the environment.

- `--config <path>` selects another configuration file.
- `--drain-seconds <n>` waits for in-flight requests on shutdown.

## Operations

### Health Checks

`GET /healthz` returns `200` while the process accepts traffic.

### Graceful Shutdown

On `SIGTERM` the proxy stops accepting connections and drains existing ones.

## License

Apache-2.0.
