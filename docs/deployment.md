# Deployment

This guide covers how to build, tag, push, and run Cosmos Evaluator service containers for production environments. Each checker runs as an independent container exposing a REST API. The default port is configurable via the `UVICORN_PORT` environment variable.

## Services Overview

| Service | Image Tag | GPU Required | Default Port |
|---------|-----------|:------------:|:------------:|
| Obstacle Correspondence | `obstacle-correspondence:1.15.0` | Yes | 8000 |
| VLM Preset | `vlm:1.15.0` | No | 8000 |
| Hallucination | `hallucination-checker:1.0.0` | No | 8080 |
| Attribute Verification | `attribute-verification-checker:1.0.0` | No | 8080 |

## Building Container Images

Set up the build environment, then use `dazel run` to build and load each image into the local Docker daemon:

```bash
source build/env_setup.sh

dazel run //services/obstacle_correspondence:image_load
dazel run //services/vlm:image_load
dazel run //services/hallucination:image_load
dazel run //services/attribute_verification:image_load
```

For detailed build and run instructions per service, see the individual checker documentation:

| Service | Documentation |
|---------|---------------|
| Obstacle Correspondence | [Obstacle Check — Docker Container](checks/obstacle.md#docker-container) |
| VLM Preset | [VLM Preset Check — Docker Container](checks/vlm-preset.md#docker-container) |
| Hallucination | [Hallucination Check](checks/hallucination.md) |
| Attribute Verification | [Attribute Verification Check](checks/attribute-verification.md) |

## Tagging and Pushing to a Registry

After building and loading the images locally, tag and push them to any OCI-compatible container registry:

```bash
# Tag the images for your registry
docker tag obstacle-correspondence:1.15.0 your-registry.example.com/cosmos-evaluator/obstacle-correspondence:1.15.0
docker tag vlm:1.15.0 your-registry.example.com/cosmos-evaluator/vlm:1.15.0
docker tag hallucination-checker:1.0.0 your-registry.example.com/cosmos-evaluator/hallucination-checker:1.0.0
docker tag attribute-verification-checker:1.0.0 your-registry.example.com/cosmos-evaluator/attribute-verification-checker:1.0.0

# Push to the registry
docker push your-registry.example.com/cosmos-evaluator/obstacle-correspondence:1.15.0
docker push your-registry.example.com/cosmos-evaluator/vlm:1.15.0
docker push your-registry.example.com/cosmos-evaluator/hallucination-checker:1.0.0
docker push your-registry.example.com/cosmos-evaluator/attribute-verification-checker:1.0.0
```

Replace `your-registry.example.com/cosmos-evaluator` with your actual registry path (e.g., an ECR, GCR, ACR, or NVCR endpoint).

## Environment Configuration for Production

Please add the following environment variables to a `~/.cosmos_evaluator/.env` file for your deployment environment configuration:

### Port Configuration

The `UVICORN_PORT` environment variable controls the port the service listens on inside the container. This is useful when running multiple services on the same host or behind a load balancer:

```bash
docker run -e UVICORN_PORT=9000 -p 9000:9000 vlm:1.15.0
```

If not set, the service uses its default port (8000 or 8080 depending on the service).

### Storage Provider

The Obstacle Correspondence and VLM Preset checks use S3 as the storage backend for staging or production:

| Variable | Description |
|----------|-------------|
| `COSMOS_EVALUATOR_ENV` | Set to `production` or `staging` |
| `COSMOS_EVALUATOR_STORAGE_TYPE` | Storage backend: `s3` (default) or `local` |
| `COSMOS_EVALUATOR_STORAGE_BUCKET` | S3 bucket name |
| `COSMOS_EVALUATOR_STORAGE_REGION` | AWS region |
| `COSMOS_EVALUATOR_STORAGE_ACCESS_KEY` | AWS access key |
| `COSMOS_EVALUATOR_STORAGE_SECRET_KEY` | AWS secret key |

> **Note:** `COSMOS_EVALUATOR_STORAGE_TYPE=local` is not recommended for cloud deployments.

### API Keys

The VLM Preset and Attribute Verification checks require API keys for their model endpoints:

| Variable | Used By | Description |
|----------|---------|-------------|
| `BUILD_NVIDIA_API_KEY` | VLM Preset, Attribute Verification | API key for LLM/VLM endpoints ([build.nvidia.com](https://build.nvidia.com/)) |

### Multistorage Configuration

The Hallucination and Attribute Verification checks use [NVIDIA Multistorage](https://nvidia.github.io/multi-storage-client/references/configuration.html) to access video inputs from cloud storage. Configure it via the `MULTISTORAGE_CONFIG` environment variable in `~/.cosmos_evaluator/.env`.

| Variable | Used By | Description |
|----------|---------|-------------|
| `MULTISTORAGE_CONFIG` | Hallucination, Attribute Verification | JSON configuration for storage backend connectivity |

The value is a JSON object that defines storage profiles and path mappings.

When setting this in the `.env` file, compact the JSON onto a single line and wrap it in single quotes to avoid parsing issues.

For more information, see the [Multistorage Configuration Example](getting-started.md#multistorage-configuration-example).

## Running in Production

### Obstacle Correspondence (GPU required)

Set `OBJECTS_PROCESS_CONCURRENCY_LIMIT` to control the maximum number of concurrent `/process` requests (default: 3) based on the load your cloud machine can handle concurrently. Requests beyond this limit are rejected immediately with a 503 status.

```bash
docker run \
  --gpus all \
  --env-file ~/.cosmos_evaluator/.env \
  -e OBJECTS_PROCESS_CONCURRENCY_LIMIT=1 \
  -p 8000:8000 \
  obstacle-correspondence:1.15.0
```

> Requires [Storage Provider](#storage-provider) configuration in the env file.

### VLM Preset

```bash
docker run \
  --env-file ~/.cosmos_evaluator/.env \
  -p 8000:8000 \
  vlm:1.15.0
```

> Requires [`BUILD_NVIDIA_API_KEY`](#api-keys) and [Storage Provider](#storage-provider) configuration in the env file.

### Hallucination

```bash
docker run \
  --env-file ~/.cosmos_evaluator/.env \
  -p 8080:8080 \
  hallucination-checker:1.0.0
```

> Requires [`MULTISTORAGE_CONFIG`](#multistorage-configuration) in the env file.

### Attribute Verification

```bash
docker run \
  --env-file ~/.cosmos_evaluator/.env \
  -p 8080:8080 \
  attribute-verification-checker:1.0.0
```

> Requires [`BUILD_NVIDIA_API_KEY`](#api-keys) and [`MULTISTORAGE_CONFIG`](#multistorage-configuration) in the env file.

### Verifying the Service

All services expose a `/health` endpoint for liveness and readiness checks:

```bash
curl http://localhost:8000/health
```

Each service also provides a Swagger UI at `/docs` for browsing the API documentation and testing endpoints from a browser:

```
http://localhost:8000/docs
```

## Cosmos3 Super Reasoner on Horde

For a full-stack Horde deployment with a local Cosmos3 Super Reasoner NIM and
runtime VLM switch endpoints, see
[Cosmos3 Super Reasoner Horde Deployment](deployment-cosmos3-super-horde.md).
