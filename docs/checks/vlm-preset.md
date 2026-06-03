# VLM Preset Check

The VLM Preset check uses a Vision-Language Model to verify that a generated video matches specified environment conditions. It answers the question: *"Does this video look like the scene that was requested?"*

## How It Works

1. **Keyframe extraction** — Frames are sampled from the video at regular intervals (default: every 2 seconds) and resized for efficient processing.
2. **Prompt construction** — A structured prompt is built from the preset conditions and a prompt template, asking the VLM to evaluate each condition.
3. **VLM evaluation** — The keyframes and prompt are sent to a Vision-Language Model endpoint.
4. **Score parsing** — The VLM response is parsed into per-condition scores with explanations.

Each condition is scored independently:

| Score | Meaning |
|-------|---------|
| **1.0** | Clearly true — the condition is visibly consistent with the video |
| **0.5** | Ambiguous — insufficient or conflicting visual evidence |
| **0.0** | Clearly false — the condition contradicts what is shown in the video |

The **overall score** is the arithmetic mean of all per-condition scores.

## Input Requirements

### Video File

Any video format supported by OpenCV (MP4, MOV, AVI, etc.). The check extracts keyframes at the configured interval, so longer videos produce more frames for evaluation.

### Preset Conditions

A JSON object specifying the expected environment. Currently, the **environment** preset is supported with four required conditions:

```json
{
  "name": "environment",
  "weather": "Clear Sky",
  "time_of_day_illumination": "Morning",
  "region_geography": "Dense City Center",
  "road_surface_conditions": "Dry"
}
```

### Valid Taxonomy Values

| Field Name | Possible Values |
|---|---|
| name | environment |
| weather | Clear Sky, Partly Cloudy, Overcast, Light Rain, Heavy Rain, Light Snow, Heavy Snow, Fog |
| time_of_day / Illumination | Morning, Mid-day, Afternoon, Dusk, Night |
| region_geography | Dense City Center, Suburban Area, Residential Area, Rural/Countryside, Highway/Freeway, Tunnels/Bridges, Hilly/Mountainous/Forest, Coastal, Desert |
| road_surface_conditions | Dry, Wet, Slippery, Damaged, Ignorable road debris |

### Deployed VLM Endpoint
We recommend [deploying the Cosmos Reason 2 NIM](https://build.nvidia.com/nvidia/cosmos-reason2-8b/deploy) for optimal results. See [Endpoint Configuration](#endpoint-configuration) for how to add a custom VLM endpoint.
> NOTE: If using the VLM Checker container, you can either [rebuild the container](#build-the-container) or use a volume mount to override `endpoints.json` at runtime — see [Using Your Own Endpoint](#using-your-own-endpoint).

## Usage

### CLI

The CLI operates on local filesystem paths only — all inputs (video file, preset JSON) must be available on disk.

First, create a preset conditions JSON file:

```bash
cat > preset.json << 'EOF'
{
  "name": "environment",
  "weather": "Clear Sky",
  "time_of_day_illumination": "Morning",
  "region_geography": "Dense City Center",
  "road_surface_conditions": "Dry"
}
EOF
```

Run the VLM preset check (if using repo sample data, run `git lfs pull` first; otherwise you may see “moov atom not found” because the files are LFS pointers):

```bash
dazel run //checks/vlm:run -- local \
  --video_path checks/sample_data/cosmos_public/01ce78ad-9e9a-4df9-95d1-1d50e41a04ce_764657799000_764677799000_0_Morning.30fps.mp4 \
  --clip_id 01ce78ad-9e9a-4df9-95d1-1d50e41a04ce_764657799000_764677799000 \
  --preset_file $(pwd)/preset.json \
  --output_dir $(pwd)/output_dir
```

**Arguments:**

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--video_path` | Yes | — | Path to the input video file |
| `--clip_id` | Yes | — | Clip identifier used for naming the output file |
| `--preset_file` | Yes | — | Path to a JSON file containing preset conditions (see [Preset Conditions](#preset-conditions)) |
| `--output_dir` | Yes | — | Directory to write the output JSON result |
| `--verbose` | No | `INFO` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

### Docker Container

#### Build the Container

```bash
dazel run //services/vlm:image_load
```

This builds the OCI image and loads it into the local Docker daemon as `vlm:<version>`.

#### Setup Environment Variables

The VLM check requires an API key for the Vision-Language Model endpoint. To get a free API key for testing, go to the model page on [build.nvidia.com](https://build.nvidia.com/) (e.g. [Qwen 3.5-397B](https://build.nvidia.com/qwen/qwen3.5-397b-a17b)), click **View Code**, then click **Generate API Key**. The key starts with `nvapi-`.

> **Important:** Generate the key from the model page, not from your NGC account settings. An account-level key passes health checks but returns **403** on inference calls. See [Setting up your LLM and VLM API key](../getting-started.md#setting-up-your-llm-and-vlm-api-key) for details and a validation command.

Create a `~/.cosmos_evaluator/.env` file with your API key:

```bash
mkdir -p ~/.cosmos_evaluator
cat > ~/.cosmos_evaluator/.env << 'EOF'
BUILD_NVIDIA_API_KEY=your_api_key_here
EOF
```

See the [Endpoint Configuration](#endpoint-configuration) section for details on configuring VLM endpoints and providers.

#### Start the Service

If using repo sample data, run `git lfs pull` first so the mounted files are real videos, not LFS pointers (otherwise you may see “moov atom not found”).

```bash
# Running in local mode with sample data mounted at /data
docker run --env-file ~/.cosmos_evaluator/.env \
  -e COSMOS_EVALUATOR_STORAGE_TYPE=local \
  -v $(pwd)/checks/sample_data/cosmos_public:/data \
  -p 8000:8000 vlm:1.15.0
```

Verify that the service is running:

```bash
curl http://localhost:8000/health
```

#### Test the Service

```bash
curl -X POST http://localhost:8000/process/preset \
  -H "Content-Type: application/json" \
  -d '{
    "augmented_video_url": "/data/01ce78ad-9e9a-4df9-95d1-1d50e41a04ce_764657799000_764677799000_0_Morning.30fps.mp4",
    "preset_conditions": {
      "name": "environment",
      "weather": "Clear Sky",
      "time_of_day_illumination": "Morning",
      "region_geography": "Dense City Center",
      "road_surface_conditions": "Dry"
    }
  }'
```

**Request fields:**

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `augmented_video_url` | Yes | — | URL or mounted filesystem path to the video file (e.g. `/data/video.mp4`) |
| `preset_conditions` | Yes | — | Preset conditions object (see [Preset Conditions](#preset-conditions) above) |
| `preset_check_config` | No | None | Partial configuration overrides (dict). Merged into the defaults from `checks/config.yaml` — only the fields you specify are overridden. Use the `/config` endpoint to see the defaults. |

**Other endpoints:**

```bash
# Health check
curl http://localhost:8000/health

# Get default configuration
curl http://localhost:8000/config
```

## Configuration

Configuration lives in `checks/config.yaml` under the `av.vlm.preset_check` key:

```yaml
av.vlm:
  preset_check:
    enabled: true
    keyframe_interval_s: 2.0
    keyframe_width: 640
    model:
      endpoint: cosmos3-super-reasoner
      temperature: 0.0
    timeout_seconds: 90
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | `true` | Enable/disable the preset check |
| `keyframe_interval_s` | `2.0` | Seconds between sampled keyframes. Lower values sample more frames (more thorough but slower). |
| `keyframe_width` | `640` | Target width for resized keyframes (aspect ratio preserved). Larger values provide more detail to the VLM but increase latency. |
| `model.endpoint` | `cosmos3-super-reasoner` | VLM endpoint key (see [Endpoint Configuration](#endpoint-configuration)) |
| `model.temperature` | `0.0` | Sampling temperature. 0.0 = deterministic output. |
| `timeout_seconds` | `90` | Request timeout in seconds |

If `preset_check_config.model.endpoint` is omitted from a request, the service
uses the runtime VLM selection stored by `/runtime/vlm/switch`. Request-level
endpoint overrides continue to take precedence.

## Endpoint Configuration

VLM endpoints are defined in `checks/vlm/config/endpoints.json`. Each endpoint specifies:

```json
{
  "endpoint_key": {
    "base_url": "https://your-vlm-endpoint/v1",
    "model": "model-name",
    "env_var": "YOUR_API_KEY_ENV_VAR",
    "timeout": 360
  }
}
```

### Bundled Endpoints

| Key | Model | Provider |
|-----|-------|----------|
| `cosmos3-super-reasoner` | nvidia/cosmos3-super-reasoner | Local Cosmos3 Reasoner NIM (`nvcr.io/nim/nvidia/cosmos3-reasoner:1.7.0`, `NIM_MODEL_SIZE=super`) |
| `cosmos3-nano-reasoner` | nvidia/cosmos3-nano-reasoner | Local Cosmos3 Reasoner NIM (`NIM_MODEL_SIZE=nano`) |
| `qwen3.5-397b-a17b` | qwen/qwen3.5-397b-a17b | [NVIDIA NIM](https://build.nvidia.com/) |

### Using Your Own Endpoint

To add a custom VLM endpoint, add a new entry to `checks/vlm/config/endpoints.json`:

```json
{
  "qwen3.5-397b-a17b": {
    "base_url": "https://integrate.api.nvidia.com/v1",
    "model": "qwen/qwen3.5-397b-a17b",
    "env_var": "BUILD_NVIDIA_API_KEY",
    "timeout": 360
  },
  "my_custom_endpoint": {
    "base_url": "https://my-vlm-service.example.com/v1",
    "model": "my-model-name",
    "env_var": "MY_VLM_API_KEY",
    "timeout": 300
  }
}
```

Then set your API key in `~/.cosmos_evaluator/.env`:

```bash
MY_VLM_API_KEY=your_api_key_here
```

And reference the endpoint in your configuration:

```yaml
av.vlm:
  preset_check:
    model:
      endpoint: my_custom_endpoint
```

The endpoint must be compatible with the [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat) format, as the client uses the `openai` Python library.

#### Container override

When using the prebuilt container, you can override `endpoints.json` with a volume mount instead of rebuilding:

```bash
docker run --env-file ~/.cosmos_evaluator/.env \
  -v $(pwd)/my_endpoints.json:/app/rest_api_deployment.runfiles/_main/checks/vlm/config/endpoints.json:ro \
  -v $(pwd)/checks/sample_data/cosmos_public:/data \
  -p 8000:8000 vlm:1.15.0
```

## Output Format

The REST API returns results in the standard response envelope:

```json
{
  "success": true,
  "data": {
    "result": {
      "environment": {
        "overall_score": 0.875,
        "scoring_details": {
          "weather": {
            "score": 1.0,
            "explanation": "Clear sunny sky visible throughout the video.",
            "preset": "Clear Sky"
          },
          "time_of_day_illumination": {
            "score": 0.5,
            "explanation": "Low-angle sunlight suggests early morning or late afternoon.",
            "preset": "Morning"
          },
          "region_geography": {
            "score": 1.0,
            "explanation": "Dense urban environment with high-rise buildings.",
            "preset": "Dense City Center"
          },
          "road_surface_conditions": {
            "score": 1.0,
            "explanation": "Dry asphalt road surface, no moisture visible.",
            "preset": "Dry"
          }
        },
        "frames_used": 18,
        "processing_time_s": 12.34,
        "model": "Cosmos-Reason2-32B-AV"
      }
    }
  },
  "metadata": {
    "duration_ms": 13450
  },
  "timestamp": "2025-01-01T00:00:00Z"
}
```

### Field Descriptions

| Field | Description |
|-------|-------------|
| `overall_score` | Mean of all per-condition scores (0.0 to 1.0). `null` if the VLM response could not be parsed. |
| `scoring_details.<condition>.score` | Individual condition score: 1.0 (true), 0.5 (ambiguous), or 0.0 (false) |
| `scoring_details.<condition>.explanation` | VLM's reasoning for the score |
| `scoring_details.<condition>.preset` | The original preset value that was evaluated |
| `frames_used` | Number of keyframes extracted and sent to the VLM |
| `processing_time_s` | End-to-end processing time in seconds |
| `model` | Name of the VLM model used |

## Troubleshooting

### API Key Errors

Ensure your API key is set in `~/.cosmos_evaluator/.env` with the correct environment variable name matching the `env_var` field in endpoint configuration.

### Null Scores

If `overall_score` is `null`, the VLM response could not be parsed into the expected format. This can happen with:
- Endpoint timeouts — try increasing `timeout_seconds`
- Model-specific output format issues — try a different endpoint

### Short Videos

Very short videos (< 2 seconds with default settings) may produce only 1 keyframe, reducing evaluation quality. Consider lowering `keyframe_interval_s` for short clips.
