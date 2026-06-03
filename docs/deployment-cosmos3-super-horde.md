# Cosmos3 Super Reasoner Horde Deployment

This deployment runs the full Cosmos Evaluator service set with the VLM-backed
checks pointed at a local Cosmos3 Super Reasoner NIM.

## Runtime

Cosmos3 Reasoner NIM 1.7.0 uses one container image for both sizes:

- Image: `nvcr.io/nim/nvidia/cosmos3-reasoner:1.7.0`
- Super size: `NIM_MODEL_SIZE=super`
- Served model: `nvidia/cosmos3-super-reasoner`

The evaluator endpoint key is `cosmos3-super-reasoner`, and the default VLM
base URL inside Docker is `http://cosmos3-nim:8000/v1`.

## Ports

| Component | Host port |
|---|---:|
| Cosmos3 NIM | 8000 |
| Obstacle correspondence | 8082 |
| VLM preset | 8083 |
| Hallucination | 8085 |
| Attribute verification | 8086 |
| VLM switch/control API | 8090 |

The control API is the VLM service exposed on a second host port. Use:

```bash
curl http://localhost:8090/runtime/vlm
curl http://localhost:8090/runtime/vlm/endpoints
curl -X POST http://localhost:8090/runtime/vlm/switch \
  -H 'Content-Type: application/json' \
  -d '{"endpoint":"cosmos3-super-reasoner"}'
```

## Launch

On `horde@10.63.158.169`, create protected env files:

```bash
mkdir -p ~/.cosmos_evaluator
chmod 700 ~/.cosmos_evaluator
$EDITOR ~/.cosmos_evaluator/nim.env       # NGC_API_KEY=...
$EDITOR ~/.cosmos_evaluator/evaluator.env # see deploy/horde/cosmos3_evaluator.env.example
chmod 600 ~/.cosmos_evaluator/nim.env ~/.cosmos_evaluator/evaluator.env
```

Build or load the customized evaluator images, then run:

```bash
bash deploy/horde/build_cosmos3_evaluator_images.sh
bash deploy/horde/launch_cosmos3_evaluator.sh
bash deploy/horde/smoke_cosmos3_evaluator.sh
```

The obstacle image requires `checks/utils/citysemsegformer.onnx`; the build
helper skips that image and prints the exact follow-up command when the model is
not present.

The VLM preset and attribute verification services read the same runtime state
file, mounted at `/runtime/vlm_runtime.json`, when a request does not provide an
explicit VLM override.
