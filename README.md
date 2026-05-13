# lorcana-training

Trains the card encoder, proposal net, and per-step evaluator that power
the Lorcana deck-builder's AI assist, then exports them to ONNX for the
web app. See [`DESIGN.md`](./DESIGN.md).

## Status

Skeleton only. Layout matches `DESIGN.md → "File layout"`.

## Develop

```bash
uv sync
uv run pytest
```
