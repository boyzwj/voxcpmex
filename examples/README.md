# examples/README.md

# VoxCPMEx Examples

Run from the project root with `mix run`:

```bash
mix run examples/hello_world.exs
mix run examples/voice_design.exs
mix run examples/voice_cloning.exs --reference voice.wav
mix run examples/streaming.exs
```

Available scripts:

| Script | Description |
|--------|-------------|
| `hello_world.exs` | Basic text-to-speech with configurable quality |
| `voice_design.exs` | Create a voice from a text description — **no reference audio needed** |
| `voice_cloning.exs` | Clone a voice from reference audio |
| `streaming.exs` | Streaming generation with metadata output |

All examples accept `--device` (`cuda`/`cpu`/`mps`), `--steps`, `--cfg`, and `--output`.
