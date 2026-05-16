# VoxCPMEx

Elixir wrapper for [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) — 2B-param, 30-language, 48kHz TTS.

Architecture: Erlang Port to a Python subprocess, communicating via binary-framed MessagePack.

## Commands

```bash
mix setup                    # mix voxcpmex.setup (installs Python deps)
mix voxcpmex.setup --cpu     # CPU-only Python deps
mix voxcpmex.setup --mps     # Apple Silicon Python deps
mix voxcpmex.setup --venv .venv  # with virtual env
mix test                     # unit tests (no Python process needed)
mix docs                     # generate ExDoc docs
mix format                   # format Elixir source
```

## Key structure

- `lib/voxcpmex.ex` — public API module (thin delegates to Server)
- `lib/voxcpmex/server.ex` — GenServer managing the Python Port + message dispatch (572 lines, the core)
- `lib/voxcpmex/application.ex` — empty OTP application (users start their own GenServers)
- `lib/mix/tasks/voxcpmex.setup.ex` — Python dependency installer via pip
- `priv/python/voxcpmex_bridge.py` — Python bridge with VoxCPM2 model (310 lines)
- `test/voxcpmex_test.exs` — compile-only smoke tests (no model needed)

## Protocol (v2.1, binary-framed MessagePack)

Frame: `[4-byte BE total_length][msgpack payload]`

Streaming: single active stream per GenServer. Python emits `stream_start` → N×`stream_chunk` → `stream_end` in strict sequence. No stream IDs.

## Gotchas

- `Msgpax.pack!()` returns **iodata**, not a binary — wrap with `IO.iodata_to_binary/1`
- Erlang Port I/O: frames can arrive split across Port messages, so the parser accumulates partial frames in a buffer
- The bridge patches `torch.load` to default `map_location="cpu"` for safe weight loading
- `mix test` runs without Python — only checks module exports and compilation
- Python deps: `voxcpm`, `msgpack`, `soundfile`, `torch`+`torchaudio`
- LoRA operations (`load_lora`/`unload_lora`) are synchronous call/response
- Streams auto-cleanup after 60s TTL (`@stream_ttl_ms 60_000`)
- All telemetry emitted under `[:voxcpmex, event]` via `:telemetry`

## Style

- Formatter: standard `.formatter.exs` inputs
- No credo, dialyzer, or lint config
- All public API functions delegate to `Server` module
