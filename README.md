# VoxCPMEx

Elixir wrapper for [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) — a **tokenizer-free, diffusion autoregressive Text-to-Speech** model from OpenBMB.

**2B parameters** · **30 languages** · **48kHz output** · trained on **2M+ hours** of speech data.

[![Hex.pm](https://img.shields.io/badge/hex-v0.3.1-blue)](https://hex.pm/packages/voxcpmex)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Model](https://img.shields.io/badge/HuggingFace-VoxCPM2-orange)](https://huggingface.co/openbmb/VoxCPM2)
[![Protocol](https://img.shields.io/badge/protocol-MessagePack-brightgreen)](https://msgpack.org/)

## ⚡ v0.3.1 — Production Hardening

- **`uv`-based setup** — auto-installs `uv`, uses `uv pip install` for faster dependency management
- **Auto `.venv`** — creates a virtual environment in the project root to handle Debian/Ubuntu externally-managed Python
- **Robust frame parsing** — `parse_frames` skips past non-msgpack data (tqdm progress bars) instead of blocking forever
- **Blocking `await_ready`** — truly blocks until the model finishes loading, instead of returning `{:error, :loading}` immediately
- **Separate stderr** — bridge progress and traceback go directly to terminal, not mixed into the msgpack stream
- **`.venv` auto-detection** — `start_link` finds `.venv/bin/python3` automatically, no manual path needed
- **`:python` option** — explicit Python path for custom environments

## Features

- 🌍 **30 Languages** — Chinese, English, Japanese, Korean, Arabic, French, German, Spanish, Italian, Portuguese, Dutch, Russian, Thai, Vietnamese, Hindi, and 15+ more. Chinese dialects: 四川话, 粤语, 吴语, 东北话, 河南话, 陕西话, 山东话, 天津话, 闽南话
- 🎨 **Voice Design** — Generate a novel voice from a natural-language description alone. *No reference audio needed.*
- 🎛️ **Controllable Cloning** — Clone any voice from a short clip, with optional style guidance (emotion, pace, tone)
- 🎙️ **Ultimate Cloning** — Continuation cloning with transcript for maximum fidelity
- 🔊 **48kHz Studio Output** — AudioVAE V2 built-in super-resolution (16kHz in → 48kHz out)
- ⚡ **Real-Time Streaming** — RTF ~0.3 on RTX 4090, ~0.13 with Nano-VLLM
- 🎓 **LoRA Fine-Tuning** — Adapt with as little as 5–10 minutes of audio
- 📜 **Apache-2.0** — Free for commercial use

## Architecture

VoxCPMEx uses **Erlang Ports** to communicate with a Python process running the VoxCPM2 model. Each `VoxCPMEx.start_link/1` spawns a dedicated Python process, allowing multiple models or instances to run concurrently.

```
+---------------+   msgpack frames   +-----------------+
|    Elixir     | -----------------> |     Python      |
|   GenServer   |                    |   VoxCPM2       |
|               | <----------------- |                 |
+---------------+   msgpack frames   +-----------------+
                     raw audio bytes

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.5.0, CUDA ≥ 12.0 (or Apple Silicon / CPU)
- Elixir ≥ 1.14
- ~8 GB VRAM (GPU recommended)

## Installation

### 1. Add the dependency

```elixir
def deps do
  [
    {:voxcpmex, "~> 0.3.0"}
  ]
end
```

### 2. Install Python dependencies

Uses `uv` for fast dependency management — auto-installs it if missing.
Creates a `.venv` in the project root automatically.

```bash
# CUDA (NVIDIA GPU) — recommended
mix voxcpmex.setup

# Apple Silicon
mix voxcpmex.setup --mps

# CPU-only (no GPU required, slower)
mix voxcpmex.setup --cpu

# Custom virtual environment path
mix voxcpmex.setup --venv /path/to/venv
```

## Quick Start

```elixir
# Start a model server (CUDA GPU)
{:ok, pid} = VoxCPMEx.start_link(device: "cuda")

# Wait for model to load (30-60s on first run, downloads ~8GB)
:ok = VoxCPMEx.await_ready(pid)

# Generate speech
{:ok, audio} = VoxCPMEx.generate(pid, "Hello, world from VoxCPM2!")

# Save to file
:ok = VoxCPMEx.save(audio, "output.wav")
```

## Voice Design 🎨

Create a voice from a **text description** — no reference audio needed. Put the description in parentheses at the start of your text:

```elixir
{:ok, audio} = VoxCPMEx.generate(pid,
  "(A young woman, gentle and sweet voice, warm tone) Hello, welcome to VoxCPM2!"
)
```

Works with any descriptive language:

```
"(A deep male voice, authoritative and confident)"
"(An elderly person, wise and slow-paced)"
"(A cheerful child, energetic and bright)"
"(A calm narrator, suitable for audiobooks)"
"(A robot voice, mechanical and precise)"
"(温柔甜美的少女声音)"          # Chinese descriptions work too!
```

## Voice Cloning 🎛️

### Basic Cloning (reference-only)

```elixir
{:ok, audio} = VoxCPMEx.generate(pid, "This is a cloned voice.",
  audio_prompt: "path/to/reference.wav"
)
```

### Cloning with Style Control

```elixir
{:ok, audio} = VoxCPMEx.generate(pid,
  "(slightly faster, cheerful tone) This clone has style guidance.",
  audio_prompt: "speaker.wav",
  cfg_value: 2.0,
  inference_timesteps: 10
)
```

### Ultimate Cloning (maximum fidelity)

```elixir
{:ok, audio} = VoxCPMEx.generate(pid, "This is ultimate cloning.",
  prompt_wav_path: "speaker.wav",
  prompt_text: "The exact transcript of the reference audio.",
  audio_prompt: "speaker.wav"
)
```

## Multilingual Support 🌍

VoxCPM2 is **tokenizer-free** — just feed text in any supported language, no language tag needed:

```elixir
# Chinese
{:ok, audio} = VoxCPMEx.generate(pid, "你好，今天天气真不错")

# Japanese
{:ok, audio} = VoxCPMEx.generate(pid, "こんにちは、今日はいい天気ですね")

# Korean
{:ok, audio} = VoxCPMEx.generate(pid, "안녕하세요, 오늘 날씨가 참 좋네요")

# French
{:ok, audio} = VoxCPMEx.generate(pid, "Bonjour, le temps est magnifique aujourd'hui")

# Arabic
{:ok, audio} = VoxCPMEx.generate(pid, "مرحبا، الطقس جميل اليوم")
```

## Quality Tuning

| Parameter | Range | Effect |
|-----------|-------|--------|
| `cfg_value` | 1.0–3.0 | Higher = stricter conditioning, less variation. Default: `2.0` |
| `inference_timesteps` | 4–30 | More steps = better detail, slower. Default: `10` |

```elixir
# High quality (more steps, slower)
{:ok, audio} = VoxCPMEx.generate(pid, "Quality matters.",
  inference_timesteps: 30, cfg_value: 3.0
)

# Fast mode (fewer steps)
{:ok, audio} = VoxCPMEx.generate(pid, "Speed matters.",
  inference_timesteps: 4
)
```

## Streaming ⚡ (v2 — true chunk-by-chunk)

```elixir
# Start async streaming — returns immediately
{:ok, ref} = VoxCPMEx.generate_streaming_async(pid, "Long text for streaming...",
  inference_timesteps: 10,
  cfg_value: 2.0
)

# Poll for chunks as they're generated
Stream.unfold(ref, fn ref ->
  case VoxCPMEx.next_chunk(pid, ref) do
    {:ok, chunk} -> {chunk, ref}       # got a chunk
    :eos -> nil                         # stream done
    {:error, reason} -> nil             # error
  end
end)
|> Enum.to_list()

# Or collect all at once
{:ok, ref} = VoxCPMEx.generate_streaming_async(pid, "Long text...")
{:ok, audio} = VoxCPMEx.collect_stream(pid, ref)
:ok = VoxCPMEx.save(audio, "streaming.wav")
```

## Named Servers

```elixir
# Start with a name for easy access
{:ok, _pid} = VoxCPMEx.start_link(device: "cuda", name: MyApp.TTS)

# Use anywhere in your app
{:ok, audio} = VoxCPMEx.generate(MyApp.TTS, "Hello!")
```

## LoRA Fine-Tuning 🎓

```elixir
# Load fine-tuned weights
{:ok, loaded, skipped} = VoxCPMEx.load_lora(pid, "path/to/lora_weights.ckpt")

# Generate with adapted voice
{:ok, audio} = VoxCPMEx.generate(pid, "This uses my fine-tuned voice.")

# Disable LoRA temporarily
:ok = VoxCPMEx.unload_lora(pid)
```

## Configuration

| Option | Description | Default |
|--------|-------------|---------|
| `:model` | HuggingFace model ID | `"openbmb/VoxCPM2"` |
| `:device` | Compute device (`"cuda"`, `"cpu"`, `"mps"`) | `"cuda"` |
| `:load_denoiser` | Load audio denoiser for reference cleanup | `false` |
| `:optimize` | Enable `torch.compile` | `true` |
| `:name` | GenServer name | `nil` |
| `:python` | Python interpreter path (auto-detects `.venv/bin/python3` first) | `nil` |

## Generation Options

| Option | Description | Default |
|--------|-------------|---------|
| `:audio_prompt` | Reference audio for voice cloning | `nil` |
| `:prompt_wav_path` | Prompt audio for continuation cloning | `nil` |
| `:prompt_text` | Transcript of prompt audio | `nil` |
| `:cfg_value` | Guidance scale (1.0–3.0) | `2.0` |
| `:inference_timesteps` | Diffusion steps (4–30) | `10` |
| `:min_len` | Minimum audio length (tokens) | `2` |
| `:max_len` | Maximum token length | `4096` |
| `:normalize` | Text normalization | `false` |
| `:denoise` | Denoise reference audio | `false` |

## Supported Languages

Arabic, Burmese, Chinese (Mandarin + 四川话, 粤语, 吴语, 东北话, 河南话, 陕西话, 山东话, 天津话, 闽南话), Danish, Dutch, English, Finnish, French, German, Greek, Hebrew, Hindi, Indonesian, Italian, Japanese, Khmer, Korean, Lao, Malay, Norwegian, Polish, Portuguese, Russian, Spanish, Swahili, Swedish, Tagalog, Thai, Turkish, Vietnamese

## Hardware

| Device | VRAM | RTF (Speed) |
|--------|------|-------------|
| RTX 4090 | ~8 GB | 0.30 (standard) / 0.13 (Nano-VLLM) |
| RTX 3090 | ~8 GB | ~0.5 estimated |
| Apple M2 Max | Unified | Supported via MPS |
| CPU | N/A | Functional, much slower |

## Troubleshooting

### Bridge exits immediately with status 1

The Python bridge failed to start. Run `mix voxcpmex.setup` to install dependencies. If using a custom Python environment, pass `:python` to `start_link`:

```elixir
{:ok, pid} = VoxCPMEx.start_link(device: "cuda", python: "/path/to/bin/python3")
```

### Bridge process is silent on console

Progress output (tqdm bars, loading messages) goes to the Python process's stderr and appears directly in your terminal. If you see nothing, the bridge may be crashing silently. Check the IEx Logger output for error messages.

### `await_ready` never returns

The model download + loading can take 30-120s on first run (downloads ~8GB). If it takes longer, check whether the bridge process is alive — the Python bridge writes progress to stderr visible in the terminal. If the bridge exited, re-run `mix voxcpmex.setup`.

### "Externally managed" Python error on Debian/Ubuntu

System Python on Debian/Ubuntu doesn't allow `pip install` system-wide. The `mix voxcpmex.setup` task now auto-creates a `.venv` in the project root to avoid this.

## License

Apache-2.0 — free for commercial use. VoxCPM2 model weights are also Apache-2.0.

⚠️ **Ethics:** Strictly forbidden to use for impersonation, fraud, or disinformation. AI-generated content should be clearly labeled.

## Links

- [VoxCPM2 on HuggingFace](https://huggingface.co/openbmb/VoxCPM2)
- [VoxCPM GitHub](https://github.com/OpenBMB/VoxCPM)
- [VoxCPM Docs](https://voxcpm.readthedocs.io)
- [Live Demo](https://huggingface.co/spaces/OpenBMB/VoxCPM-Demo)
- [Audio Samples](https://openbmb.github.io/voxcpm2-demopage)
- [Nano-VLLM Acceleration](https://github.com/a710128/nanovllm-voxcpm)

## Inspired By

This project follows the architecture pioneered by [chatterbex](https://github.com/holsee/chatterbex), an Elixir wrapper for Chatterbox TTS.
