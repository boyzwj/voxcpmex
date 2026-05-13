defmodule VoxCPMEx do
  @moduledoc """
  Elixir wrapper for [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) —
  a tokenizer-free, diffusion autoregressive Text-to-Speech model from OpenBMB.

  **2B parameters**, **30 languages**, **48kHz** output,
  trained on over **2 million hours** of multilingual speech data.

  ## Features

    * 🌍 **30-Language Multilingual** — Supports Chinese, English, Japanese,
      Korean, Arabic, French, German, and 23+ more
    * 🎨 **Voice Design** — Generate a novel voice from a natural-language
      description alone; *no reference audio required*
    * 🎛️ **Controllable Cloning** — Clone any voice from a short clip, with
      optional style guidance
    * 🎙️ **Ultimate Cloning** — Audio-continuation cloning with transcript
      for every vocal nuance
    * 🔊 **48kHz Studio-Quality Output** — AudioVAE V2 super-resolution
    * ⚡ **Streaming** — RTF as low as ~0.3 on RTX 4090
    * 🎓 **LoRA Fine-Tuning** — Adapt with as little as 5–10 minutes of audio

  ## Quick Start

      # Start a model server
      {:ok, pid} = VoxCPMEx.start_link(device: "cuda")

      # Wait for model to load (30-60s on first run, downloads ~8GB)
      :ok = VoxCPMEx.await_ready(pid)

      # Generate speech
      {:ok, audio} = VoxCPMEx.generate(pid, "Hello, world!")
      :ok = VoxCPMEx.save(audio, "output.wav")

  ## Voice Design

  Describe the voice in parentheses at the start of text:

      {:ok, audio} = VoxCPMEx.generate(pid,
        "(A young woman, gentle and sweet voice) Hello, welcome!"
      )

  ## Voice Cloning

  Provide a reference audio file (short clip):

      {:ok, audio} = VoxCPMEx.generate(pid, "Hello in my voice!",
        audio_prompt: "reference.wav"
      )

  ## Ultimate Cloning

  Provide reference audio + its exact transcript:

      {:ok, audio} = VoxCPMEx.generate(pid, "This is an ultimate clone.",
        prompt_wav_path: "speaker.wav",
        prompt_text: "The transcript of the reference.",
        audio_prompt: "speaker.wav"
      )

  ## Named Servers

      {:ok, _pid} = VoxCPMEx.start_link(device: "cuda", name: MyApp.TTS)
      {:ok, audio} = VoxCPMEx.generate(MyApp.TTS, "Hello!")

  ## Requirements

    * Python ≥ 3.10, `voxcpm` pip package
    * CUDA GPU (8+ GB VRAM recommended), Apple Silicon (MPS), or CPU
    * Elixir ≥ 1.14

  ## Installation

  1. Add to `mix.exs`:

      ```elixir
      {:voxcpmex, "~> 0.1.0"}
      ```

  2. Install Python dependencies:

      ```bash
      mix voxcpmex.setup
      ```
  """

  alias VoxCPMEx.Server

  @typedoc "Audio data as a binary WAV file in memory"
  @type audio :: binary()

  @type generate_opt ::
          {:audio_prompt, String.t()}
          | {:prompt_wav_path, String.t()}
          | {:prompt_text, String.t()}
          | {:cfg_value, float()}
          | {:inference_timesteps, pos_integer()}
          | {:min_len, pos_integer()}
          | {:max_len, pos_integer()}
          | {:normalize, boolean()}
          | {:denoise, boolean()}

  # ---------------------------------------------------------------------------
  # Lifecycle
  # ---------------------------------------------------------------------------

  @doc """
  Starts a VoxCPMEx model server.

  ## Options

    * `:model` — HuggingFace model ID. Default: `"openbmb/VoxCPM2"`
    * `:device` — Compute device (`"cuda"`, `"cpu"`, `"mps"`). Default: `"cuda"`
    * `:load_denoiser` — Load audio denoiser for reference audio cleanup. Default: `false`
    * `:optimize` — Enable `torch.compile` for faster inference. Default: `true`
    * `:name` — Optional GenServer name

  ## Examples

      {:ok, pid} = VoxCPMEx.start_link(device: "cuda")
      {:ok, pid} = VoxCPMEx.start_link(device: "cpu", name: TTS)

  """
  @spec start_link(Server.start_opts()) :: GenServer.on_start()
  defdelegate start_link(opts), to: Server

  @doc """
  Waits for the model to finish loading and be ready for generation.

  Returns `:ok` when ready, or `{:error, :not_ready}` if the timeout is reached.
  """
  @spec await_ready(GenServer.server(), timeout()) :: :ok | {:error, term()}
  defdelegate await_ready(server, timeout \\ 120_000), to: Server

  # ---------------------------------------------------------------------------
  # Generation
  # ---------------------------------------------------------------------------

  @doc """
  Generates speech audio from text.

  Returns `{:ok, audio_binary}` where `audio_binary` is a valid WAV file in memory.

  ## Options

    * `:audio_prompt` — Reference audio path for voice cloning
    * `:prompt_wav_path` + `:prompt_text` — For ultimate cloning
    * `:cfg_value` — Guidance scale (recommended: 1.0–3.0). Default: `2.0`
    * `:inference_timesteps` — Diffusion steps (recommended: 4–30). Default: `10`
    * `:min_len` — Minimum audio length in tokens. Default: `2`
    * `:max_len` — Maximum token length. Default: `4096`
    * `:normalize` — Run text normalization (expand numbers/dates). Default: `false`
    * `:denoise` — Denoise reference audio before cloning. Default: `false`

  ## Examples

      # Basic TTS
      {:ok, audio} = VoxCPMEx.generate(pid, "Hello, world!")

      # Voice cloning
      {:ok, audio} = VoxCPMEx.generate(pid, "Hello!",
        audio_prompt: "reference.wav"
      )

      # Voice Design
      {:ok, audio} = VoxCPMEx.generate(pid,
        "(warm male voice, confident) Welcome to the demo."
      )

      # High quality (more steps, slower)
      {:ok, audio} = VoxCPMEx.generate(pid, "Quality matters.",
        inference_timesteps: 30, cfg_value: 3.0
      )

      # Fast (fewer steps)
      {:ok, audio} = VoxCPMEx.generate(pid, "Speed matters.",
        inference_timesteps: 4
      )

  """
  @spec generate(GenServer.server(), String.t(), [generate_opt()]) :: {:ok, audio()} | {:error, term()}
  defdelegate generate(server, text, opts \\ []), to: Server

  @doc """
  Generates speech with a custom timeout. Same as `generate/3` but allows
  specifying how long to wait for the generation to complete.
  """
  @spec generate(GenServer.server(), String.t(), [generate_opt()], timeout()) ::
          {:ok, audio()} | {:error, term()}
  defdelegate generate(server, text, opts, timeout), to: Server

  @doc """
  Generates speech using VoxCPM2's internal streaming pipeline.

  Returns `{:ok, %{audio: binary, sample_rate: int, duration: float, num_chunks: int}}`.

  Useful for long utterances where you want progressive output.
  """
  @spec generate_streaming(GenServer.server(), String.t(), [generate_opt()]) ::
          {:ok, map()} | {:error, term()}
  defdelegate generate_streaming(server, text, opts \\ []), to: Server

  # ---------------------------------------------------------------------------
  # I/O
  # ---------------------------------------------------------------------------

  @doc """
  Saves audio binary to a WAV file.

  ## Examples

      {:ok, audio} = VoxCPMEx.generate(pid, "Hello!")
      :ok = VoxCPMEx.save(audio, "output.wav")

  """
  @spec save(audio(), Path.t()) :: :ok | {:error, term()}
  defdelegate save(audio, path), to: Server

  # ---------------------------------------------------------------------------
  # LoRA
  # ---------------------------------------------------------------------------

  @doc """
  Loads LoRA fine-tuning weights from a checkpoint file.

  Returns `{:ok, loaded_count, skipped_count}` on success.
  """
  @spec load_lora(GenServer.server(), String.t()) ::
          {:ok, non_neg_integer(), non_neg_integer()} | {:error, term()}
  defdelegate load_lora(server, lora_path), to: Server

  @doc """
  Resets all LoRA weights to zero (disables LoRA without unloading).
  """
  @spec unload_lora(GenServer.server()) :: :ok | {:error, term()}
  defdelegate unload_lora(server), to: Server
end
