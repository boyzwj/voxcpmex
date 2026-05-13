defmodule VoxCPMEx do
  @moduledoc """
  Elixir wrapper for [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) —
  a tokenizer-free, diffusion autoregressive Text-to-Speech model from OpenBMB.

  **2B parameters** · **30 languages** · **48kHz output** · 2M+ hours training data.

  ## Features

    * 🌍 **30-Language Multilingual** — Chinese, English, Japanese, Korean, Arabic,
      French, German, and 23+ more
    * 🎨 **Voice Design** — Generate a novel voice from text description alone
    * 🎛️ **Controllable Cloning** — Clone any voice from a short clip, with style guidance
    * 🎙️ **Ultimate Cloning** — Audio-continuation cloning for maximum fidelity
    * 🔊 **48kHz Studio Output** — AudioVAE V2 super-resolution
    * ⚡ **True Streaming** — Get audio chunks as they're generated
    * 🎓 **LoRA Fine-Tuning** — Adapt with 5–10 minutes of audio

  ## Protocol (v2)

  VoxCPMEx uses **MessagePack** over **binary-framed** Erlang Ports.
  Audio is transmitted as raw bytes — no base64 encoding overhead.

  ## Quick Start

      {:ok, pid} = VoxCPMEx.start_link(device: "cuda")
      :ok = VoxCPMEx.await_ready(pid)
      {:ok, audio} = VoxCPMEx.generate(pid, "Hello, world!")
      :ok = VoxCPMEx.save(audio, "output.wav")

  ## Voice Design

      {:ok, audio} = VoxCPMEx.generate(pid,
        "(A young woman, gentle and sweet voice) Welcome!"
      )

  ## Voice Cloning

      {:ok, audio} = VoxCPMEx.generate(pid, "Hello in my voice!",
        audio_prompt: "reference.wav"
      )

  ## Streaming (v2 — true chunk-by-chunk)

      {:ok, ref} = VoxCPMEx.generate_streaming_async(pid, "Long text...")
      stream_loop(ref)

      defp stream_loop(ref) do
        case VoxCPMEx.next_chunk(pid, ref) do
          {:ok, chunk} -> IO.puts("got chunk"); stream_loop(ref)
          :eos -> IO.puts("done!")
          {:error, reason} -> IO.puts("error")
        end
      end

  Or collect everything at once:

      {:ok, ref} = VoxCPMEx.generate_streaming_async(pid, "Long text...")
      {:ok, audio} = VoxCPMEx.collect_stream(pid, ref)

  ## Requirements

    * Python ≥ 3.10, `voxcpm` + `msgpack` pip packages
    * CUDA GPU (8+ GB VRAM), Apple Silicon (MPS), or CPU
    * Elixir ≥ 1.14

  ## Installation

      # mix.exs
      {:voxcpmex, "~> 0.2.0"}

      # Install Python deps
      mix voxcpmex.setup
  """

  alias VoxCPMEx.Server

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

  @doc "Returns runtime model information."
  @spec info(GenServer.server()) :: map()
  defdelegate info(server), to: Server

  @doc "Gracefully stops the server and Python bridge."
  @spec stop(GenServer.server()) :: :ok
  defdelegate stop(server), to: Server

  @doc """
  Starts a VoxCPMEx model server.

  ## Options

    * `:model` — HuggingFace model ID. Default: `"openbmb/VoxCPM2"`
    * `:device` — `"cuda"`, `"cpu"`, `"mps"`. Default: `"cuda"`
    * `:load_denoiser` — Load audio denoiser. Default: `false`
    * `:optimize` — Enable `torch.compile`. Default: `true`
    * `:name` — Optional GenServer name
  """
  @spec start_link(Server.start_opts()) :: GenServer.on_start()
  defdelegate start_link(opts), to: Server

  @doc """
  Waits for the model to finish loading. Returns `:ok` when ready.
  """
  @spec await_ready(GenServer.server(), timeout()) :: :ok | {:error, term()}
  defdelegate await_ready(server, timeout \\ 120_000), to: Server

  # ---------------------------------------------------------------------------
  # Synchronous Generation
  # ---------------------------------------------------------------------------

  @doc """
  Generates speech audio from text. Returns `{:ok, audio_wav}`.

  ## Options

    * `:audio_prompt` — Reference audio for voice cloning
    * `:prompt_wav_path` + `:prompt_text` — Ultimate cloning
    * `:cfg_value` — Guidance scale (1.0–3.0). Default: `2.0`
    * `:inference_timesteps` — Diffusion steps (4–30). Default: `10`
    * `:min_len` — Min audio length in tokens. Default: `2`
    * `:max_len` — Max token length. Default: `4096`
    * `:normalize` — Text normalization. Default: `false`
    * `:denoise` — Denoise reference audio. Default: `false`

  ## Examples

      # Basic
      {:ok, audio} = VoxCPMEx.generate(pid, "Hello!")
      :ok = VoxCPMEx.save(audio, "out.wav")

      # Voice Design
      {:ok, audio} = VoxCPMEx.generate(pid,
        "(warm male voice) Welcome to the demo."
      )

      # Voice Cloning
      {:ok, audio} = VoxCPMEx.generate(pid, "Hello!",
        audio_prompt: "ref.wav"
      )

      # Quality tuning
      {:ok, audio} = VoxCPMEx.generate(pid, "Quality matters.",
        inference_timesteps: 30, cfg_value: 3.0
      )
  """
  @spec generate(GenServer.server(), String.t(), [generate_opt()]) ::
          {:ok, audio()} | {:error, term()}
  defdelegate generate(server, text, opts \\ []), to: Server

  @spec generate(GenServer.server(), String.t(), [generate_opt()], timeout()) ::
          {:ok, audio()} | {:error, term()}
  defdelegate generate(server, text, opts, timeout), to: Server

  # ---------------------------------------------------------------------------
  # Streaming Generation (v2)
  # ---------------------------------------------------------------------------

  @doc """
  Starts **asynchronous streaming** generation.

  Returns `{:ok, stream_ref}` immediately — the model generates in the
  background and chunks are delivered to the GenServer as they're produced.

  Poll for chunks with `next_chunk/2`:
      {:ok, chunk} → raw float32 PCM bytes
      :eos → stream complete
      {:error, reason}

  Or collect everything at once with `collect_stream/2`.

  ## Example

      {:ok, ref} = VoxCPMEx.generate_streaming_async(pid, "Long text...")

      # Poll for chunks
      stream_loop(pid, ref)

      defp stream_loop(pid, ref) do
        case VoxCPMEx.next_chunk(pid, ref) do
          {:ok, chunk} ->
            play_chunk(chunk)
            stream_loop(pid, ref)
          :eos -> :ok
          {:error, reason} -> Logger.error("Stream error")
        end
      end
  """
  @spec generate_streaming_async(GenServer.server(), String.t(), [generate_opt()]) ::
          {:ok, reference()} | {:error, term()}
  defdelegate generate_streaming_async(server, text, opts \\ []), to: Server

  @doc """
  Returns the next chunk from an active streaming session.

  Returns:
    * `{:ok, chunk}` — raw float32 PCM bytes for this chunk
    * `:eos` — stream is complete, no more chunks
    * `{:error, reason}`
  """
  @spec next_chunk(GenServer.server(), reference()) ::
          {:ok, binary()} | :eos | {:error, term()}
  defdelegate next_chunk(server, ref), to: Server

  @doc """
  Collects all remaining chunks from a streaming session and returns
  the full concatenated audio as raw bytes.

  Returns `{:ok, audio_bytes}` when all chunks are collected,
  or `{:error, reason}`.
  """
  @spec collect_stream(GenServer.server(), reference()) ::
          {:ok, binary()} | {:error, term()}
  defdelegate collect_stream(server, ref), to: Server

  # ---------------------------------------------------------------------------
  # I/O
  # ---------------------------------------------------------------------------

  @doc """
  Saves audio binary to a WAV file.
  """
  @spec save(audio(), Path.t()) :: :ok | {:error, term()}
  defdelegate save(audio, path), to: Server

  # ---------------------------------------------------------------------------
  # LoRA
  # ---------------------------------------------------------------------------

  @doc """
  Loads LoRA fine-tuning weights. Returns `{:ok, loaded, skipped}`.
  """
  @spec load_lora(GenServer.server(), String.t()) ::
          {:ok, non_neg_integer(), non_neg_integer()} | {:error, term()}
  defdelegate load_lora(server, lora_path), to: Server

  @doc """
  Resets all LoRA weights to zero.
  """
  @spec unload_lora(GenServer.server()) :: :ok | {:error, term()}
  defdelegate unload_lora(server), to: Server
end
