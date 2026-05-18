defmodule MossTTSNano do
  @moduledoc """
  Elixir wrapper for [MOSS-TTS-Nano](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M) —
  a multilingual, tiny, 0.1B-parameter Text-to-Speech model from OpenMOSS.

  **0.1B params** · **20 languages** · **48kHz stereo** · CPU-friendly · Voice cloning

  ## Features

    * 🤏 **Tiny** — 0.1B parameters, designed for realtime speech generation
    * 🌍 **Multilingual** — Chinese, English, Japanese, Korean, and 16 more
    * 🎙️ **Voice Cloning** — Clone any voice from a short reference clip
    * 🧠 **Continuation mode** — Continue speech from a reference clip
    * 💻 **CPU-friendly** — Streaming generation can run on a 4-core CPU
    * 🔊 **48kHz stereo** — Native high-quality audio output
    * ⚡ **True streaming** — Get audio chunks as they are generated

  ## Protocol

  Same MessagePack binary-framing protocol as VoxCPMEx — drop-in compatible
  at the Erlang Port level, different model backend.

  ## Quick Start

      {:ok, pid} = MossTTSNano.start_link(device: "cpu")
      :ok = MossTTSNano.await_ready(pid)

      # List available voice presets
      voices = MossTTSNano.list_voices(pid)

      # Basic generation with a built-in voice
      {:ok, audio} = MossTTSNano.generate(pid, "Hello, world!", voice: "Junhao")
      :ok = MossTTSNano.save(audio, "output.wav")

  ## Voice Cloning

      {:ok, audio} = MossTTSNano.generate(pid, "Hello in my voice!",
        mode: :voice_clone,
        prompt_audio_path: "reference.wav"
      )

  ## Continuation mode

      {:ok, audio} = MossTTSNano.generate(pid, "to be continued...",
        mode: :continuation,
        prompt_audio_path: "reference.wav",
        prompt_text: "transcript of the reference audio"
      )

  ## Streaming

      {:ok, ref} = MossTTSNano.generate_streaming_async(pid, "Long text...")
      {:ok, audio} = MossTTSNano.collect_stream(pid, ref)

  Or poll chunk by chunk:

      {:ok, ref} = MossTTSNano.generate_streaming_async(pid, "Long text...")

      stream_loop(pid, ref)

      defp stream_loop(pid, ref) do
        case MossTTSNano.next_chunk(pid, ref) do
          {:ok, chunk} -> play(chunk); stream_loop(pid, ref)
          :eos -> :ok
          {:error, _} -> :ok
        end
      end

  ## Requirements

    * Python ≥ 3.10, with `torch`, `transformers`, and the MOSS-TTS-Nano runtime
    * CPU (4+ cores) or CUDA GPU (recommended)
    * Elixir ≥ 1.14

  ## Installation

      # mix.exs
      {:voxcpmex, "~> 0.3.0"}

      # Install Python deps
      pip install torch torchaudio transformers soundfile msgpack numpy
  """

  alias MossTTSNano.Server

  @type audio :: binary()

  @type generate_opt ::
          {:voice, String.t()}
          | {:mode, :voice_clone | :continuation}
          | {:prompt_audio_path, String.t()}
          | {:prompt_text, String.t()}
          | {:max_new_frames, pos_integer()}
          | {:voice_clone_max_text_tokens, integer()}
          | {:voice_clone_max_memory_per_sample_gb, float()}
          | {:do_sample, boolean()}
          | {:text_temperature, float()}
          | {:text_top_p, float()}
          | {:text_top_k, pos_integer()}
          | {:audio_temperature, float()}
          | {:audio_top_p, float()}
          | {:audio_top_k, pos_integer()}
          | {:audio_repetition_penalty, float()}
          | {:nq, pos_integer()}
          | {:seed, pos_integer()}

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
  Starts a MossTTSNano model server.

  ## Options

    * `:checkpoint` — Model path/HF ID. Default: `"OpenMOSS-Team/MOSS-TTS-Nano-100M"`
    * `:device` — `"cpu"`, `"cuda"`, `"auto"`. Default: `"cpu"`
    * `:name` — Optional GenServer name
  """
  @spec start_link(Server.start_opts()) :: GenServer.on_start()
  defdelegate start_link(opts), to: Server

  @doc """
  Waits for the model to finish loading. Returns `:ok` when ready.
  """
  @spec await_ready(GenServer.server(), timeout()) :: :ok | {:error, term()}
  defdelegate await_ready(server, timeout \\ 180_000), to: Server

  # ---------------------------------------------------------------------------
  # Voices
  # ---------------------------------------------------------------------------

  @doc """
  Lists available built-in voice presets.

  Returns `{:ok, [voice_names]}` or `{:error, reason}`.
  """
  @spec list_voices(GenServer.server()) :: {:ok, [String.t()]} | {:error, term()}
  defdelegate list_voices(server), to: Server

  # ---------------------------------------------------------------------------
  # Synchronous Generation
  # ---------------------------------------------------------------------------

  @doc """
  Generates speech audio from text. Returns `{:ok, audio_wav}`.

  ## Options

    * `:voice` — Built-in voice name (e.g., "Junhao"). Default: first available.
    * `:mode` — `:voice_clone` (default) or `:continuation`
    * `:prompt_audio_path` — Reference audio file for voice cloning
    * `:prompt_text` — Transcript of reference audio (for continuation)
    * `:max_new_frames` — Max audio frames. Default: 375
    * `:do_sample` — Enable sampling (non-greedy). Default: true
    * `:seed` — Random seed for reproducibility

  ## Examples

      # Default voice
      {:ok, audio} = MossTTSNano.generate(pid, "Hello world!")
      :ok = MossTTSNano.save(audio, "out.wav")

      # Specific voice
      {:ok, audio} = MossTTSNano.generate(pid,
        "你好世界!",
        voice: "Xiaoyu"
      )

      # Voice cloning
      {:ok, audio} = MossTTSNano.generate(pid, "Hello!",
        mode: :voice_clone,
        prompt_audio_path: "my_voice.wav"
      )

      # Continuation
      {:ok, audio} = MossTTSNano.generate(pid,
        " ...to be continued",
        mode: :continuation,
        prompt_audio_path: "reference.wav",
        prompt_text: "original transcript here"
      )
  """
  @spec generate(GenServer.server(), String.t(), [generate_opt()]) ::
          {:ok, audio()} | {:error, term()}
  defdelegate generate(server, text, opts \\ []), to: Server

  @spec generate(GenServer.server(), String.t(), [generate_opt()], timeout()) ::
          {:ok, audio()} | {:error, term()}
  defdelegate generate(server, text, opts, timeout), to: Server

  # ---------------------------------------------------------------------------
  # Streaming Generation
  # ---------------------------------------------------------------------------

  @doc """
  Starts **asynchronous streaming** generation.

  Returns `{:ok, stream_ref}` immediately — the model generates in the
  background and chunks are delivered as they are produced.

  Poll for chunks with `next_chunk/2`:
      {:ok, chunk} → raw float32 PCM bytes
      :eos → stream complete
      {:error, reason}

  Or collect everything at once with `collect_stream/2`.

  ## Example

      {:ok, ref} = MossTTSNano.generate_streaming_async(pid, "Long text...")

      stream_loop(pid, ref)

      defp stream_loop(pid, ref) do
        case MossTTSNano.next_chunk(pid, ref) do
          {:ok, chunk} -> play_chunk(chunk); stream_loop(pid, ref)
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
    * `:eos` — stream is complete
    * `{:error, reason}`
  """
  @spec next_chunk(GenServer.server(), reference()) ::
          {:ok, binary()} | :eos | {:error, term()}
  defdelegate next_chunk(server, ref), to: Server

  @doc """
  Collects all remaining chunks from a streaming session and returns
  the full concatenated audio as raw bytes.
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
end
