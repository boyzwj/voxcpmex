defmodule VoxCPMEx.Server do
  @moduledoc """
  GenServer that manages a Python VoxCPM2 bridge process via Erlang Port.

  Each `start_link/1` spawns a dedicated Python process with the loaded model,
  allowing multiple models or instances to run concurrently.

  Architecture:

      +---------------+      JSON/stdin       +-----------------+
      |    Elixir     | --------------------> |     Python      |
      |   GenServer   |                       |   VoxCPM2       |
      |               | <-------------------- |                 |
      +---------------+    JSON/stdout        +-----------------+
                                    Base64 WAV
  """

  use GenServer

  require Logger

  @type model_option ::
          {:model, String.t()}
          | {:device, String.t()}
          | {:load_denoiser, boolean()}
          | {:optimize, boolean()}
          | {:name, atom()}

  @type start_opts :: [model_option()]

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

  @default_model "openbmb/VoxCPM2"
  @default_device "cuda"

  # ---------------------------------------------------------------------------
  # Client API
  # ---------------------------------------------------------------------------

  @doc """
  Starts a VoxCPMEx model server.

  ## Options

    * `:model` — HuggingFace model ID. Default: `"openbmb/VoxCPM2"`
    * `:device` — Compute device (`"cuda"`, `"cpu"`, `"mps"`). Default: `"cuda"`
    * `:load_denoiser` — Whether to load the audio denoiser. Default: `false`
    * `:optimize` — Enable `torch.compile` optimizations. Default: `true`
    * `:name` — Optional GenServer name for easy access

  ## Examples

      {:ok, pid} = VoxCPMEx.Server.start_link(device: "cuda")
      {:ok, pid} = VoxCPMEx.Server.start_link(device: "cpu", name: MyApp.TTS)

  """
  @spec start_link(start_opts()) :: GenServer.on_start()
  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc """
  Waits for the model to finish loading.

  Returns `:ok` when ready, or `{:error, reason}` if initialization fails.
  """
  @spec await_ready(GenServer.server(), timeout()) :: :ok | {:error, term()}
  def await_ready(server, timeout \\ 120_000) do
    GenServer.call(server, :await_ready, timeout)
  end

  @doc """
  Generates speech audio from text. Returns `{:ok, audio_binary}` or `{:error, reason}`.
  `audio_binary` is a valid WAV file in memory.

  ## Options

    * `:audio_prompt` — Path to reference audio for voice cloning
    * `:prompt_wav_path` + `:prompt_text` — For ultimate cloning
    * `:cfg_value` — Guidance scale (1.0–3.0). Default: `2.0`
    * `:inference_timesteps` — Diffusion steps (4–30). Default: `10`
    * `:min_len` — Minimum audio length in tokens. Default: `2`
    * `:max_len` — Maximum token length. Default: `4096`
    * `:normalize` — Run text normalization. Default: `false`
    * `:denoise` — Denoise reference audio. Default: `false`

  ## Examples

      # Simple TTS
      {:ok, audio} = VoxCPMEx.Server.generate(pid, "Hello, world!")

      # Voice cloning
      {:ok, audio} = VoxCPMEx.Server.generate(pid, "Hello!",
        audio_prompt: "reference.wav"
      )

      # Voice Design — describe the voice in parentheses
      {:ok, audio} = VoxCPMEx.Server.generate(pid,
        "(A young woman, gentle and sweet voice) Hello!"
      )

  """
  @spec generate(GenServer.server(), String.t(), [generate_opt()]) :: {:ok, binary()} | {:error, term()}
  def generate(server, text, opts \\ []) do
    generate(server, text, opts, 120_000)
  end

  @spec generate(GenServer.server(), String.t(), [generate_opt()], timeout()) :: {:ok, binary()} | {:error, term()}
  def generate(server, text, opts, timeout) do
    GenServer.call(server, {:generate, text, opts}, timeout)
  end

  @doc """
  Generates speech with streaming pipeline.
  Same interface as `generate/3` but uses VoxCPM2's streaming mode.
  Returns audio plus metadata.
  """
  @spec generate_streaming(GenServer.server(), String.t(), [generate_opt()]) ::
          {:ok, map()} | {:error, term()}
  def generate_streaming(server, text, opts \\ []) do
    GenServer.call(server, {:generate_streaming, text, opts}, 120_000)
  end

  @doc """
  Saves audio binary to a WAV file.

  ## Examples

      {:ok, audio} = VoxCPMEx.Server.generate(pid, "Hello!")
      :ok = VoxCPMEx.Server.save(audio, "output.wav")

  """
  @spec save(binary(), Path.t()) :: :ok | {:error, term()}
  def save(audio, path) when is_binary(audio) do
    File.write(path, audio)
  end

  @doc """
  Loads LoRA weights from a checkpoint file.
  Returns `{:ok, loaded_count, skipped_count}`.
  """
  @spec load_lora(GenServer.server(), String.t()) ::
          {:ok, non_neg_integer(), non_neg_integer()} | {:error, term()}
  def load_lora(server, lora_path) do
    GenServer.call(server, {:load_lora, lora_path}, 30_000)
  end

  @doc """
  Resets all LoRA weights to zero.
  """
  @spec unload_lora(GenServer.server()) :: :ok | {:error, term()}
  def unload_lora(server) do
    GenServer.call(server, :unload_lora, 15_000)
  end

  # ---------------------------------------------------------------------------
  # GenServer Callbacks
  # ---------------------------------------------------------------------------

  @impl true
  def init(opts) do
    model = Keyword.get(opts, :model, @default_model)
    device = Keyword.get(opts, :device, @default_device)
    load_denoiser = Keyword.get(opts, :load_denoiser, false)
    optimize = Keyword.get(opts, :optimize, true)

    bridge_path = Path.join(:code.priv_dir(:voxcpmex), "python/voxcpmex_bridge.py")

    unless File.exists?(bridge_path) do
      raise "Python bridge not found at #{bridge_path}"
    end

    python_cmd = System.find_executable("python3") || System.find_executable("python") || "python3"

    port =
      Port.open({:spawn_executable, python_cmd}, [
        :binary,
        :use_stdio,
        :exit_status,
        :stderr_to_stdout,
        args: ["-u", bridge_path]
      ])

    init_msg = %{
      type: "init",
      model: model,
      device: device,
      load_denoiser: load_denoiser,
      optimize: optimize
    }

    send(port, {self(), {:command, Jason.encode!(init_msg) <> "\n"}})

    state = %{
      port: port,
      ready: false,
      device: nil,
      sample_rate: nil,
      pending: :queue.new()
    }

    {:ok, state}
  end

  @impl true
  def handle_call(:await_ready, _from, state) do
    if state.ready do
      {:reply, :ok, state}
    else
      {:reply, {:error, :not_ready}, state}
    end
  end

  @impl true
  def handle_call({:generate, text, opts}, from, state) do
    if not state.ready do
      {:reply, {:error, :not_ready}, state}
    else
      cmd = Map.merge(%{type: "generate", text: text}, Map.new(opts))
      send(state.port, {self(), {:command, Jason.encode!(cmd) <> "\n"}})
      {:noreply, %{state | pending: :queue.in({from, :generate}, state.pending)}}
    end
  end

  @impl true
  def handle_call({:generate_streaming, text, opts}, from, state) do
    if not state.ready do
      {:reply, {:error, :not_ready}, state}
    else
      cmd = Map.merge(%{type: "generate_streaming", text: text}, Map.new(opts))
      send(state.port, {self(), {:command, Jason.encode!(cmd) <> "\n"}})
      {:noreply, %{state | pending: :queue.in({from, :generate}, state.pending)}}
    end
  end

  @impl true
  def handle_call({:load_lora, lora_path}, from, state) do
    cmd = %{type: "load_lora", lora_path: lora_path}
    send(state.port, {self(), {:command, Jason.encode!(cmd) <> "\n"}})
    {:noreply, %{state | pending: :queue.in({from, :lora_load}, state.pending)}}
  end

  @impl true
  def handle_call(:unload_lora, from, state) do
    cmd = %{type: "unload_lora"}
    send(state.port, {self(), {:command, Jason.encode!(cmd) <> "\n"}})
    {:noreply, %{state | pending: :queue.in({from, :lora_unload}, state.pending)}}
  end

  # ---------------------------------------------------------------------------
  # Port messages
  # ---------------------------------------------------------------------------

  @impl true
  def handle_info({_port, {:data, data}}, state) do
    data
    |> String.split("\n", trim: true)
    |> Enum.reduce(state, fn line, acc ->
      case Jason.decode(line) do
        {:ok, msg} -> handle_response(msg, acc)
        {:error, _} ->
          Logger.debug("VoxCPM bridge: #{line}")
          acc
      end
    end)
  end

  @impl true
  def handle_info({_port, {:exit_status, status}}, state) do
    Logger.error("VoxCPM bridge exited with status #{status}")
    {:stop, {:bridge_exit, status}, state}
  end

  # ---------------------------------------------------------------------------
  # Response handling
  # ---------------------------------------------------------------------------

  defp handle_response(%{"status" => "ok"} = msg, state) do
    cond do
      Map.has_key?(msg, "device") and Map.has_key?(msg, "sample_rate") ->
        Logger.info("VoxCPM model loaded on #{msg["device"]}, sr=#{msg["sample_rate"]}")
        %{state | ready: true, device: msg["device"], sample_rate: msg["sample_rate"]}

      Map.has_key?(msg, "audio") ->
        audio = Base.decode64!(msg["audio"])
        case :queue.out(state.pending) do
          {{:value, {from, _type}}, new_queue} ->
            GenServer.reply(from, {:ok, audio})
            %{state | pending: new_queue}
          {:empty, _} ->
            Logger.warning("Received generate response with no pending caller")
            state
        end

      Map.has_key?(msg, "loaded") ->
        case :queue.out(state.pending) do
          {{:value, {from, _type}}, new_queue} ->
            GenServer.reply(from, {:ok, msg["loaded"], msg["skipped"]})
            %{state | pending: new_queue}
          {:empty, _} ->
            state
        end

      true ->
        case :queue.out(state.pending) do
          {{:value, {from, _type}}, new_queue} ->
            GenServer.reply(from, :ok)
            %{state | pending: new_queue}
          {:empty, _} ->
            state
        end
    end
  end

  defp handle_response(%{"status" => "error", "error" => error}, state) do
    Logger.error("VoxCPM bridge error: #{error}")

    case :queue.out(state.pending) do
      {{:value, {from, _type}}, new_queue} ->
        GenServer.reply(from, {:error, error})
        %{state | pending: new_queue}
      {:empty, _} ->
        state
    end
  end
end
