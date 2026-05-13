defmodule VoxCPMEx.Server do
  @moduledoc """
  GenServer that manages a Python VoxCPM2 bridge via Erlang Port.

  ## Protocol (v2.1 — MessagePack binary framing, single-ref streaming)

  Frame format: `[4-byte BE total_length][msgpack-encoded payload]`

  Audio is raw WAV bytes inside msgpack — no base64.

  Streaming: unlike v2.0 which used a dual stream_id/ref mapping, v2.1
  eliminates stream IDs entirely. The Elixir ref (sent in the request) is
  the sole identifier — Python just echoes stream_start/chunk/end in order.
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
  @frame_header_bytes 4
  @stream_ttl_ms 60_000

  # =========================================================================
  # Client API
  # =========================================================================

  @spec start_link(start_opts()) :: GenServer.on_start()
  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @doc "Returns runtime model info: device, sample_rate, status."
  @spec info(GenServer.server()) :: map()
  def info(server) do
    GenServer.call(server, :info)
  end

  @doc """
  Waits for the model to finish loading.

  Returns `:ok` when ready, `{:error, :loading}` if still initializing,
  or `{:error, reason}` if initialization failed.
  """
  @spec await_ready(GenServer.server(), timeout()) :: :ok | {:error, term()}
  def await_ready(server, timeout \\ 120_000) do
    GenServer.call(server, :await_ready, timeout)
  end

  @doc "Gracefully stops the GenServer and the Python bridge process."
  @spec stop(GenServer.server()) :: :ok
  def stop(server) do
    GenServer.stop(server, :normal)
  end

  @spec generate(GenServer.server(), String.t(), [generate_opt()]) :: {:ok, binary()} | {:error, term()}
  def generate(server, text, opts \\ []) do
    generate(server, text, opts, 120_000)
  end

  @spec generate(GenServer.server(), String.t(), [generate_opt()], timeout()) :: {:ok, binary()} | {:error, term()}
  def generate(server, text, opts, timeout) do
    GenServer.call(server, {:generate, text, opts}, timeout)
  end

  @doc """
  Starts async streaming. Returns `{:ok, ref}` immediately.

  Poll with `next_chunk/2`: `{:ok, chunk}` (raw float32 PCM) | `:eos` | `{:error, reason}`.
  Collect everything with `collect_stream/2`: `{:ok, wav_binary}`.
  """
  @spec generate_streaming_async(GenServer.server(), String.t(), [generate_opt()]) :: {:ok, reference()} | {:error, term()}
  def generate_streaming_async(server, text, opts \\ []) do
    GenServer.call(server, {:generate_streaming_async, text, opts}, 10_000)
  end

  @spec next_chunk(GenServer.server(), reference()) :: {:ok, binary()} | :eos | {:error, term()}
  def next_chunk(server, ref) do
    GenServer.call(server, {:next_chunk, ref}, 60_000)
  end

  @spec collect_stream(GenServer.server(), reference()) :: {:ok, binary()} | {:error, term()}
  def collect_stream(server, ref) do
    GenServer.call(server, {:collect_stream, ref}, 120_000)
  end

  @spec save(binary(), Path.t()) :: :ok | {:error, term()}
  def save(audio, path) when is_binary(audio) do
    File.write(path, audio)
  end

  @spec load_lora(GenServer.server(), String.t()) :: {:ok, non_neg_integer(), non_neg_integer()} | {:error, term()}
  def load_lora(server, lora_path) do
    GenServer.call(server, {:load_lora, lora_path}, 30_000)
  end

  @spec unload_lora(GenServer.server()) :: :ok | {:error, term()}
  def unload_lora(server) do
    GenServer.call(server, :unload_lora, 15_000)
  end

  # =========================================================================
  # GenServer Callbacks
  # =========================================================================

  @impl true
  def init(opts) do
    model = Keyword.get(opts, :model, @default_model)
    device = Keyword.get(opts, :device, @default_device)
    load_denoiser = Keyword.get(opts, :load_denoiser, false)
    optimize = Keyword.get(opts, :optimize, true)

    bridge_path = Path.join(:code.priv_dir(:voxcpmex), "python/voxcpmex_bridge.py")

    unless File.exists?(bridge_path) do
      {:stop, "Python bridge not found: #{bridge_path}"}
    else
      python_cmd = System.find_executable("python3") || System.find_executable("python") || "python3"

      port =
        Port.open({:spawn_executable, python_cmd}, [
          :binary,
          :use_stdio,
          :exit_status,
          :stderr_to_stdout,
          args: ["-u", bridge_path]
        ])

      send_frame(port, %{
        "type" => "init",
        "model" => model,
        "device" => device,
        "load_denoiser" => load_denoiser,
        "optimize" => optimize
      })

      state = %{
        port: port,
        model: model,
        status: :loading,
        device: nil,
        sample_rate: nil,
        buffer: <<>>,
        pending: :queue.new(),
        streams: %{},
        stream_timers: %{}
      }

      {:ok, state}
    end
  end

  @impl true
  def terminate(_reason, state) do
    if state.port do
      Port.close(state.port)
    end

    # Cancel all stream timers
    for {_ref, tref} <- state.stream_timers do
      Process.cancel_timer(tref)
    end

    :ok
  end

  # -- call handlers ----------------------------------------------------------

  @impl true
  def handle_call(:info, _from, state) do
    {:reply, %{
      status: state.status,
      device: state.device,
      sample_rate: state.sample_rate,
      model: state.model,
      stream_count: map_size(state.streams)
    }, state}
  end

  def handle_call(:await_ready, _from, state) do
    case state.status do
      :ready -> {:reply, :ok, state}
      :loading -> {:reply, {:error, :loading}, state}
      {:error, reason} -> {:reply, {:error, reason}, state}
    end
  end

  def handle_call({:generate, text, opts}, from, state) do
    if state.status != :ready do
      {:reply, {:error, :not_ready}, state}
    else
      t0 = System.monotonic_time()
      msg = Map.merge(%{"type" => "generate", "text" => text}, stringify_keys(opts))
      send_frame(state.port, msg)
      {:noreply, %{state | pending: :queue.in({from, :generate, t0}, state.pending)}}
    end
  end

  def handle_call({:generate_streaming_async, text, opts}, from, state) do
    if state.status != :ready do
      {:reply, {:error, :not_ready}, state}
    else
      ref = make_ref()
      msg = Map.merge(%{"type" => "generate_streaming", "text" => text}, stringify_keys(opts))
      send_frame(state.port, msg)

      # Set cleanup timer
      tref = Process.send_after(self(), {:stream_timeout, ref}, @stream_ttl_ms)

      streams = Map.put(state.streams, ref, %{
        chunks: [],
        sample_rate: nil,
        done: false,
        waiter: nil,
        collector: nil,
        t0: System.monotonic_time()
      })

      timers = Map.put(state.stream_timers, ref, tref)

      {:noreply,
       %{state | pending: :queue.in({from, :stream_ref, ref}, state.pending),
                 streams: streams, stream_timers: timers}}
    end
  end

  def handle_call({:next_chunk, ref}, from, state) do
    case Map.get(state.streams, ref) do
      nil ->
        {:reply, {:error, :unknown_stream}, state}

      %{chunks: [chunk | rest], done: _done} = stream ->
        # Reset TTL
        timers = reset_stream_timer(state.stream_timers, ref)

        new_streams = Map.put(state.streams, ref, %{stream | chunks: rest})
        {:reply, {:ok, chunk}, %{state | streams: new_streams, stream_timers: timers}}

      %{chunks: [], done: true} ->
        # Clean up
        timers = cancel_stream_timer(state.stream_timers, ref)
        {:reply, :eos, %{state | streams: Map.delete(state.streams, ref), stream_timers: timers}}

      %{chunks: [], done: false} ->
        # Reset TTL and block
        timers = reset_stream_timer(state.stream_timers, ref)
        {:noreply,
         %{state | streams: put_in(state.streams[ref].waiter, from), stream_timers: timers}}
    end
  end

  def handle_call({:collect_stream, ref}, from, state) do
    case Map.get(state.streams, ref) do
      nil ->
        {:reply, {:error, :unknown_stream}, state}

      %{done: false} ->
        {:noreply,
         %{state | streams: put_in(state.streams[ref].collector, from)}}

      %{chunks: chunks, sample_rate: sr, done: true} ->
        timers = cancel_stream_timer(state.stream_timers, ref)
        audio = build_wav(chunks, sr)
        {:reply, {:ok, audio}, %{state | streams: Map.delete(state.streams, ref), stream_timers: timers}}
    end
  end

  def handle_call({:load_lora, lora_path}, from, state) do
    send_frame(state.port, %{"type" => "load_lora", "lora_path" => lora_path})
    {:noreply, %{state | pending: :queue.in({from, :lora_load, nil}, state.pending)}}
  end

  def handle_call(:unload_lora, from, state) do
    send_frame(state.port, %{"type" => "unload_lora"})
    {:noreply, %{state | pending: :queue.in({from, :lora_unload, nil}, state.pending)}}
  end

  # -- port messages ----------------------------------------------------------

  @impl true
  def handle_info({_port, {:data, data}}, state) do
    {msgs, new_buffer} = parse_frames(state.buffer <> data)

    state = Enum.reduce(msgs, %{state | buffer: new_buffer}, fn msg, acc ->
      handle_message(msg, acc)
    end)

    {:noreply, state}
  end

  @impl true
  def handle_info({_port, {:exit_status, status}}, state) do
    Logger.error("VoxCPM bridge exited with status #{status}")
    {:stop, {:bridge_exit, status}, state}
  end

  # Stream timeout — cleanup stale streams
  def handle_info({:stream_timeout, ref}, state) do
    case Map.get(state.streams, ref) do
      nil ->
        {:noreply, state}
      stream ->
        Logger.warning("Stream #{inspect(ref)} timed out after #{@stream_ttl_ms}ms — cleaning up")
        if stream.waiter, do: GenServer.reply(stream.waiter, {:error, :timeout})
        if stream.collector, do: GenServer.reply(stream.collector, {:error, :timeout})
        timers = Map.delete(state.stream_timers, ref)
        {:noreply, %{state | streams: Map.delete(state.streams, ref), stream_timers: timers}}
    end
  end

  # =========================================================================
  # Message dispatch
  # =========================================================================

  defp handle_message(%{"status" => "ok"} = msg, state) do
    cond do
      Map.has_key?(msg, "device") ->
        Logger.info("VoxCPM loaded on #{msg["device"]}, sr=#{msg["sample_rate"]}")
        %{state | status: :ready, device: msg["device"], sample_rate: msg["sample_rate"]}

      Map.has_key?(msg, "audio") ->
        {{:value, {from, _type, t0}}, new_pending} = :queue.out(state.pending)
        emit_telemetry(:generate, t0)
        GenServer.reply(from, {:ok, msg["audio"]})
        %{state | pending: new_pending}

      Map.has_key?(msg, "loaded") ->
        {{:value, {from, _type, _}}, new_pending} = :queue.out(state.pending)
        GenServer.reply(from, {:ok, msg["loaded"], msg["skipped"]})
        %{state | pending: new_pending}

      true ->
        case :queue.out(state.pending) do
          {{:value, {from, _type, _}}, new_pending} ->
            GenServer.reply(from, :ok)
            %{state | pending: new_pending}
          {:empty, _} -> state
        end
    end
  end

  defp handle_message(%{"status" => "error", "error" => error}, state) do
    Logger.error("Bridge error: #{error}")

    {new_pending, updated_state} =
      case :queue.out(state.pending) do
        {{:value, {from, _type, _t0}}, new_q} ->
          GenServer.reply(from, {:error, error})
          {new_q, state}
        {:empty, _} ->
          {state.pending, state}
      end

    # If this error arrived during init, update status
    updated_state =
      if updated_state.status == :loading do
        %{updated_state | status: {:error, error}}
      else
        updated_state
      end

    %{updated_state | pending: new_pending}
  end

  # -- streaming messages (v2.1: no stream_id, ref is implicit from pending) ---

  defp handle_message(%{"type" => "stream_start"} = msg, state) do
    sample_rate = msg["sample_rate"]

    {{:value, {from, :stream_ref, ref}}, new_pending} = :queue.out(state.pending)
    GenServer.reply(from, {:ok, ref})

    streams = put_in(state.streams[ref].sample_rate, sample_rate)
    %{state | pending: new_pending, streams: streams}
  end

  defp handle_message(%{"type" => "stream_chunk"} = msg, state) do
    chunk = msg["chunk"]

    # The stream is the only one whose ref is not in streams... 
    # We need to find which stream is active. Since only one streaming
    # session can be active, just find the one that isn't done.
    case Enum.find(state.streams, fn {_ref, s} -> not s.done end) do
      {ref, stream} ->
        new_chunks = stream.chunks ++ [chunk]
        streams = put_in(state.streams[ref].chunks, new_chunks)
        state = %{state | streams: streams}

        case Map.get(state.streams, ref) do
          %{waiter: waiter} when not is_nil(waiter) ->
            %{chunks: [next | rest]} = Map.get(state.streams, ref)
            GenServer.reply(waiter, {:ok, next})
            new_s = put_in(state.streams[ref].chunks, rest)
            new_s = put_in(new_s[ref].waiter, nil)
            %{state | streams: new_s}
          _ ->
            state
        end

      nil ->
        Logger.warning("Stream chunk with no active stream")
        state
    end
  end

  defp handle_message(%{"type" => "stream_end"} = msg, state) do
    total_chunks = msg["total_chunks"]

    case Enum.find(state.streams, fn {_ref, s} -> not s.done end) do
      {ref, stream} ->
        t0 = stream.t0
        streams = put_in(state.streams[ref].done, true)
        state = %{state | streams: streams}
        timers = cancel_stream_timer(state.stream_timers, ref)

        emit_telemetry(:stream, t0, %{chunks: total_chunks})

        cond do
          not is_nil(stream.collector) ->
            audio = build_wav(stream.chunks, stream.sample_rate)
            GenServer.reply(stream.collector, {:ok, audio})
            %{state | streams: Map.delete(state.streams, ref), stream_timers: timers}

          not is_nil(stream.waiter) ->
            case stream.chunks do
              [last] ->
                GenServer.reply(stream.waiter, {:ok, last})
                %{state | streams: Map.delete(state.streams, ref), stream_timers: timers}
              chunks when chunks != [] ->
                [next | rest] = chunks
                GenServer.reply(stream.waiter, {:ok, next})
                new_s = put_in(state.streams[ref].chunks, rest)
                new_s = put_in(new_s[ref].waiter, nil)
                %{state | streams: new_s, stream_timers: timers}
              [] ->
                GenServer.reply(stream.waiter, :eos)
                %{state | streams: Map.delete(state.streams, ref), stream_timers: timers}
            end

          true ->
            %{state | stream_timers: timers}
        end

      nil ->
        Logger.warning("Stream end with no active stream")
        state
    end
  end

  defp handle_message(%{"type" => "stream_error"} = msg, state) do
    error = msg["error"]
    Logger.error("Stream error: #{error}")

    case Enum.find(state.streams, fn {_ref, s} -> not s.done end) do
      {ref, stream} ->
        if stream.waiter, do: GenServer.reply(stream.waiter, {:error, error})
        if stream.collector, do: GenServer.reply(stream.collector, {:error, error})
        timers = cancel_stream_timer(state.stream_timers, ref)
        %{state | streams: Map.delete(state.streams, ref), stream_timers: timers}
      nil ->
        state
    end
  end

  # =========================================================================
  # Helpers
  # =========================================================================

  defp send_frame(port, msg) do
    data = Msgpax.pack!(msg)
    len = byte_size(data) + @frame_header_bytes
    frame = <<len::32-unsigned-big-integer>> <> data
    send(port, {self(), {:command, frame}})
  end

  defp parse_frames(binary) do
    parse_frames_loop(binary, [])
  end

  defp parse_frames_loop(<<len::32-unsigned-big-integer, rest::binary>>, acc)
       when byte_size(rest) >= len - @frame_header_bytes do
    payload_size = len - @frame_header_bytes
    <<payload::binary-size(payload_size), remaining::binary>> = rest

    msg = Msgpax.unpack!(payload)
    parse_frames_loop(remaining, [msg | acc])
  end

  defp parse_frames_loop(partial, acc) do
    {Enum.reverse(acc), partial}
  end

  defp stringify_keys(kwlist) do
    Map.new(kwlist, fn {k, v} -> {to_string(k), v} end)
  end

  # -- WAV builder ------------------------------------------------------------

  # Build a valid WAV file from raw float32 PCM chunks.
  # Chunks are raw float32 interleaved samples.
  defp build_wav(chunks, sample_rate) when is_list(chunks) and is_integer(sample_rate) do
    pcm = IO.iodata_to_binary(chunks)
    data_size = byte_size(pcm)
    # WAV header: 44 bytes for PCM
    riff_size = data_size + 36
    header = <<
      "RIFF", riff_size::32-little-unsigned,
      "WAVE",
      "fmt ", 16::32-little-unsigned,
      3::16-little-unsigned,        # IEEE float
      1::16-little-unsigned,        # mono
      sample_rate::32-little-unsigned,
      sample_rate * 4::32-little-unsigned,  # byte rate (float32 mono)
      4::16-little-unsigned,        # block align
      32::16-little-unsigned,       # bits per sample
      "data", data_size::32-little-unsigned
    >>
    header <> pcm
  end

  defp build_wav(_, _), do: <<>>

  # -- Stream timer helpers ---------------------------------------------------

  defp reset_stream_timer(timers, ref) do
    if Map.has_key?(timers, ref) do
      _ = Process.cancel_timer(timers[ref])
      Map.put(timers, ref, Process.send_after(self(), {:stream_timeout, ref}, @stream_ttl_ms))
    else
      timers
    end
  end

  defp cancel_stream_timer(timers, ref) do
    if Map.has_key?(timers, ref) do
      Process.cancel_timer(timers[ref])
      Map.delete(timers, ref)
    else
      timers
    end
  end

  # -- Telemetry --------------------------------------------------------------

  defp emit_telemetry(event, t0, extra \\ %{}) do
    t1 = System.monotonic_time()
    duration_us = System.convert_time_unit(t1 - t0, :native, :microsecond)

    :telemetry.execute([:voxcpmex, event], %{duration_us: duration_us}, extra)
  end
end
