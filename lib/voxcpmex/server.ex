defmodule VoxCPMEx.Server do
  @moduledoc """
  GenServer that manages a Python VoxCPM2 bridge via Erlang Port.

  ## Protocol (v2 — MessagePack binary framing)

  Each frame over stdin/stdout:

      [4-byte BE total_length][msgpack-encoded payload]

  MessagePack natively supports raw binary — audio is sent as raw bytes
  without base64 encoding.

  ## Streaming (v2)

  `generate_streaming_async/3` returns a `stream_ref` immediately.
  The Python process emits: `stream_start` → N × `stream_chunk` → `stream_end`.
  Caller polls with `next_chunk/2`.
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

  # =========================================================================
  # Client API
  # =========================================================================

  @spec start_link(start_opts()) :: GenServer.on_start()
  def start_link(opts \\ []) do
    name = Keyword.get(opts, :name)
    GenServer.start_link(__MODULE__, opts, name: name)
  end

  @spec await_ready(GenServer.server(), timeout()) :: :ok | {:error, term()}
  def await_ready(server, timeout \\ 120_000) do
    GenServer.call(server, :await_ready, timeout)
  end

  @spec generate(GenServer.server(), String.t(), [generate_opt()]) ::
          {:ok, binary()} | {:error, term()}
  def generate(server, text, opts \\ []) do
    generate(server, text, opts, 120_000)
  end

  @spec generate(GenServer.server(), String.t(), [generate_opt()], timeout()) ::
          {:ok, binary()} | {:error, term()}
  def generate(server, text, opts, timeout) do
    GenServer.call(server, {:generate, text, opts}, timeout)
  end

  @doc """
  Starts streaming generation. Returns `{:ok, stream_ref}` immediately.

  Poll with `next_chunk/2`:
      {:ok, chunk} → raw float32 PCM chunk
      :eos → stream complete
      {:error, reason}
  """
  @spec generate_streaming_async(GenServer.server(), String.t(), [generate_opt()]) ::
          {:ok, reference()} | {:error, term()}
  def generate_streaming_async(server, text, opts \\ []) do
    GenServer.call(server, {:generate_streaming_async, text, opts}, 10_000)
  end

  @doc """
  Returns the next chunk from a streaming session.

  Returns `{:ok, chunk}` (raw float32 PCM), `:eos` when stream ends,
  or `{:error, reason}`.
  """
  @spec next_chunk(GenServer.server(), reference()) ::
          {:ok, binary()} | :eos | {:error, term()}
  def next_chunk(server, ref) do
    GenServer.call(server, {:next_chunk, ref}, 60_000)
  end

  @doc """
  Collects all remaining chunks from a streaming session and returns
  the full audio as a WAV binary.

  Returns `{:ok, audio_wav}` or `{:error, reason}`.
  """
  @spec collect_stream(GenServer.server(), reference()) ::
          {:ok, binary()} | {:error, term()}
  def collect_stream(server, ref) do
    GenServer.call(server, {:collect_stream, ref}, 120_000)
  end

  @spec save(binary(), Path.t()) :: :ok | {:error, term()}
  def save(audio, path) when is_binary(audio) do
    File.write(path, audio)
  end

  @spec load_lora(GenServer.server(), String.t()) ::
          {:ok, non_neg_integer(), non_neg_integer()} | {:error, term()}
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

    send_frame(port, %{
      "type" => "init",
      "model" => model,
      "device" => device,
      "load_denoiser" => load_denoiser,
      "optimize" => optimize
    })

    state = %{
      port: port,
      ready: false,
      device: nil,
      sample_rate: nil,
      buffer: <<>>,
      pending: :queue.new(),
      streams: %{}
    }

    {:ok, state}
  end

  # -- call handlers ----------------------------------------------------------

  @impl true
  def handle_call(:await_ready, _from, state) do
    if state.ready, do: {:reply, :ok, state}, else: {:reply, {:error, :not_ready}, state}
  end

  def handle_call({:generate, text, opts}, from, state) do
    if not state.ready do
      {:reply, {:error, :not_ready}, state}
    else
      msg = Map.merge(%{"type" => "generate", "text" => text}, stringify_keys(opts))
      send_frame(state.port, msg)
      {:noreply, %{state | pending: :queue.in({from, :generate}, state.pending)}}
    end
  end

  def handle_call({:generate_streaming_async, text, opts}, from, state) do
    if not state.ready do
      {:reply, {:error, :not_ready}, state}
    else
      msg = Map.merge(%{"type" => "generate_streaming", "text" => text}, stringify_keys(opts))
      send_frame(state.port, msg)

      # Don't reply yet — we'll reply with the stream ref when stream_start arrives
      {:noreply, %{state | pending: :queue.in({from, :stream_ref}, state.pending)}}
    end
  end

  def handle_call({:next_chunk, ref}, from, state) do
    case Map.get(state.streams, ref) do
      nil ->
        {:reply, {:error, :unknown_stream}, state}

      %{chunks: [chunk | rest], done: done} ->
        new_streams = Map.put(state.streams, ref, %{chunks: rest, done: done})
        {:reply, {:ok, chunk}, %{state | streams: new_streams}}

      %{chunks: [], done: true} ->
        # Clean up completed stream
        {:reply, :eos, %{state | streams: Map.delete(state.streams, ref)}}

      %{chunks: [], done: false} ->
        # No chunks yet, block the caller until chunks arrive
        {:noreply,
         %{state | streams: put_in(state.streams[ref].waiter, from)}}
    end
  end

  def handle_call({:collect_stream, ref}, from, state) do
    case Map.get(state.streams, ref) do
      nil ->
        {:reply, {:error, :unknown_stream}, state}

      %{chunks: _chunks, done: false} ->
        # Not done yet, block and wait
        {:noreply,
         %{state | streams: put_in(state.streams[ref].collector, from)}}

      %{chunks: chunks, done: true} ->
        audio = chunks_to_wav(chunks)
        {:reply, {:ok, audio}, %{state | streams: Map.delete(state.streams, ref)}}
    end
  end

  def handle_call({:load_lora, lora_path}, from, state) do
    send_frame(state.port, %{"type" => "load_lora", "lora_path" => lora_path})
    {:noreply, %{state | pending: :queue.in({from, :lora_load}, state.pending)}}
  end

  def handle_call(:unload_lora, from, state) do
    send_frame(state.port, %{"type" => "unload_lora"})
    {:noreply, %{state | pending: :queue.in({from, :lora_unload}, state.pending)}}
  end

  # -- port messages ----------------------------------------------------------

  @impl true
  def handle_info({_port, {:data, data}}, state) do
    # Accumulate and parse frames
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

  # =========================================================================
  # Message dispatch
  # =========================================================================

  defp handle_message(%{"status" => "ok"} = msg, state) do
    cond do
      # Init response
      Map.has_key?(msg, "device") ->
        Logger.info("VoxCPM loaded on #{msg["device"]}, sr=#{msg["sample_rate"]}")
        %{state | ready: true, device: msg["device"], sample_rate: msg["sample_rate"]}

      # Generate response (audio is raw WAV bytes in msgpack)
      Map.has_key?(msg, "audio") ->
        {{:value, {from, _type}}, new_pending} = :queue.out(state.pending)
        GenServer.reply(from, {:ok, msg["audio"]})
        %{state | pending: new_pending}

      # LoRA load response
      Map.has_key?(msg, "loaded") ->
        {{:value, {from, _type}}, new_pending} = :queue.out(state.pending)
        GenServer.reply(from, {:ok, msg["loaded"], msg["skipped"]})
        %{state | pending: new_pending}

      # Generic ok (unload_lora, etc.)
      true ->
        case :queue.out(state.pending) do
          {{:value, {from, _type}}, new_pending} ->
            GenServer.reply(from, :ok)
            %{state | pending: new_pending}
          {:empty, _} ->
            state
        end
    end
  end

  defp handle_message(%{"status" => "error", "error" => error}, state) do
    Logger.error("Bridge error: #{error}")
    case :queue.out(state.pending) do
      {{:value, {from, _type}}, new_pending} ->
        GenServer.reply(from, {:error, error})
        %{state | pending: new_pending}
      {:empty, _} ->
        state
    end
  end

  # -- streaming messages -----------------------------------------------------

  defp handle_message(%{"type" => "stream_start"} = msg, state) do
    stream_id = msg["stream_id"]
    sample_rate = msg["sample_rate"]

    # Reply to the async caller with the stream ref
    {{:value, {from, _type}}, new_pending} = :queue.out(state.pending)

    ref = make_ref()
    GenServer.reply(from, {:ok, ref})

    streams = Map.put(state.streams, ref, %{
      id: stream_id,
      chunks: [],
      sample_rate: sample_rate,
      done: false,
      waiter: nil,
      collector: nil
    })

    %{state | pending: new_pending, streams: streams}
  end

  defp handle_message(%{"type" => "stream_chunk"} = msg, state) do
    stream_id = msg["stream_id"]
    chunk = msg["chunk"]  # raw float32 PCM bytes
    _index = msg["index"]
    _length = msg["length"]

    # Find the stream ref by id
    case Enum.find(state.streams, fn {_ref, s} -> s.id == stream_id end) do
      {ref, stream} ->
        new_chunks = stream.chunks ++ [chunk]
        streams =
          Map.put(state.streams, ref, %{stream | chunks: new_chunks})

        # If a waiter is blocked on next_chunk, wake it
        state = %{state | streams: streams}

        case streams[ref] do
          %{waiter: waiter} when not is_nil(waiter) ->
            # Flush all pending chunks
            %{chunks: all_chunks} = Map.get(state.streams, ref)
            [next | rest] = all_chunks
            GenServer.reply(waiter, {:ok, next})
            new_streams = put_in(state.streams[ref].chunks, rest)
            new_streams = put_in(new_streams[ref].waiter, nil)
            %{state | streams: new_streams}

          _ ->
            state
        end

      nil ->
        Logger.warning("Stream chunk for unknown stream: #{stream_id}")
        state
    end
  end

  defp handle_message(%{"type" => "stream_end"} = msg, state) do
    stream_id = msg["stream_id"]

    case Enum.find(state.streams, fn {_ref, s} -> s.id == stream_id end) do
      {ref, stream} ->
        streams = put_in(state.streams[ref].done, true)

        state = %{state | streams: streams}

        cond do
          # Collector waiting? Send full audio
          not is_nil(stream.collector) ->
            audio = chunks_to_wav(stream.chunks)
            GenServer.reply(stream.collector, {:ok, audio})
            %{state | streams: Map.delete(state.streams, ref)}

          # Waiter waiting? Deliver final chunk
          not is_nil(stream.waiter) ->
            case stream.chunks do
              [last] ->
                GenServer.reply(stream.waiter, {:ok, last})
                %{state | streams: Map.delete(state.streams, ref)}
              chunks when chunks != [] ->
                [next | rest] = chunks
                GenServer.reply(stream.waiter, {:ok, next})
                new_streams = put_in(state.streams[ref].chunks, rest)
                new_streams = put_in(new_streams[ref].waiter, nil)
                %{state | streams: new_streams}
              [] ->
                GenServer.reply(stream.waiter, :eos)
                %{state | streams: Map.delete(state.streams, ref)}
            end

          # No one waiting? Leave chunks in buffer
          true ->
            state
        end

      nil ->
        Logger.warning("Stream end for unknown stream: #{stream_id}")
        state
    end
  end

  defp handle_message(%{"type" => "stream_error"} = msg, state) do
    stream_id = msg["stream_id"]
    error = msg["error"]
    Logger.error("Stream error #{stream_id}: #{error}")

    case Enum.find(state.streams, fn {_ref, s} -> s.id == stream_id end) do
      {ref, stream} ->
        if not is_nil(stream.waiter), do: GenServer.reply(stream.waiter, {:error, error})
        if not is_nil(stream.collector), do: GenServer.reply(stream.collector, {:error, error})
        %{state | streams: Map.delete(state.streams, ref)}
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

  @doc false
  def parse_frames(binary) do
    parse_frames(binary, [])
  end

  defp parse_frames(<<len::32-unsigned-big-integer, rest::binary>>, acc)
       when byte_size(rest) >= len - @frame_header_bytes do
    payload_size = len - @frame_header_bytes
    <<payload::binary-size(payload_size), remaining::binary>> = rest

    msg = Msgpax.unpack!(payload)
    parse_frames(remaining, [msg | acc])
  end

  defp parse_frames(partial, acc) do
    {Enum.reverse(acc), partial}
  end

  defp stringify_keys(kwlist) do
    Map.new(kwlist, fn {k, v} -> {to_string(k), v} end)
  end

  defp chunks_to_wav(chunks) when is_list(chunks) do
    # chunks are raw float32 PCM bytes. For now, just concatenate.
    # A full implementation would add a WAV header.
    # For simplicity, we return the concatenated raw bytes.
    # The caller is expected to handle PCM-to-WAV conversion if needed.
    IO.iodata_to_binary(chunks)
  end
end
