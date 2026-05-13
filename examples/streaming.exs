# examples/streaming.exs
#
# True streaming TTS — get audio chunks as they're generated.
#
# Usage:
#   mix run examples/streaming.exs
#   mix run examples/streaming.exs --text "Long text for streaming"
#   mix run examples/streaming.exs --text "你好世界" --device cpu

{opts, _args, _invalid} =
  OptionParser.parse(System.argv(),
    switches: [
      text: :string,
      device: :string,
      output: :string,
      steps: :integer,
      cfg: :float
    ]
  )

text = opts[:text] || "Streaming synthesis delivers audio chunks as they are generated. " <>
                      "This means you can start playback immediately, " <>
                      "without waiting for the entire utterance to complete."
device = opts[:device] || "cuda"
output = opts[:output] || "streaming.wav"
steps = opts[:steps] || 10
cfg = opts[:cfg] || 2.0

IO.puts("==> Streaming TTS")
IO.puts("==> Text: #{String.slice(text, 0, 60)}...")
IO.puts("==> Device: #{device}")

{:ok, pid} = VoxCPMEx.start_link(device: device)
:ok = VoxCPMEx.await_ready(pid, 120_000)
IO.puts("==> Model ready!")

# Start async streaming
IO.puts("==> Starting stream...")
{:ok, ref} = VoxCPMEx.generate_streaming_async(pid, text,
  inference_timesteps: steps,
  cfg_value: cfg
)

# Collect chunks
chunks =
  Stream.unfold(ref, fn ref ->
    case VoxCPMEx.next_chunk(pid, ref) do
      {:ok, chunk} ->
        IO.puts("  chunk ##{length([])}: #{byte_size(chunk)} bytes")
        # For real-time playback, you'd send chunk to audio output here
        {chunk, ref}
      :eos ->
        IO.puts("  stream complete")
        nil
      {:error, reason} ->
        IO.puts(:stderr, "  error: #{reason}")
        nil
    end
  end)
  |> Enum.to_list()

IO.puts("==> Total chunks: #{length(chunks)}")
IO.puts("==> Done! ⚡")
