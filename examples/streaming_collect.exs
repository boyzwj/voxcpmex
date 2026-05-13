# examples/streaming_collect.exs
#
# Streaming with auto-collect — start async, collect all at once.
# Simpler than polling, still gets the benefits of streaming pipeline.
#
# Usage:
#   mix run examples/streaming_collect.exs
#   mix run examples/streaming_collect.exs --text "Hello world" --output out.wav

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

text = opts[:text] || "This example shows the simpler collect_stream API. " <>
                      "Start generation, then collect all chunks at once."
device = opts[:device] || "cuda"
output = opts[:output] || "streaming_collect.wav"
steps = opts[:steps] || 10
cfg = opts[:cfg] || 2.0

IO.puts("==> Streaming (collect mode)")
IO.puts("==> Text: #{text}")

{:ok, pid} = VoxCPMEx.start_link(device: device)
:ok = VoxCPMEx.await_ready(pid, 120_000)

# Start, then collect
{:ok, ref} = VoxCPMEx.generate_streaming_async(pid, text,
  inference_timesteps: steps,
  cfg_value: cfg
)

IO.puts("==> Generating (streaming pipeline, collecting at end)...")
{:ok, audio} = VoxCPMEx.collect_stream(pid, ref)

:ok = VoxCPMEx.save(audio, output)
IO.puts("==> Saved #{output} (#{byte_size(audio)} bytes)")
IO.puts("==> Done! 🎵")
