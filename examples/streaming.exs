# examples/streaming.exs
#
# Streaming TTS example — uses VoxCPM2's streaming pipeline for progressive output.
#
# Usage:
#   mix run examples/streaming.exs
#   mix run examples/streaming.exs --text "This is a long text that benefits from streaming synthesis."
#   mix run examples/streaming.exs --text "你好,这是一个流式语音合成示例" --device cpu

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

text = opts[:text] || "Streaming synthesis is great for real-time applications. " <>
                      "You can start playing audio while the rest is still being generated. " <>
                      "This reduces perceived latency significantly."
device = opts[:device] || "cuda"
output = opts[:output] || "streaming.wav"
steps = opts[:steps] || 10
cfg = opts[:cfg] || 2.0

IO.puts("==> Streaming TTS")
IO.puts("==> Text: #{text}")
IO.puts("==> Device: #{device}")

{:ok, pid} = VoxCPMEx.start_link(device: device)
:ok = VoxCPMEx.await_ready(pid, 120_000)
IO.puts("==> Model ready!")

IO.puts("==> Generating with streaming...")
{:ok, result} = VoxCPMEx.generate_streaming(pid, text,
  inference_timesteps: steps,
  cfg_value: cfg
)

audio = result["audio"] || result[:audio]
duration = result["duration"] || result[:duration]
num_chunks = result["num_chunks"] || result[:num_chunks]
sample_rate = result["sample_rate"] || result[:sample_rate]

:ok = VoxCPMEx.save(audio, output)

IO.puts("==> Audio saved to #{output} (#{byte_size(audio)} bytes)")
IO.puts("==> Duration: #{Float.round(duration, 2)}s")
IO.puts("==> Chunks: #{num_chunks}")
IO.puts("==> Sample rate: #{sample_rate} Hz")
IO.puts("==> Done! ⚡")
