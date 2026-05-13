# examples/hello_world.exs
#
# Basic Text-to-Speech example with proper lifecycle.
#
# Usage:
#   mix run examples/hello_world.exs
#   mix run examples/hello_world.exs --text "你好世界"
#   mix run examples/hello_world.exs --text "Hello" --device cpu

{opts, _args, _invalid} =
  OptionParser.parse(System.argv(),
    switches: [text: :string, device: :string, output: :string, steps: :integer, cfg: :float]
  )

text = opts[:text] || "Hello, world! This is VoxCPM2 running through Elixir."
device = opts[:device] || "cuda"
output = opts[:output] || "hello_world.wav"
steps = opts[:steps] || 10
cfg = opts[:cfg] || 2.0

IO.puts("==> Starting VoxCPM2 on #{device}...")
{:ok, pid} = VoxCPMEx.start_link(device: device)

# Check model info
info = VoxCPMEx.info(pid)
IO.puts("==> Info: #{inspect(info)}")

IO.puts("==> Waiting for model to load...")
:ok = VoxCPMEx.await_ready(pid, 120_000)
IO.puts("==> Model ready! device=#{info.device} sr=#{info.sample_rate}")

IO.puts("==> Generating: \"#{text}\"")
{:ok, audio} = VoxCPMEx.generate(pid, text,
  inference_timesteps: steps,
  cfg_value: cfg
)

:ok = VoxCPMEx.save(audio, output)
IO.puts("==> Audio saved to #{output} (#{byte_size(audio)} bytes)")

# Clean shutdown
VoxCPMEx.stop(pid)
IO.puts("==> Server stopped. Done! 🎉")
