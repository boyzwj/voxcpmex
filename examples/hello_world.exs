# examples/hello_world.exs
#
# Basic Text-to-Speech example.
#
# Usage:
#   mix run examples/hello_world.exs
#   mix run examples/hello_world.exs --text "你好世界"
#   mix run examples/hello_world.exs --text "Hello" --device cpu
#   mix run examples/hello_world.exs --text "Hello" --device mps
#   mix run examples/hello_world.exs --text "Hello" --output my_audio.wav
#   mix run examples/hello_world.exs --text "Hello" --steps 30 --cfg 3.0

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

text = opts[:text] || "Hello, world! This is VoxCPM2 running through Elixir."
device = opts[:device] || "cuda"
output = opts[:output] || "hello_world.wav"
steps = opts[:steps] || 10
cfg = opts[:cfg] || 2.0

IO.puts("==> Starting VoxCPM2 on #{device}...")
{:ok, pid} = VoxCPMEx.start_link(device: device)

IO.puts("==> Waiting for model to load (may take 30-60s on first run)...")
:ok = VoxCPMEx.await_ready(pid, 120_000)
IO.puts("==> Model ready!")

IO.puts("==> Generating: \"#{text}\"")
{:ok, audio} = VoxCPMEx.generate(pid, text,
  inference_timesteps: steps,
  cfg_value: cfg
)

:ok = VoxCPMEx.save(audio, output)
IO.puts("==> Audio saved to #{output} (#{byte_size(audio)} bytes)")
IO.puts("==> Done! 🎉")
