# examples/voice_design.exs
#
# Voice Design example — generate a novel voice from a natural-language description.
# No reference audio needed!
#
# Usage:
#   mix run examples/voice_design.exs
#   mix run examples/voice_design.exs --text "你好，欢迎来到语音合成演示"
#   mix run examples/voice_design.exs --control "A young woman, gentle and sweet voice"
#   mix run examples/voice_design.exs --device cpu

{opts, _args, _invalid} =
  OptionParser.parse(System.argv(),
    switches: [
      text: :string,
      control: :string,
      device: :string,
      output: :string,
      steps: :integer,
      cfg: :float
    ]
  )

control = opts[:control] || "A warm, professional female voice, calm and clear"
text = opts[:text] || "Hello, welcome to VoxCPM2 voice design! This voice was created purely from a text description."
device = opts[:device] || "cuda"
output = opts[:output] || "voice_design.wav"
steps = opts[:steps] || 15
cfg = opts[:cfg] || 2.0

# Voice Design: prepend control description in parentheses
full_text = "(#{control}) #{text}"

IO.puts("==> Voice Design")
IO.puts("==> Control: #{control}")
IO.puts("==> Text: #{text}")
IO.puts("==> Device: #{device}")

{:ok, pid} = VoxCPMEx.start_link(device: device)
:ok = VoxCPMEx.await_ready(pid, 120_000)
IO.puts("==> Model ready!")

{:ok, audio} = VoxCPMEx.generate(pid, full_text,
  inference_timesteps: steps,
  cfg_value: cfg
)

:ok = VoxCPMEx.save(audio, output)
IO.puts("==> Audio saved to #{output} (#{byte_size(audio)} bytes)")
IO.puts("==> Done! 🔊")

# Try different voice descriptions:
#
#   "A young woman, gentle and sweet voice"
#   "A deep male voice, authoritative and confident"
#   "An elderly person, wise and slow-paced"
#   "A cheerful child, energetic and bright"
#   "A calm narrator, suitable for audiobooks"
#   "A robot voice, mechanical and precise"
#   "A news anchor, professional and articulate"
