# examples/moss_tts_nano_hello.exs
#
# Basic MOSS-TTS-Nano Text-to-Speech example.
#
# This requires:
#   - Python with torch, transformers, soundfile, msgpack installed
#   - A CPU (4+ cores recommended) or CUDA GPU
#
# Usage:
#   mix run examples/moss_tts_nano_hello.exs
#   mix run examples/moss_tts_nano_hello.exs --text "你好世界"
#   mix run examples/moss_tts_nano_hello.exs --device cuda --voice "Ava"
#
{opts, _args, _invalid} =
  OptionParser.parse(System.argv(),
    switches: [
      text: :string,
      device: :string,
      voice: :string,
      output: :string
    ]
  )

text = opts[:text] || "Hello, world! This is MOSS-TTS-Nano, a tiny but powerful text-to-speech model running through Elixir."
device = opts[:device] || "cpu"
voice = opts[:voice]
output = opts[:output] || "moss_tts_nano_hello.wav"

IO.puts("==> Starting MOSS-TTS-Nano on #{device}...")
{:ok, pid} = MossTTSNano.start_link(device: device)

IO.puts("==> Waiting for model to load (this takes a moment on first run)...")
:ok = MossTTSNano.await_ready(pid, 180_000)

info = MossTTSNano.info(pid)
IO.puts("==> Model ready! device=#{info.device} sr=#{info.sample_rate}")

# List available voices
voices = MossTTSNano.list_voices(pid)
case voices do
  {:ok, v} ->
    IO.puts("==> Available voices: #{Enum.join(v, ", ")}")
  {:error, _} ->
    IO.puts("==> (Could not list voices)")
end

# Build generation options
gen_opts =
  if voice do
    [voice: voice]
  else
    []
  end

IO.puts("==> Generating: \"#{text}\"")
{:ok, audio} = MossTTSNano.generate(pid, text, gen_opts)

:ok = MossTTSNano.save(audio, output)
IO.puts("==> Audio saved to #{output} (#{byte_size(audio)} bytes)")

# Clean shutdown
MossTTSNano.stop(pid)
IO.puts("==> Server stopped. Done! 🎉")
