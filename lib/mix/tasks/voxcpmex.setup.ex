defmodule Mix.Tasks.Voxcpmex.Setup do
  @moduledoc """
  Installs Python dependencies for VoxCPMEx.

  ## Usage

      # CUDA (NVIDIA GPU) — default
      mix voxcpmex.setup

      # Apple Silicon
      mix voxcpmex.setup --mps

      # CPU-only
      mix voxcpmex.setup --cpu

      # With virtual environment
      mix voxcpmex.setup --cuda --venv .venv
  """

  use Mix.Task

  @shortdoc "Install Python dependencies for VoxCPMEx"

  @impl Mix.Task
  def run(argv) do
    {opts, _args, _invalid} =
      OptionParser.parse(argv,
        switches: [
          cuda: :boolean,
          mps: :boolean,
          cpu: :boolean,
          venv: :string,
          no_torch: :boolean
        ]
      )

    device = cond do
      opts[:cpu] -> "cpu"
      opts[:mps] -> "mps"
      true -> "cuda"
    end

    venv = opts[:venv]
    skip_torch = opts[:no_torch]

    # Resolve python/pip paths
    {python, pip} = resolve_python_pip(venv)

    IO.puts("==> Using Python: #{python}")
    IO.puts("==> Target device: #{device}")

    # Install PyTorch
    unless skip_torch do
      IO.puts("==> Installing PyTorch + torchaudio...")
      torch_cmd = case device do
        "cuda" -> ["install", "torch", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu121"]
        "mps"  -> ["install", "torch", "torchaudio"]
        "cpu"  -> ["install", "torch", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu"]
      end

      case System.cmd(pip, torch_cmd, into: IO.stream(:stdio, :line), stderr_to_stdout: true) do
        {_, 0} -> IO.puts("==> PyTorch installed successfully")
        {_, code} -> IO.puts("==> Warning: PyTorch install returned exit code #{code} — continuing")
      end
    end

    # Install voxcpm
    IO.puts("==> Installing voxcpm...")
    case System.cmd(pip, ["install", "voxcpm", "soundfile"], into: IO.stream(:stdio, :line), stderr_to_stdout: true) do
      {_, 0} -> IO.puts("==> voxcpm installed successfully")
      {_, code} ->
        IO.puts(:stderr, "==> Error: voxcpm install failed with exit code #{code}")
        System.halt(1)
    end

    # Verify
    IO.puts("==> Verifying installation...")
    verify_cmd = ~s[import voxcpm; import torch; print(f'voxcpm OK, torch {torch.__version__}, cuda={torch.cuda.is_available()}, mps={getattr(torch.backends, "mps", type(None,(),{"is_available":lambda:False})()).is_available()}')]

    case System.cmd(python, ["-c", verify_cmd], stderr_to_stdout: true) do
      {output, 0} ->
        IO.puts("==> Verification: #{String.trim(output)}")
        IO.puts("""
        ==>
        ==> ✅ VoxCPMEx Python dependencies installed!
        ==>
        ==> Next steps:
        ==>   1. Start a server: {:ok, pid} = VoxCPMEx.start_link(device: "#{device}")
        ==>   2. Wait for loading: :ok = VoxCPMEx.await_ready(pid)
        ==>   3. Generate speech: {:ok, audio} = VoxCPMEx.generate(pid, "Hello!")
        ==>   4. Save to file: :ok = VoxCPMEx.save(audio, "output.wav")
        """)
      {output, code} ->
        IO.puts(:stderr, "==> Verification failed (exit #{code}):\n#{output}")
        System.halt(1)
    end
  end

  defp resolve_python_pip(nil) do
    python = System.find_executable("python3") || System.find_executable("python") || "python3"
    pip = System.find_executable("pip3") || System.find_executable("pip") || "pip3"
    {python, pip}
  end

  defp resolve_python_pip(venv) do
    IO.puts("==> Creating virtual environment: #{venv}")
    python_base = System.find_executable("python3") || System.find_executable("python") || "python3"
    System.cmd(python_base, ["-m", "venv", venv])
    python = Path.join([venv, "bin", "python"])
    pip = Path.join([venv, "bin", "pip"])
    {python, pip}
  end
end
