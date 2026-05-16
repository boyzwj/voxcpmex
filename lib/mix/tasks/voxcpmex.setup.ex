defmodule Mix.Tasks.Voxcpmex.Setup do
  @moduledoc """
  Installs Python dependencies for VoxCPMEx.

  Uses `uv` for fast dependency management — auto-installs it if missing.

  ## Usage

      # CUDA (NVIDIA GPU) — default
      mix voxcpmex.setup

      # Apple Silicon
      mix voxcpmex.setup --mps

      # CPU-only
      mix voxcpmex.setup --cpu

      # With virtual environment
      mix voxcpmex.setup --venv .venv
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

    device =
      cond do
        opts[:cpu] -> "cpu"
        opts[:mps] -> "mps"
        true -> "cuda"
      end

    venv = opts[:venv] || ".venv"
    skip_torch = opts[:no_torch]

    uv = ensure_uv!()
    {python, pip_args} = setup_venv!(uv, venv)

    IO.puts("==> Using Python: #{python}")
    IO.puts("==> Target device: #{device}")

    unless skip_torch do
      IO.puts("==> Installing PyTorch + torchaudio...")
      args = ["pip", "install"] ++ pip_args ++ torch_install_args(device)

      case System.cmd(uv, args, into: IO.stream(:stdio, :line), stderr_to_stdout: true) do
        {_, 0} -> IO.puts("==> PyTorch installed successfully")
        {_, code} -> IO.puts("==> Warning: PyTorch install returned exit code #{code} — continuing")
      end
    end

    IO.puts("==> Installing voxcpm + msgpack...")
    vox_args = ["pip", "install"] ++ pip_args ++ ["voxcpm", "soundfile", "msgpack"]

    case System.cmd(uv, vox_args, into: IO.stream(:stdio, :line), stderr_to_stdout: true) do
      {_, 0} ->
        IO.puts("==> voxcpm + msgpack installed successfully")

      {_, code} ->
        IO.puts(:stderr, "==> Error: voxcpm install failed with exit code #{code}")
        System.halt(1)
    end

    IO.puts("==> Verifying installation...")

    verify_cmd =
      ~s[import voxcpm; import torch; print(f'voxcpm OK, torch {torch.__version__}, cuda={torch.cuda.is_available()}, mps={getattr(torch.backends, "mps", type(None,(),{"is_available":lambda:False})()).is_available()}')]

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

  defp ensure_uv! do
    case System.find_executable("uv") do
      nil ->
        IO.puts("==> uv not found, installing via pip...")
        python = System.find_executable("python3") || System.find_executable("python") || "python3"

        case System.cmd(python, ["-m", "pip", "install", "uv"], stderr_to_stdout: true) do
          {_, 0} ->
            IO.puts("==> uv installed")
            "uv"

          {output, code} ->
            IO.puts(:stderr, "==> Failed to install uv (exit #{code}):\n#{output}")
            System.halt(1)
        end

      path ->
        IO.puts("==> Using uv: #{path}")
        path
    end
  end

  defp setup_venv!(_uv, nil) do
    python = System.find_executable("python3") || System.find_executable("python") || "python3"
    {python, ["--system"]}
  end

  defp setup_venv!(uv, venv_path) do
    IO.puts("==> Creating virtual environment: #{venv_path}")

    case System.cmd(uv, ["venv", venv_path]) do
      {_, 0} -> :ok
      {_, code} ->
        IO.puts(:stderr, "==> Failed to create virtual environment (exit #{code})")
        System.halt(1)
    end

    python = Path.join([venv_path, "bin", "python"])
    {python, []}
  end

  defp torch_install_args("cpu") do
    ["torch", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu"]
  end

  defp torch_install_args(_device) do
    ["torch", "torchaudio"]
  end
end
