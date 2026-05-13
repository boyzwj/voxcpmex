defmodule VoxCPMEx.MixProject do
  use Mix.Project

  @version "0.3.0"
  @source_url "https://github.com/holsee/voxcpmex"

  def project do
    [
      app: :voxcpmex,
      version: @version,
      elixir: "~> 1.14",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      description: description(),
      package: package(),
      source_url: @source_url,
      docs: docs(),
      aliases: aliases()
    ]
  end

  def application do
    [
      extra_applications: [:logger],
      mod: {VoxCPMEx.Application, []}
    ]
  end

  defp deps do
    [
      {:msgpax, "~> 2.4"},
      {:telemetry, "~> 1.2"},
      {:jason, "~> 1.4"},
      {:ex_doc, ">= 0.0.0", only: :dev, runtime: false}
    ]
  end

  defp description do
    """
    Elixir wrapper for VoxCPM2 — 2B-param, 30-language, 48kHz TTS with
    Voice Design, controllable cloning, and true chunk-by-chunk streaming.
    Uses MessagePack binary protocol for zero-overhead audio transfer.
    """
  end

  defp package do
    [
      name: "voxcpmex",
      licenses: ["Apache-2.0"],
      links: %{
        "GitHub" => @source_url,
        "VoxCPM2" => "https://huggingface.co/openbmb/VoxCPM2",
        "VoxCPM" => "https://github.com/OpenBMB/VoxCPM"
      },
      files: ~w(lib priv examples mix.exs README.md LICENSE .formatter.exs)
    ]
  end

  defp docs do
    [
      main: "VoxCPMEx",
      source_ref: "v#{@version}",
      source_url: @source_url
    ]
  end

  defp aliases do
    [
      setup: ["voxcpmex.setup"]
    ]
  end
end
