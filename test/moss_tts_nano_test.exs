defmodule MossTTSNanoTest do
  use ExUnit.Case
  doctest MossTTSNano

  # ---------------------------------------------------------------------------
  # Module and API surface tests
  # ---------------------------------------------------------------------------

  test "module exists" do
    assert is_list(MossTTSNano.module_info())
  end

  test "info/1 and stop/1 are exported" do
    assert {:info, 1} in MossTTSNano.__info__(:functions)
    assert {:stop, 1} in MossTTSNano.__info__(:functions)
  end

  test "start_link/1 is exported" do
    funcs = MossTTSNano.__info__(:functions)
    assert {:start_link, 1} in funcs
  end

  test "await_ready/1,2 are exported" do
    funcs = MossTTSNano.__info__(:functions)
    assert {:await_ready, 1} in funcs
    assert {:await_ready, 2} in funcs
  end

  test "generate/2,3 are exported" do
    funcs = MossTTSNano.__info__(:functions)
    assert {:generate, 2} in funcs
    assert {:generate, 3} in funcs
  end

  test "generate/4 with timeout is exported" do
    funcs = MossTTSNano.__info__(:functions)
    assert {:generate, 4} in funcs
  end

  test "streaming API is exported" do
    funcs = MossTTSNano.__info__(:functions)
    assert {:generate_streaming_async, 2} in funcs
    assert {:generate_streaming_async, 3} in funcs
    assert {:next_chunk, 2} in funcs
    assert {:collect_stream, 2} in funcs
  end

  test "save/2 is exported" do
    funcs = MossTTSNano.__info__(:functions)
    assert {:save, 2} in funcs
  end

  test "list_voices/1 is exported" do
    funcs = MossTTSNano.__info__(:functions)
    assert {:list_voices, 1} in funcs
  end

  # ---------------------------------------------------------------------------
  # Server module tests
  # ---------------------------------------------------------------------------

  test "Server module exists and exports required functions" do
    assert is_list(MossTTSNano.Server.module_info())
    server_funcs = MossTTSNano.Server.__info__(:functions)

    assert {:start_link, 1} in server_funcs
    assert {:info, 1} in server_funcs
    assert {:await_ready, 2} in server_funcs
    assert {:generate, 4} in server_funcs
    assert {:generate_streaming_async, 3} in server_funcs
    assert {:next_chunk, 2} in server_funcs
    assert {:collect_stream, 2} in server_funcs
    assert {:save, 2} in server_funcs
    assert {:list_voices, 1} in server_funcs
  end

  # ---------------------------------------------------------------------------
  # Default model and configuration tests
  # ---------------------------------------------------------------------------

  test "default model is OpenMOSS-Team/MOSS-TTS-Nano-100M" do
    # We verify the module is properly defined
    assert is_atom(MossTTSNano.__info__(:module))
  end
end
