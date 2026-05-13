defmodule VoxCPMExTest do
  use ExUnit.Case
  doctest VoxCPMEx

  test "module exists" do
    assert is_list(VoxCPMEx.module_info())
  end

  test "info/1 and stop/1 are exported" do
    assert {:info, 1} in VoxCPMEx.__info__(:functions)
    assert {:stop, 1} in VoxCPMEx.__info__(:functions)
  end

  test "generate/2,3,4 are exported" do
    funcs = VoxCPMEx.__info__(:functions)
    assert {:generate, 2} in funcs
    assert {:generate, 3} in funcs
    assert {:generate, 4} in funcs
  end

  test "streaming API is exported" do
    funcs = VoxCPMEx.__info__(:functions)
    assert {:generate_streaming_async, 2} in funcs
    assert {:generate_streaming_async, 3} in funcs
    assert {:next_chunk, 2} in funcs
    assert {:collect_stream, 2} in funcs
  end

  test "LoRA API is exported" do
    funcs = VoxCPMEx.__info__(:functions)
    assert {:load_lora, 2} in funcs
    assert {:unload_lora, 1} in funcs
  end
end
