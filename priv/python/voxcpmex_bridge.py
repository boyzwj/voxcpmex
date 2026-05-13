#!/usr/bin/env python3
"""
VoxCPMEx Bridge v2.1 — Python bridge for Elixir VoxCPMEx library.

Protocol: binary frames over stdin/stdout
  Frame: [4-byte big-endian total_length][msgpack-encoded message]

Audio is raw WAV bytes inside msgpack — no base64 encoding.

Streaming (v2.1): simplified — no stream IDs. The bridge processes one
streaming request at a time, emitting stream_start → N×stream_chunk →
stream_end in strict sequence.
"""

import sys
import io
import os
import struct
import signal

import msgpack
import numpy as np
import soundfile as sf

# ---------------------------------------------------------------------------
# Graceful shutdown on SIGPIPE / SIGTERM
# ---------------------------------------------------------------------------
_shutting_down = False

def _handle_signal(signum, frame):
    global _shutting_down
    _shutting_down = True
    sys.stderr.write(f"Bridge received signal {signum}, shutting down\n")
    sys.stderr.flush()

signal.signal(signal.SIGPIPE, signal.SIG_DFL)  # Let OS handle broken pipe
signal.signal(signal.SIGTERM, _handle_signal)

# ---------------------------------------------------------------------------
# PyTorch early import check
# ---------------------------------------------------------------------------
try:
    import torch
except ImportError as e:
    err = msgpack.dumps({"status": "error", "error": f"Missing dependency: {e}"})
    _write_frame(err)
    sys.exit(1)

_original_torch_load = torch.load

def _patched_torch_load(f, map_location=None, **kwargs):
    if map_location is None:
        map_location = "cpu"
    return _original_torch_load(f, map_location=map_location, **kwargs)

torch.load = _patched_torch_load


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def _read_exact(n: int) -> bytes:
    """Read exactly n bytes from stdin. Raises EOFError on EOF."""
    data = b""
    while len(data) < n:
        chunk = sys.stdin.buffer.read(n - len(data))
        if not chunk:
            raise EOFError("stdin closed")
        data += chunk
    return data


def _read_frame() -> dict:
    """Read one msgpack frame from stdin."""
    header = _read_exact(4)
    total_len = struct.unpack(">I", header)[0]
    payload = _read_exact(total_len - 4)
    return msgpack.loads(payload, raw=False)


def _write_frame(data: bytes) -> None:
    """Write one msgpack frame to stdout. Returns False on broken pipe."""
    try:
        frame_len = struct.pack(">I", len(data) + 4)
        sys.stdout.buffer.write(frame_len + data)
        sys.stdout.buffer.flush()
        return True
    except (BrokenPipeError, OSError):
        return False


def _send(msg: dict) -> bool:
    """Encode and send. Returns False on failure."""
    return _write_frame(msgpack.dumps(msg))


def _send_error(error: str) -> bool:
    return _send({"status": "error", "error": error})


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------
def _detect_device(requested: str) -> str:
    req = (requested or "cuda").strip().lower()
    if req.startswith("cuda"):
        return req if torch.cuda.is_available() else "cpu"
    if req == "mps":
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        return "mps" if has_mps else "cpu"
    return "cpu"


# ---------------------------------------------------------------------------
# VoxCPM Bridge
# ---------------------------------------------------------------------------
class VoxCPMBridge:
    def __init__(self):
        self.model = None
        self.device = None
        self.sample_rate = None

    def init_model(self, msg: dict) -> dict:
        try:
            from voxcpm import VoxCPM

            hf_model_id = msg.get("model", "openbmb/VoxCPM2")
            load_denoiser = msg.get("load_denoiser", False)
            optimize = msg.get("optimize", True)
            requested_device = msg.get("device", "cuda")

            self.device = _detect_device(requested_device)

            sys.stderr.write(f"Loading {hf_model_id} on {self.device}...\n")
            sys.stderr.flush()

            self.model = VoxCPM.from_pretrained(
                hf_model_id,
                load_denoiser=load_denoiser,
                device=self.device,
                optimize=optimize,
            )

            self.sample_rate = self.model.tts_model.sample_rate
            sys.stderr.write(f"Loaded. device={self.device} sr={self.sample_rate}\n")
            sys.stderr.flush()

            return {"status": "ok", "device": self.device, "sample_rate": self.sample_rate}

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def generate(self, msg: dict) -> dict:
        if self.model is None:
            return {"status": "error", "error": "Model not initialized"}

        try:
            text = msg["text"]
            wav = self.model.generate(
                text,
                cfg_value=msg.get("cfg_value", 2.0),
                inference_timesteps=msg.get("inference_timesteps", 10),
                min_len=msg.get("min_len", 2),
                max_len=msg.get("max_len", 4096),
                normalize=msg.get("normalize", False),
                denoise=msg.get("denoise", False),
                reference_wav_path=msg.get("audio_prompt"),
                prompt_wav_path=msg.get("prompt_wav_path"),
                prompt_text=msg.get("prompt_text"),
                retry_badcase=msg.get("retry_badcase", True),
                retry_badcase_max_times=msg.get("retry_badcase_max_times", 3),
                retry_badcase_ratio_threshold=msg.get("retry_badcase_ratio_threshold", 6.0),
            )

            audio_bytes = self._wav_to_bytes(wav, self.sample_rate)
            return {
                "status": "ok",
                "audio": audio_bytes,
                "sample_rate": self.sample_rate,
                "duration": len(wav) / self.sample_rate,
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def generate_streaming(self, msg: dict) -> None:
        """Generate speech with streaming — emits frames in strict sequence."""
        if self.model is None:
            _send_error("Model not initialized")
            return

        text = msg["text"]

        try:
            # Announce stream start
            if not _send({"type": "stream_start", "sample_rate": self.sample_rate}):
                return  # Elixir side disconnected

            idx = 0
            for chunk in self.model.generate_streaming(
                text,
                cfg_value=msg.get("cfg_value", 2.0),
                inference_timesteps=msg.get("inference_timesteps", 10),
                min_len=msg.get("min_len", 2),
                max_len=msg.get("max_len", 4096),
                normalize=msg.get("normalize", False),
                denoise=msg.get("denoise", False),
                reference_wav_path=msg.get("audio_prompt"),
                prompt_wav_path=msg.get("prompt_wav_path"),
                prompt_text=msg.get("prompt_text"),
                retry_badcase=msg.get("retry_badcase", False),
                retry_badcase_max_times=msg.get("retry_badcase_max_times", 3),
                retry_badcase_ratio_threshold=msg.get("retry_badcase_ratio_threshold", 6.0),
            ):
                if _shutting_down:
                    return

                chunk_bytes = chunk.astype(np.float32).tobytes()
                if not _send({
                    "type": "stream_chunk",
                    "chunk": chunk_bytes,
                    "index": idx,
                    "length": len(chunk),
                }):
                    return  # Elixir side disconnected
                idx += 1

            _send({
                "type": "stream_end",
                "total_chunks": idx,
            })

        except Exception as e:
            _send({"type": "stream_error", "error": str(e)})

    def load_lora(self, msg: dict) -> dict:
        if self.model is None:
            return {"status": "error", "error": "Model not initialized"}
        try:
            loaded, skipped = self.model.load_lora(msg["lora_path"])
            return {"status": "ok", "loaded": len(loaded), "skipped": len(skipped)}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def unload_lora(self) -> dict:
        if self.model is None:
            return {"status": "error", "error": "Model not initialized"}
        try:
            self.model.unload_lora()
            return {"status": "ok"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _wav_to_bytes(self, wav: np.ndarray, sr: int) -> bytes:
        buf = io.BytesIO()
        if wav.dtype != np.float32:
            wav = wav.astype(np.float32)
        sf.write(buf, wav, sr, format="WAV")
        buf.seek(0)
        return buf.read()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    bridge = VoxCPMBridge()

    while not _shutting_down:
        try:
            msg = _read_frame()
        except EOFError:
            sys.stderr.write("stdin closed, exiting\n")
            sys.stderr.flush()
            break
        except Exception as e:
            if not _send_error(f"Frame read error: {e}"):
                break
            continue

        msg_type = msg.get("type")

        try:
            if msg_type == "init":
                _send(bridge.init_model(msg))

            elif msg_type == "generate":
                _send(bridge.generate(msg))

            elif msg_type == "generate_streaming":
                bridge.generate_streaming(msg)

            elif msg_type == "load_lora":
                _send(bridge.load_lora(msg))

            elif msg_type == "unload_lora":
                _send(bridge.unload_lora())

            elif msg_type == "ping":
                _send({"status": "ok", "message": "pong"})

            else:
                _send_error(f"Unknown request type: {msg_type}")

        except Exception as e:
            _send_error(f"Unhandled error: {e}")


if __name__ == "__main__":
    main()
