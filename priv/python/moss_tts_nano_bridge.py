#!/usr/bin/env python3
"""
MOSS-TTS-Nano Bridge — Python bridge for Elixir MossTTSNano.

Protocol: binary frames over stdin/stdout
  Frame: [4-byte big-endian total_length][msgpack-encoded message]

Audio is raw WAV bytes inside msgpack — no base64 encoding.

Streaming: emits stream_start → N×stream_chunk → stream_end in strict sequence.
"""

import sys
import io
import os
import struct
import signal

import msgpack
import numpy as np
import soundfile as sf
import torch

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


def _write_frame(data: bytes) -> bool:
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
# Device resolution
# ---------------------------------------------------------------------------
def _resolve_device(requested: str) -> torch.device:
    req = (requested or "cpu").strip().lower()
    if req == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if req.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(req)
        sys.stderr.write(
            f"CUDA not available, requested '{req}', falling back to cpu\n"
        )
        sys.stderr.flush()
        return torch.device("cpu")
    return torch.device(req)


# ---------------------------------------------------------------------------
# MOSS-TTS-Nano Bridge
# ---------------------------------------------------------------------------
class MOSSNanoBridge:
    DEFAULT_VOICES = {
        "Junhao",
        "Zhiming",
        "Weiguo",
        "Xiaoyu",
        "Yuewen",
        "Lingyu",
        "Trump",
        "Ava",
        "Bella",
        "Adam",
        "Nathan",
        "Sakura",
        "Yui",
        "Aoi",
        "Hina",
        "Mei",
    }

    def __init__(self):
        self.model = None
        self.audio_tokenizer = None
        self.device = None
        self.sample_rate = 48000  # Default for MOSS-TTS-Nano

    def init_model(self, msg: dict) -> dict:
        try:
            from transformers import AutoModelForCausalLM

            checkpoint = msg.get(
                "checkpoint", "OpenMOSS-Team/MOSS-TTS-Nano-100M"
            )
            requested_device = msg.get("device", "cpu")
            self.device = _resolve_device(requested_device)

            sys.stderr.write(
                f"Loading MOSS-TTS-Nano from {checkpoint} on {self.device}...\n"
            )
            sys.stderr.flush()

            # Load the model with trust_remote_code=True
            self.model = AutoModelForCausalLM.from_pretrained(
                checkpoint,
                trust_remote_code=True,
            )

            # Apply device and attention implementation
            self.model.to(device=self.device)
            # Use sdpa by default (works on CPU + CUDA)
            self.model._set_attention_implementation("sdpa")
            self.model.eval()

            sys.stderr.write(
                f"MOSS-TTS-Nano loaded. device={self.device} sr={self.sample_rate}\n"
            )
            sys.stderr.flush()

            # Return available voices (hardcoded defaults)
            return {
                "status": "ok",
                "device": str(self.device),
                "sample_rate": self.sample_rate,
                "voices": sorted(list(self.DEFAULT_VOICES)),
            }

        except Exception as e:
            import traceback

            traceback.print_exc(file=sys.stderr)
            return {"status": "error", "error": str(e)}

    def list_voices(self) -> dict:
        if self.model is None:
            return {"status": "error", "error": "Model not initialized"}
        return {"status": "ok", "voices": sorted(list(self.DEFAULT_VOICES))}

    def generate(self, msg: dict) -> dict:
        if self.model is None:
            return {"status": "error", "error": "Model not initialized"}

        try:
            text = msg["text"]
            mode = str(msg.get("mode", "voice_clone"))
            voice = msg.get("voice")
            prompt_audio_path = msg.get("prompt_audio_path")
            prompt_text = msg.get("prompt_text") or None
            max_new_frames = int(msg.get("max_new_frames", 375))
            voice_clone_max_text_tokens = int(
                msg.get("voice_clone_max_text_tokens", 75)
            )
            do_sample = bool(msg.get("do_sample", True))
            seed = msg.get("seed")

            # Sampling / decoding params
            text_temperature = float(msg.get("text_temperature", 1.0))
            text_top_p = float(msg.get("text_top_p", 1.0))
            text_top_k = int(msg.get("text_top_k", 50))
            audio_temperature = float(msg.get("audio_temperature", 0.8))
            audio_top_p = float(msg.get("audio_top_p", 0.95))
            audio_top_k = int(msg.get("audio_top_k", 25))
            audio_repetition_penalty = float(
                msg.get("audio_repetition_penalty", 1.2)
            )

            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

            # Run inference
            output_bytes = io.BytesIO()
            wav_path = output_bytes.name if hasattr(output_bytes, "name") else None

            result = self.model.inference(
                text=text,
                output_audio_path=None,
                mode=mode,
                prompt_text=prompt_text,
                prompt_audio_path=prompt_audio_path,
                text_tokenizer_path=self.model.name_or_path,
                audio_tokenizer_type="moss-tts-nano",
                audio_tokenizer_pretrained_name_or_path="OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano",
                device=self.device,
                max_new_frames=max_new_frames,
                voice_clone_max_text_tokens=voice_clone_max_text_tokens,
                do_sample=do_sample,
                use_kv_cache=True,
                text_temperature=text_temperature,
                text_top_p=text_top_p,
                text_top_k=text_top_k,
                audio_temperature=audio_temperature,
                audio_top_p=audio_top_p,
                audio_top_k=audio_top_k,
                audio_repetition_penalty=audio_repetition_penalty,
            )

            # result["waveform"] is (T,) or (C, T)
            waveform = result["waveform"].detach().cpu()
            sample_rate = int(result.get("sample_rate", self.sample_rate))

            # Ensure float32
            if not torch.is_floating_point(waveform):
                waveform = waveform.float()

            # Convert to numpy
            if waveform.ndim == 1:
                waveform_np = waveform.numpy().astype(np.float32)
            else:
                # (C, T) -> (T, C) if needed by soundfile
                waveform_np = waveform.numpy().astype(np.float32)
                if waveform_np.shape[0] <= 4 and waveform_np.shape[0] < waveform_np.shape[1]:
                    waveform_np = waveform_np.T

            # Write WAV
            buf = io.BytesIO()
            sf.write(buf, waveform_np, sample_rate, format="WAV")
            buf.seek(0)
            audio_bytes = buf.read()

            return {
                "status": "ok",
                "audio": audio_bytes,
                "sample_rate": sample_rate,
            }

        except Exception as e:
            import traceback

            traceback.print_exc(file=sys.stderr)
            return {"status": "error", "error": str(e)}

    def generate_streaming(self, msg: dict) -> None:
        """Generate speech with streaming — emits frames in strict sequence."""
        if self.model is None:
            _send_error("Model not initialized")
            return

        text = msg["text"]

        try:
            mode = str(msg.get("mode", "voice_clone"))
            prompt_audio_path = msg.get("prompt_audio_path")
            prompt_text = msg.get("prompt_text") or None
            max_new_frames = int(msg.get("max_new_frames", 375))
            voice_clone_max_text_tokens = int(
                msg.get("voice_clone_max_text_tokens", 75)
            )
            do_sample = bool(msg.get("do_sample", True))

            # Sampling / decoding params
            text_temperature = float(msg.get("text_temperature", 1.0))
            text_top_p = float(msg.get("text_top_p", 1.0))
            text_top_k = int(msg.get("text_top_k", 50))
            audio_temperature = float(msg.get("audio_temperature", 0.8))
            audio_top_p = float(msg.get("audio_top_p", 0.95))
            audio_top_k = int(msg.get("audio_top_k", 25))
            audio_repetition_penalty = float(
                msg.get("audio_repetition_penalty", 1.2)
            )

            # Announce stream start
            if not _send({"type": "stream_start", "sample_rate": self.sample_rate}):
                return

            idx = 0
            seed = msg.get("seed")
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

            for event in self.model.inference_stream(
                text=text,
                output_audio_path=None,
                mode=mode,
                prompt_text=prompt_text,
                prompt_audio_path=prompt_audio_path,
                text_tokenizer_path=self.model.name_or_path,
                audio_tokenizer_type="moss-tts-nano",
                audio_tokenizer_pretrained_name_or_path="OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano",
                device=self.device,
                max_new_frames=max_new_frames,
                voice_clone_max_text_tokens=voice_clone_max_text_tokens,
                do_sample=do_sample,
                use_kv_cache=True,
                text_temperature=text_temperature,
                text_top_p=text_top_p,
                text_top_k=text_top_k,
                audio_temperature=audio_temperature,
                audio_top_p=audio_top_p,
                audio_top_k=audio_top_k,
                audio_repetition_penalty=audio_repetition_penalty,
            ):
                if _shutting_down:
                    return

                event_type = event.get("type", "")
                if event_type == "audio":
                    waveform = event["waveform"]
                    # Normalize to numpy float32
                    if torch.is_tensor(waveform):
                        chunk_np = waveform.detach().cpu().numpy().astype(np.float32)
                    else:
                        chunk_np = np.asarray(waveform, dtype=np.float32)

                    chunk_bytes = chunk_np.tobytes()

                    if not _send(
                        {
                            "type": "stream_chunk",
                            "chunk": chunk_bytes,
                            "index": idx,
                            "length": len(chunk_np),
                        }
                    ):
                        return  # Elixir side disconnected
                    idx += 1

            _send(
                {
                    "type": "stream_end",
                    "total_chunks": idx,
                }
            )

        except Exception as e:
            import traceback

            traceback.print_exc(file=sys.stderr)
            _send({"type": "stream_error", "error": str(e)})


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    bridge = MOSSNanoBridge()

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

            elif msg_type == "list_voices":
                _send(bridge.list_voices())

            elif msg_type == "generate":
                _send(bridge.generate(msg))

            elif msg_type == "generate_streaming":
                bridge.generate_streaming(msg)

            elif msg_type == "ping":
                _send({"status": "ok", "message": "pong"})

            else:
                _send_error(f"Unknown request type: {msg_type}")

        except Exception as e:
            _send_error(f"Unhandled error: {e}")


if __name__ == "__main__":
    main()
