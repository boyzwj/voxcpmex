#!/usr/bin/env python3
"""
VoxCPMEx Bridge — Python bridge for Elixir VoxCPMEx library.

Communicates with the Elixir GenServer via stdin/stdout JSON-line protocol,
loading and running VoxCPM2 text-to-speech models.
"""

import sys
import json
import base64
import io
import os

# Attempt to import PyTorch early to catch missing deps
try:
    import torch
except ImportError as e:
    print(json.dumps({"status": "error", "error": f"Missing dependency: {e}"}))
    sys.exit(1)

import numpy as np
import soundfile as sf

# ---------------------------------------------------------------------------
# Monkey-patch torch.load for device mapping compatibility
# ---------------------------------------------------------------------------
_original_torch_load = torch.load


def _patched_torch_load(f, map_location=None, **kwargs):
    if map_location is None:
        map_location = "cpu"
    return _original_torch_load(f, map_location=map_location, **kwargs)


torch.load = _patched_torch_load


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------
def _detect_device(requested_device: str) -> str:
    """Return the best available device, falling back to CPU if unavailable."""
    requested = (requested_device or "cuda").strip().lower()

    if requested.startswith("cuda"):
        if torch.cuda.is_available():
            return requested
        return "cpu"

    if requested == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    return "cpu"


# ---------------------------------------------------------------------------
# VoxCPM Bridge
# ---------------------------------------------------------------------------
class VoxCPMBridge:
    """Bridge between Elixir and VoxCPM2 TTS models."""

    def __init__(self):
        self.model = None
        self.device = None
        self.sample_rate = None

    def init_model(self, hf_model_id: str = "openbmb/VoxCPM2", device: str = "cuda",
                   load_denoiser: bool = False, optimize: bool = True) -> dict:
        """Initialize the VoxCPM model from HuggingFace Hub."""
        try:
            from voxcpm import VoxCPM

            actual_device = _detect_device(device)
            self.device = actual_device
            print(f"Loading VoxCPM model: {hf_model_id} on {actual_device}...", file=sys.stderr, flush=True)

            self.model = VoxCPM.from_pretrained(
                hf_model_id,
                load_denoiser=load_denoiser,
                device=actual_device,
                optimize=optimize,
            )

            self.sample_rate = self.model.tts_model.sample_rate
            print(f"VoxCPM model loaded. sample_rate={self.sample_rate} device={self.device}", file=sys.stderr, flush=True)

            return {"status": "ok", "device": actual_device, "sample_rate": self.sample_rate}

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def generate(self, text: str, **kwargs) -> dict:
        """Generate speech from text."""
        if self.model is None:
            return {"status": "error", "error": "Model not initialized"}

        try:
            # Build generation kwargs
            gen_kwargs = {}

            # Core generation params
            gen_kwargs["cfg_value"] = kwargs.get("cfg_value", 2.0)
            gen_kwargs["inference_timesteps"] = kwargs.get("inference_timesteps", 10)
            gen_kwargs["min_len"] = kwargs.get("min_len", 2)
            gen_kwargs["max_len"] = kwargs.get("max_len", 4096)
            gen_kwargs["normalize"] = kwargs.get("normalize", False)
            gen_kwargs["denoise"] = kwargs.get("denoise", False)

            # Voice cloning: reference audio
            audio_prompt = kwargs.get("audio_prompt")
            if audio_prompt and os.path.exists(audio_prompt):
                gen_kwargs["reference_wav_path"] = audio_prompt

            # Ultimate cloning: prompt audio + transcript
            prompt_wav = kwargs.get("prompt_wav_path")
            prompt_text = kwargs.get("prompt_text")
            if prompt_wav and prompt_text:
                gen_kwargs["prompt_wav_path"] = prompt_wav
                gen_kwargs["prompt_text"] = prompt_text

            # Bad case retry
            gen_kwargs["retry_badcase"] = kwargs.get("retry_badcase", True)
            gen_kwargs["retry_badcase_max_times"] = kwargs.get("retry_badcase_max_times", 3)
            gen_kwargs["retry_badcase_ratio_threshold"] = kwargs.get("retry_badcase_ratio_threshold", 6.0)

            # Generate
            wav = self.model.generate(text, **gen_kwargs)

            # Convert numpy array to WAV bytes
            audio_bytes = self._wav_to_bytes(wav, self.sample_rate)

            # Base64 encode for JSON transport
            audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

            return {
                "status": "ok",
                "audio": audio_base64,
                "sample_rate": self.sample_rate,
                "duration": len(wav) / self.sample_rate,
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def generate_streaming(self, text: str, **kwargs) -> dict:
        """Generate speech with streaming (returns concatenated result with metadata)."""
        if self.model is None:
            return {"status": "error", "error": "Model not initialized"}

        try:
            gen_kwargs = {}
            gen_kwargs["cfg_value"] = kwargs.get("cfg_value", 2.0)
            gen_kwargs["inference_timesteps"] = kwargs.get("inference_timesteps", 10)
            gen_kwargs["min_len"] = kwargs.get("min_len", 2)
            gen_kwargs["max_len"] = kwargs.get("max_len", 4096)
            gen_kwargs["normalize"] = kwargs.get("normalize", False)
            gen_kwargs["denoise"] = kwargs.get("denoise", False)

            audio_prompt = kwargs.get("audio_prompt")
            if audio_prompt and os.path.exists(audio_prompt):
                gen_kwargs["reference_wav_path"] = audio_prompt

            prompt_wav = kwargs.get("prompt_wav_path")
            prompt_text = kwargs.get("prompt_text")
            if prompt_wav and prompt_text:
                gen_kwargs["prompt_wav_path"] = prompt_wav
                gen_kwargs["prompt_text"] = prompt_text

            gen_kwargs["retry_badcase"] = kwargs.get("retry_badcase", False)

            # Collect all streaming chunks
            chunks = []
            for chunk in self.model.generate_streaming(text, **gen_kwargs):
                chunks.append(chunk)

            wav = np.concatenate(chunks)
            audio_bytes = self._wav_to_bytes(wav, self.sample_rate)
            audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

            return {
                "status": "ok",
                "audio": audio_base64,
                "sample_rate": self.sample_rate,
                "duration": len(wav) / self.sample_rate,
                "num_chunks": len(chunks),
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _wav_to_bytes(self, wav: np.ndarray, sample_rate: int) -> bytes:
        """Convert numpy array to WAV bytes."""
        buffer = io.BytesIO()
        # Ensure float32 mono
        if wav.dtype != np.float32:
            wav = wav.astype(np.float32)
        sf.write(buffer, wav, sample_rate, format="WAV")
        buffer.seek(0)
        return buffer.read()

    def load_lora(self, lora_weights_path: str) -> dict:
        """Load LoRA weights from a checkpoint file."""
        if self.model is None:
            return {"status": "error", "error": "Model not initialized"}
        try:
            loaded, skipped = self.model.load_lora(lora_weights_path)
            return {"status": "ok", "loaded": len(loaded), "skipped": len(skipped)}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def unload_lora(self) -> dict:
        """Reset all LoRA weights to zero."""
        if self.model is None:
            return {"status": "error", "error": "Model not initialized"}
        try:
            self.model.unload_lora()
            return {"status": "ok"}
        except Exception as e:
            return {"status": "error", "error": str(e)}


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    """Read JSON commands from stdin, write responses to stdout."""
    bridge = VoxCPMBridge()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError as e:
            response = {"status": "error", "error": f"Invalid JSON: {e}"}
            print(json.dumps(response), flush=True)
            continue

        request_type = request.get("type")

        if request_type == "init":
            response = bridge.init_model(
                hf_model_id=request.get("model", "openbmb/VoxCPM2"),
                device=request.get("device", "cuda"),
                load_denoiser=request.get("load_denoiser", False),
                optimize=request.get("optimize", True),
            )

        elif request_type == "generate":
            text = request.get("text", "")
            kwargs = {k: v for k, v in request.items() if k not in ("type", "text")}
            response = bridge.generate(text, **kwargs)

        elif request_type == "generate_streaming":
            text = request.get("text", "")
            kwargs = {k: v for k, v in request.items() if k not in ("type", "text")}
            response = bridge.generate_streaming(text, **kwargs)

        elif request_type == "load_lora":
            response = bridge.load_lora(request.get("lora_path", ""))

        elif request_type == "unload_lora":
            response = bridge.unload_lora()

        elif request_type == "ping":
            response = {"status": "ok", "message": "pong"}

        else:
            response = {"status": "error", "error": f"Unknown request type: {request_type}"}

        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
