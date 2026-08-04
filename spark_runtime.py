"""Single place holding the DGX Spark / ARM64 runtime setup.

See README-spark.md for how the numbers below were arrived at. Anything that
loads the model outside `app.py` (headless bulk generators, scripts) should go
through `create_engine()` so it gets the same configuration the Gradio app gets
from the Dockerfile CMD, instead of the library defaults — those either hang
CosyVoice or blow up the KV cache on this box.
"""

import io
import logging

import numpy as np
import soundfile as sf
import torch
import torchaudio

from model_loader import ModelSource
from tokenizer import StepAudioTokenizer
from tts import StepAudioTTS

logger = logging.getLogger(__name__)

MODEL_PATH = "/models/Step-Audio-EditX"
TOKENIZER_PATH = "/models/Step-Audio-Tokenizer"

# Verified on DGX Spark, 2026-08-04. Keep in sync with the Dockerfile.spark CMD.
ENGINE_KWARGS = dict(
    model_source=ModelSource.LOCAL,
    quantization=None,
    tensor_parallel_size=1,
    # a single-user box: the vLLM default reserves a KV cache for hundreds of
    # concurrent requests we will never issue
    gpu_memory_utilization=0.20,
    max_model_len=3072,
    max_num_seqs=1,
    # no Inductor, no vLLM CUDA graphs — both misbehave here
    enforce_eager=True,
    dtype="bfloat16",
    # CosyVoice in bfloat16 sits at 100% CPU forever instead of loading, and
    # with CUDA graphs on it dies with "GET was unable to find an engine"
    cosyvoice_dtype="float32",
    cosyvoice_cuda_graph=False,
)


def _sf_load(uri, frame_offset=0, num_frames=-1, normalize=True, channels_first=True, **_):
    """`torchaudio.load` on libsndfile, returning (Tensor[C, T], sample_rate)."""
    stop = None if num_frames in (-1, None) else frame_offset + num_frames
    data, sample_rate = sf.read(
        uri,
        start=frame_offset,
        stop=stop,
        dtype="float32" if normalize else "int16",
        always_2d=True,
    )
    wav = torch.from_numpy(data)                      # sf gives [T, C]
    return (wav.t() if channels_first else wav), sample_rate


def _sf_save(uri, src, sample_rate, channels_first=True, **_):
    """`torchaudio.save` on libsndfile. Int tensors stay PCM, floats go to 16-bit."""
    wav = src.detach().cpu()
    if wav.dtype not in (torch.int16, torch.int32):
        wav = wav.float()
    data = wav.numpy()
    if data.ndim == 1:
        data = data[:, np.newaxis]
    elif channels_first:
        data = data.T                                 # sf wants [T, C]
    sf.write(uri, data, sample_rate)


def install_audio_compat() -> None:
    """Route `torchaudio` file I/O through libsndfile.

    torchaudio 2.11 (the one in nvcr.io/nvidia/vllm:26.04-py3) forwards
    load()/save() to TorchCodec, and the aarch64 torchcodec wheel double-frees
    its decoder against this container's ffmpeg: the process dies with
    `free(): invalid pointer`. That is a native abort, not an exception, so no
    caller can defend against it — and every clone/edit reads its prompt wav
    through torchaudio.load (`tts.preprocess_prompt_wav`, and funasr's
    `load_audio_text_image_video` for the vq02 codes), so the model is unusable
    until this is in place. libsndfile handles the wav/flac we feed it, and the
    resample/transform half of torchaudio is untouched by this.
    """
    if getattr(torchaudio, "_spark_audio_compat", False):
        return
    torchaudio.load = _sf_load
    torchaudio.save = _sf_save
    torchaudio._spark_audio_compat = True
    logger.info("🔧 torchaudio load/save routed through soundfile (Spark)")


def create_engine(
    model_path: str = MODEL_PATH,
    tokenizer_path: str = TOKENIZER_PATH,
    **overrides,
) -> StepAudioTTS:
    """Load the tokenizer + TTS engine with the Spark settings applied."""
    install_audio_compat()
    kwargs = {**ENGINE_KWARGS, **overrides}
    tokenizer = StepAudioTokenizer(tokenizer_path, model_source=kwargs["model_source"])
    return StepAudioTTS(model_path, tokenizer, **kwargs)
