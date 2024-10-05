# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Updated to account for UI changes from https://github.com/rkfg/audiocraft/blob/long/app.py
# also released under the MIT license.

import argparse
from concurrent.futures import ProcessPoolExecutor
import logging
import os
from pathlib import Path
import subprocess as sp
import sys
from tempfile import NamedTemporaryFile
import time
import typing as tp
import warnings

from einops import rearrange
import torch
import gradio as gr

from audiocraft.data.audio_utils import convert_audio
from audiocraft.data.audio import audio_write
from audiocraft.models.encodec import InterleaveStereoCompressionModel
from audiocraft.models import AudioGen, MultiBandDiffusion

from google.cloud import translate
# GOOGLE_APPLICATION_CREDENTIALS にクレデンシャルをセットしておくこと
def translate_text_v3(text, target_language="en"):
    client = translate.TranslationServiceClient()

    # プロジェクト ID の設定
    project_id = "personal-test-330205"
    location = "global"

    parent = f"projects/{project_id}/locations/{location}"

    response = client.translate_text(
        contents=[text],
        target_language_code=target_language,
        mime_type="text/plain",
        parent=parent,
    )

    # 翻訳結果の取得
    for translation in response.translations:
        print(f"Original Text: {text}")
        print(f"Translated Text: {translation.translated_text}")
        return translation.translated_text


MODEL = None  # Last used model
SPACE_ID = os.environ.get('SPACE_ID', '')
IS_BATCHED = "facebook/AudioGen" in SPACE_ID or 'musicgen-internal/audiogen_dev' in SPACE_ID
print("IS_BATCHED", IS_BATCHED)
MAX_BATCH_SIZE = 12
BATCHED_DURATION = 15
INTERRUPTING = False
MBD = None
# We have to wrap subprocess call to clean a bit the log when using gr.make_waveform
_old_call = sp.call


def _call_nostderr(*args, **kwargs):
    # Avoid ffmpeg vomiting on the logs.
    kwargs['stderr'] = sp.DEVNULL
    kwargs['stdout'] = sp.DEVNULL
    _old_call(*args, **kwargs)


sp.call = _call_nostderr
# Preallocating the pool of processes.
pool = ProcessPoolExecutor(4)
pool.__enter__()


def interrupt():
    global INTERRUPTING
    INTERRUPTING = True


class FileCleaner:
    def __init__(self, file_lifetime: float = 3600):
        self.file_lifetime = file_lifetime
        self.files = []

    def add(self, path: tp.Union[str, Path]):
        self._cleanup()
        self.files.append((time.time(), Path(path)))

    def _cleanup(self):
        now = time.time()
        for time_added, path in list(self.files):
            if now - time_added > self.file_lifetime:
                if path.exists():
                    path.unlink()
                self.files.pop(0)
            else:
                break
                
file_cleaner = FileCleaner()


def make_waveform(*args, **kwargs):
    # Further remove some warnings.
    be = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = gr.make_waveform(*args, **kwargs)
        print("Make a video took", time.time() - be)
        return out


def load_model(version='facebook/audiogen-medium', device='cuda'):
    global MODEL
    print("Loading model", version)
    if MODEL is None or MODEL.name != version:
        # Clear PyTorch CUDA cache and delete model
        del MODEL
        torch.cuda.empty_cache()
        MODEL = None  # in case loading would crash
        MODEL = AudioGen.get_pretrained(version, device)


def load_diffusion():
    global MBD
    if MBD is None:
        print("loading MBD")
        MBD = MultiBandDiffusion.get_mbd_musicgen()


def _do_predictions(texts, duration, progress=False, gradio_progress=None, **gen_kwargs):
    MODEL.set_generation_params(duration=duration, **gen_kwargs)
    print("new batch", len(texts), texts)
    be = time.time()
    try:
        outputs = MODEL.generate(texts, progress=progress)
    except RuntimeError as e:
        raise gr.Error("Error while generating " + e.args[0])   
    outputs = outputs.detach().cpu().float()
    out_videos = [None]
    # pending_videos = []
    out_wavs = []
    for output in outputs:
        with NamedTemporaryFile("wb", suffix=".wav", delete=False) as file:
            audio_write(
                file.name, output, MODEL.sample_rate, strategy="loudness",
                loudness_headroom_db=16, loudness_compressor=True, add_suffix=False)
            # pending_videos.append(pool.submit(make_waveform, file.name))
            out_wavs.append(file.name)
            file_cleaner.add(file.name)
    # out_videos = [pending_video.result() for pending_video in pending_videos]
    # for video in out_videos:
    #     file_cleaner.add(video)       
    print("batch finished", len(texts), time.time() - be)
    print("Tempfiles currently stored: ", len(file_cleaner.files))
    return out_videos, out_wavs

def predict_batched(texts):
    max_text_length = 512
    texts = [text[:max_text_length] for text in texts]
    load_model('facebook/audiogen-medium')
    res = _do_predictions(texts, BATCHED_DURATION)
    return res

def predict_full(model, model_path, text, duration, topk, topp, temperature, cfg_coef, progress=gr.Progress()):
    global INTERRUPTING
    INTERRUPTING = False
    progress(0, desc="Loading model...")
    model_path = model_path.strip()
    if model_path:
        if not Path(model_path).exists():
            raise gr.Error(f"Model path {model_path} doesn't exist.")
        if not Path(model_path).is_dir():
            raise gr.Error(f"Model path {model_path} must be a folder containing "
                           "state_dict.bin and compression_state_dict_.bin.")
        model = model_path
    if temperature < 0:
        raise gr.Error("Temperature must be >= 0.")
    if topk < 0:
        raise gr.Error("Topk must be non-negative.")
    if topp < 0:
        raise gr.Error("Topp must be non-negative.")

    topk = int(topk)
    load_model(model)

    max_generated = 0

    def _progress(generated, to_generate):
        nonlocal max_generated
        max_generated = max(generated, max_generated)
        progress((min(max_generated, to_generate), to_generate))
        if INTERRUPTING:
            raise gr.Error("Interrupted.")
    MODEL.set_custom_progress_callback(_progress)

    videos, wavs = _do_predictions(
        [text], duration, progress=True,
        top_k=topk, top_p=topp, temperature=temperature, cfg_coef=cfg_coef,
        gradio_progress=progress)
    return videos[0], wavs[0], None, None


def toggle_audio_src(choice):
    if choice == "mic":
        return gr.update(source="microphone", value=None, label="Microphone")
    else:
        return gr.update(source="upload", value=None, label="File")


def toggle_diffusion(choice):
    if choice == "MultiBand_Diffusion":
        return [gr.update(visible=True)] * 2
    else:
        return [gr.update(visible=False)] * 2


def do_translate(text, enable):
    if enable:
        return translate_text_v3(text)
    return text

def ui_full(launch_kwargs):
    with gr.Blocks() as interface:
        gr.Markdown(
            """
            # AudioGen
            This is your private demo for [AudioGen](https://github.com/facebookresearch/audiocraft),
            a simple and controllable model for audio generation
            presented at: ["Simple and Controllable Audio Generation"](https://huggingface.co/papers/2306.05284)
            """
        )
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        text = gr.Text(label="Input Text", interactive=True)
                        useTranslation = gr.Checkbox(label="Use Translation", value=True, interactive=True)
                with gr.Row():
                    duration = gr.Slider(minimum=1, maximum=30, value=3, label="Duration(sec)", interactive=True)
                with gr.Row():
                    submit = gr.Button("Generate")
                    _ = gr.Button("Interrupt").click(fn=interrupt, queue=False)
                with gr.Row():
                    with gr.Accordion(label="Advanced Options", open=False):                
                        with gr.Row():
                            model = gr.Radio(["facebook/audiogen-medium"],
                                            label="Model", value="facebook/audiogen-medium", interactive=True)
                            model_path = gr.Text(label="Model Path (custom models)")
                        with gr.Row():
                            topk = gr.Number(label="Top-k", value=250, interactive=True)
                            topp = gr.Number(label="Top-p", value=0, interactive=True)
                            temperature = gr.Number(label="Temperature", value=1.0, interactive=True)
                            cfg_coef = gr.Number(label="Classifier Free Guidance", value=3.0, interactive=True)
            with gr.Column():
                output = gr.Video(label="Generated Music", visible=False)
                audio_output = gr.Audio(label="Generated Music (wav)", type='filepath')

        submit.click(do_translate, 
                     inputs=[text, useTranslation], 
                     outputs=text,
                     queue=False).then(
                        predict_full, 
                        inputs=[model, model_path, text, duration, topk, topp, temperature, cfg_coef],
                        outputs=[output, audio_output],
                        concurrency_limit=1,
                        concurrency_id='gpu_queue', queue=True)

        interface.queue().launch(**launch_kwargs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--listen',
        type=str,
        default='0.0.0.0' if 'SPACE_ID' in os.environ else '127.0.0.1',
        help='IP to listen on for connections to Gradio',
    )
    parser.add_argument(
        '--username', type=str, default='', help='Username for authentication'
    )
    parser.add_argument(
        '--password', type=str, default='', help='Password for authentication'
    )
    parser.add_argument(
        '--server_port',
        type=int,
        default=0,
        help='Port to run the server listener on',
    )
    parser.add_argument(
        '--inbrowser', action='store_true', help='Open in browser'
    )
    parser.add_argument(
        '--share', action='store_true', help='Share the gradio UI'
    )

    args = parser.parse_args()

    launch_kwargs = {}
    launch_kwargs['server_name'] = args.listen

    if args.username and args.password:
        launch_kwargs['auth'] = (args.username, args.password)
    if args.server_port:
        launch_kwargs['server_port'] = args.server_port
    if args.inbrowser:
        launch_kwargs['inbrowser'] = args.inbrowser
    if args.share:
        launch_kwargs['share'] = args.share

    logging.basicConfig(level=logging.INFO, stream=sys.stderr)

    # Show the interface
    ui_full(launch_kwargs)
