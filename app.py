import os
import random
import uuid
import json
import time
import asyncio
from threading import Thread

import gradio as gr
import spaces
import torch
import numpy as np
from PIL import Image
import cv2

from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    TextIteratorStreamer,
)
from transformers.image_utils import load_image

# Constants for text generation
MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load Vision-Matters-7B
MODEL_ID_M = "Yuting6/Vision-Matters-7B"
processor_m = AutoProcessor.from_pretrained(MODEL_ID_M, trust_remote_code=True)
model_m = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_M, trust_remote_code=True,
    torch_dtype=torch.float16).to(device).eval()

# Load ViGaL-7B
MODEL_ID_X = "yunfeixie/ViGaL-7B"
processor_x = AutoProcessor.from_pretrained(MODEL_ID_X, trust_remote_code=True)
model_x = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_X, trust_remote_code=True,
    torch_dtype=torch.float16).to(device).eval()

# Load prithivMLmods/WR30a-Deep-7B-0711
MODEL_ID_T = "prithivMLmods/WR30a-Deep-7B-0711"
processor_t = AutoProcessor.from_pretrained(MODEL_ID_T, trust_remote_code=True)
model_t = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_T, trust_remote_code=True,
    torch_dtype=torch.float16).to(device).eval()

# Load Visionary-R1
MODEL_ID_O = "maifoundations/Visionary-R1"
processor_o = AutoProcessor.from_pretrained(MODEL_ID_O, trust_remote_code=True)
model_o = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_O, trust_remote_code=True,
    torch_dtype=torch.float16).to(device).eval()

#-----------------------------subfolder-----------------------------#
# Load MonkeyOCR-pro-1.2B
MODEL_ID_W = "echo840/MonkeyOCR-pro-1.2B"
SUBFOLDER = "Recognition"
processor_w = AutoProcessor.from_pretrained(MODEL_ID_W, trust_remote_code=True, subfolder=SUBFOLDER)
model_w = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID_W, trust_remote_code=True,
    subfolder=SUBFOLDER,
    torch_dtype=torch.float16).to(device).eval()
#-----------------------------subfolder-----------------------------#

# Function to downsample video frames
def downsample_video(video_path):
    """
    Downsamples the video to evenly spaced frames.
    Each frame is returned as a PIL image along with its timestamp.
    """
    vidcap = cv2.VideoCapture(video_path)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frames = []
    frame_indices = np.linspace(0, total_frames - 1, 10, dtype=int)
    for i in frame_indices:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, image = vidcap.read()
        if success:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
            timestamp = round(i / fps, 2)
            frames.append((pil_image, timestamp))
    vidcap.release()
    return frames

# Function to generate text responses based on image input
@spaces.GPU
def generate_image(model_name: str,
                   text: str,
                   image: Image.Image,
                   max_new_tokens: int = 1024,
                   temperature: float = 0.6,
                   top_p: float = 0.9,
                   top_k: int = 50,
                   repetition_penalty: float = 1.2):
    """
    Generates responses using the selected model for image input.
    """
    if model_name == "Vision-Matters-7B":
        processor = processor_m
        model = model_m
    elif model_name == "ViGaL-7B":
        processor = processor_x
        model = model_x
    elif model_name == "Visionary-R1-3B":
        processor = processor_o
        model = model_o
    elif model_name == "WR30a-Deep-7B-0711":
        processor = processor_t
        model = model_t
    elif model_name == "MonkeyOCR-pro-1.2B":
        processor = processor_w
        model = model_w
    else:
        yield "Invalid model selected.", "Invalid model selected."
        return

    if image is None:
        yield "Please upload an image.", "Please upload an image."
        return

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": text},
        ]
    }]
    prompt_full = processor.apply_chat_template(messages,
                                                tokenize=False,
                                                add_generation_prompt=True)
    inputs = processor(text=[prompt_full],
                       images=[image],
                       return_tensors="pt",
                       padding=True,
                       truncation=False,
                       max_length=MAX_INPUT_TOKEN_LENGTH).to(device)
    streamer = TextIteratorStreamer(processor,
                                    skip_prompt=True,
                                    skip_special_tokens=True)
    generation_kwargs = {
        **inputs, "streamer": streamer,
        "max_new_tokens": max_new_tokens
    }
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    buffer = ""
    for new_text in streamer:
        buffer += new_text
        time.sleep(0.01)
        yield buffer, buffer

# Function to generate text responses based on video input
@spaces.GPU
def generate_video(model_name: str,
                   text: str,
                   video_path: str,
                   max_new_tokens: int = 1024,
                   temperature: float = 0.6,
                   top_p: float = 0.9,
                   top_k: int = 50,
                   repetition_penalty: float = 1.2):
    """
    Generates responses using the selected model for video input.
    """
    if model_name == "Vision-Matters-7B":
        processor = processor_m
        model = model_m
    elif model_name == "ViGaL-7B":
        processor = processor_x
        model = model_x
    elif model_name == "Visionary-R1-3B":
        processor = processor_o
        model = model_o
    elif model_name == "WR30a-Deep-7B-0711":
        processor = processor_t
        model = model_t
    elif model_name == "MonkeyOCR-pro-1.2B":
        processor = processor_w
        model = model_w
    else:
        yield "Invalid model selected.", "Invalid model selected."
        return

    if video_path is None:
        yield "Please upload a video.", "Please upload a video."
        return

    frames = downsample_video(video_path)
    messages = [{
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant."}]
    }, {
        "role": "user",
        "content": [{"type": "text", "text": text}]
    }]
    for frame in frames:
        image, timestamp = frame
        messages[1]["content"].append({"type": "text", "text": f"Frame {timestamp}:"})
        messages[1]["content"].append({"type": "image", "image": image})
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        truncation=False,
        max_length=MAX_INPUT_TOKEN_LENGTH).to(device)
    streamer = TextIteratorStreamer(processor,
                                    skip_prompt=True,
                                    skip_special_tokens=True)
    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
    }
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    buffer = ""
    for new_text in streamer:
        buffer += new_text
        buffer = buffer.replace("<|im_end|>", "")
        time.sleep(0.01)
        yield buffer, buffer

# Define examples for image and video inference
image_examples = [
    ["Extract the content.", "images/7.png"],
    ["Solve the problem to find the value.", "images/1.jpg"],
    ["Explain the scene.", "images/6.JPG"],
    ["Solve the problem step by step.", "images/2.jpg"],
    ["Find the value of 'X'.", "images/3.jpg"],
    ["Simplify the expression.", "images/4.jpg"],
    ["Solve for the value.", "images/5.png"]
]

video_examples = [
    ["Explain the video in detail.", "videos/1.mp4"],
    ["Explain the video in detail.", "videos/2.mp4"]
]

# Updated CSS with the new submit button theme
css = """
.submit-btn {
  --clr-font-main: hsla(0 0% 20% / 100);
  --btn-bg-1: hsla(194 100% 69% / 1);
  --btn-bg-2: hsla(217 100% 56% / 1);
  --btn-bg-color: hsla(360 100% 100% / 1);
  --radii: 0.5em;
  cursor: pointer;
  padding: 0.9em 1.4em;
  min-width: 120px;
  min-height: 44px;
  font-size: var(--size, 1rem);
  font-weight: 500;
  transition: 0.8s;
  background-size: 280% auto;
  background-image: linear-gradient(
    325deg,
    var(--btn-bg-2) 0%,
    var(--btn-bg-1) 55%,
    var(--btn-bg-2) 90%
  );
  border: none;
  border-radius: var(--radii);
  color: var(--btn-bg-color);
  box-shadow:
    0px 0px 20px rgba(71, 184, 255, 0.5),
    0px 5px 5px -1px rgba(58, 125, 233, 0.25),
    inset 4px 4px 8px rgba(175, 230, 255, 0.5),
    inset -4px -4px 8px rgba(19, 95, 216, 0.35);
}
.submit-btn:hover {
  background-position: right top;
}
.submit-btn:is(:focus, :focus-visible, :active) {
  outline: none;
  box-shadow:
    0 0 0 3px var(--btn-bg-color),
    0 0 0 6px var(--btn-bg-2);
}
@media (prefers-reduced-motion: reduce) {
  .submit-btn {
    transition: linear;
  }
}
.canvas-output {
    border: 2px solid #4682B4;
    border-radius: 10px;
    padding: 20px;
}
"""

# Create the Gradio Interface
with gr.Blocks(css=css, theme="bethecloud/storj_theme") as demo:
    gr.Markdown(
        "# **[Multimodal VLMs [OCR | VQA]](https://huggingface.co/collections/prithivMLmods/multimodal-implementations-67c9982ea04b39f0608badb0)**"
    )
    with gr.Row():
        with gr.Column():
            with gr.Tabs():
                with gr.TabItem("Image Inference"):
                    image_query = gr.Textbox(
                        label="Query Input",
                        placeholder="Enter your query here...")
                    image_upload = gr.Image(type="pil", label="Image")
                    image_submit = gr.Button("Submit",
                                             elem_classes="submit-btn")
                    gr.Examples(examples=image_examples,
                                inputs=[image_query, image_upload])
                with gr.TabItem("Video Inference"):
                    video_query = gr.Textbox(
                        label="Query Input",
                        placeholder="Enter your query here...")
                    video_upload = gr.Video(label="Video")
                    video_submit = gr.Button("Submit",
                                             elem_classes="submit-btn")
                    gr.Examples(examples=video_examples,
                                inputs=[video_query, video_upload])

            with gr.Accordion("Advanced options", open=False):
                max_new_tokens = gr.Slider(label="Max new tokens",
                                           minimum=1,
                                           maximum=MAX_MAX_NEW_TOKENS,
                                           step=1,
                                           value=DEFAULT_MAX_NEW_TOKENS)
                temperature = gr.Slider(label="Temperature",
                                        minimum=0.1,
                                        maximum=4.0,
                                        step=0.1,
                                        value=0.6)
                top_p = gr.Slider(label="Top-p (nucleus sampling)",
                                  minimum=0.05,
                                  maximum=1.0,
                                  step=0.05,
                                  value=0.9)
                top_k = gr.Slider(label="Top-k",
                                  minimum=1,
                                  maximum=1000,
                                  step=1,
                                  value=50)
                repetition_penalty = gr.Slider(label="Repetition penalty",
                                               minimum=1.0,
                                               maximum=2.0,
                                               step=0.05,
                                               value=1.2)

        with gr.Column():
            with gr.Column(elem_classes="canvas-output"):
                gr.Markdown("## Output")
                output = gr.Textbox(label="Raw Output Stream",
                                    interactive=False,
                                    lines=2, show_copy_button=True)
                with gr.Accordion("(Result.md)", open=False):
                    markdown_output = gr.Markdown(
                        label="markup.md")
                #download_btn = gr.Button("Download Result.md")

            model_choice = gr.Radio(choices=[
                 "Vision-Matters-7B", "WR30a-Deep-7B-0711",
                 "ViGaL-7B", "MonkeyOCR-pro-1.2B", "Visionary-R1-3B"
            ],
                                    label="Select Model",
                                    value="Vision-Matters-7B")

            gr.Markdown("**Model Info 💻** | [Report Bug](https://huggingface.co/spaces/prithivMLmods/Multimodal-VLMs-5x/discussions)")         
            gr.Markdown("> [WR30a-Deep-7B-0711](https://huggingface.co/prithivMLmods/WR30a-Deep-7B-0711): wr30a-deep-7b-0711 model is a fine-tuned version of qwen2.5-vl-7b-instruct, optimized for image captioning, visual analysis, and image reasoning. Built on top of the qwen2.5-vl architecture, this experimental model enhances visual comprehension capabilities with focused training on 1,500k image pairs for superior image understanding.")
            gr.Markdown("> [MonkeyOCR-pro-1.2B](https://huggingface.co/echo840/MonkeyOCR-pro-1.2B): MonkeyOCR adopts a structure-recognition-relation (SRR) triplet paradigm, which simplifies the multi-tool pipeline of modular approaches while avoiding the inefficiency of using large multimodal models for full-page document processing.")
            gr.Markdown("> [Vision Matters 7B](https://huggingface.co/Yuting6/Vision-Matters-7B): vision-matters is a simple visual perturbation framework that can be easily integrated into existing post-training pipelines including sft, dpo, and grpo. our findings highlight the critical role of visual perturbation: better reasoning begins with better seeing.")
            gr.Markdown("> [ViGaL 7B](https://huggingface.co/yunfeixie/ViGaL-7B): vigal-7b shows that training a 7b mllm on simple games like snake using reinforcement learning boosts performance on benchmarks like mathvista and mmmu without needing worked solutions or diagrams indicating transferable reasoning skills.")
            gr.Markdown("> [Visionary-R1](https://huggingface.co/maifoundations/Visionary-R1): visionary-r1 is a novel framework for training visual language models (vlms) to perform robust visual reasoning using reinforcement learning (rl). unlike traditional approaches that rely heavily on (sft) or (cot) annotations, visionary-r1 leverages only visual question-answer pairs and rl, making the process more scalable and accessible.")
            gr.Markdown(">⚠️note: all the models in space are not guaranteed to perform well in video inference use cases.")  

    # Define the submit button actions
    image_submit.click(fn=generate_image,
                       inputs=[
                           model_choice, image_query, image_upload,
                           max_new_tokens, temperature, top_p, top_k,
                           repetition_penalty
                       ],
                       outputs=[output, markdown_output])
    video_submit.click(fn=generate_video,
                       inputs=[
                           model_choice, video_query, video_upload,
                           max_new_tokens, temperature, top_p, top_k,
                           repetition_penalty
                       ],
                       outputs=[output, markdown_output])

if __name__ == "__main__":
    demo.queue(max_size=30).launch(share=True, mcp_server=True, ssr_mode=False, show_error=True)
