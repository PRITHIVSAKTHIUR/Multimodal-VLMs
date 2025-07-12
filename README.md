# **Multimodal VLMs - OCR & VQA Interface**

> A comprehensive Gradio-based interface for running multiple state-of-the-art Vision-Language Models (VLMs) for Optical Character Recognition (OCR) and Visual Question Answering (VQA) tasks. This application supports both image and video inference across 5 different pre-trained models.

## Features

- **Multi-Model Support**: Integration of 5 different VLMs with specialized capabilities
- **Dual Input Modes**: Support for both image and video input processing
- **Real-time Streaming**: Live text generation with streaming output
- **Advanced Configuration**: Customizable generation parameters (temperature, top-p, top-k, etc.)
- **User-friendly Interface**: Clean Gradio web interface with example datasets
- **GPU Acceleration**: Optimized for CUDA-enabled environments

## Supported Models

### 1. Vision-Matters-7B
- **Source**: Yuting6/Vision-Matters-7B
- **Specialty**: Enhanced visual perturbation framework for better reasoning
- **Architecture**: Based on Qwen2.5-VL with improved visual comprehension

### 2. WR30a-Deep-7B-0711
- **Source**: prithivMLmods/WR30a-Deep-7B-0711
- **Specialty**: Image captioning, visual analysis, and image reasoning
- **Training**: Fine-tuned on 1,500k image pairs for superior understanding

### 3. ViGaL-7B
- **Source**: yunfeixie/ViGaL-7B
- **Specialty**: Game-trained model with transferable reasoning skills
- **Performance**: Enhanced performance on MathVista and MMMU benchmarks

### 4. MonkeyOCR-pro-1.2B
- **Source**: echo840/MonkeyOCR-pro-1.2B
- **Specialty**: Document OCR with Structure-Recognition-Relation (SRR) paradigm
- **Focus**: Efficient full-page document processing

### 5. Visionary-R1-3B
- **Source**: maifoundations/Visionary-R1
- **Specialty**: Reinforcement learning-based visual reasoning
- **Approach**: Scalable training using only visual question-answer pairs

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM
- 50GB+ storage for models

### Dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers
pip install gradio
pip install spaces
pip install opencv-python
pip install pillow
pip install numpy
```

### Clone Repository
```bash
git clone https://github.com/PRITHIVSAKTHIUR/Multimodal-VLMs.git
cd Multimodal-VLMs
```

## Usage

### Basic Setup
```bash
python app.py
```

### Environment Variables
- `MAX_INPUT_TOKEN_LENGTH`: Maximum input token length (default: 4096)

### Interface Access
Once launched, access the interface through your browser at the provided local URL.

## API Reference

### Image Inference
```python
generate_image(
    model_name: str,           # Selected model identifier
    text: str,                 # Query text
    image: Image.Image,        # PIL Image object
    max_new_tokens: int = 1024,
    temperature: float = 0.6,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.2
)
```

### Video Inference
```python
generate_video(
    model_name: str,           # Selected model identifier
    text: str,                 # Query text
    video_path: str,           # Path to video file
    max_new_tokens: int = 1024,
    temperature: float = 0.6,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.2
)
```

## Configuration Parameters

### Generation Settings
- **max_new_tokens**: Maximum number of tokens to generate (1-2048)
- **temperature**: Randomness in generation (0.1-4.0)
- **top_p**: Nucleus sampling threshold (0.05-1.0)
- **top_k**: Top-k sampling parameter (1-1000)
- **repetition_penalty**: Penalty for repeated tokens (1.0-2.0)

### Video Processing
- **Frame Sampling**: Extracts 10 evenly spaced frames from input video
- **Format Support**: Standard video formats (MP4, AVI, MOV)
- **Resolution**: Automatic frame preprocessing and RGB conversion

## Examples

### Image Tasks
- Mathematical problem solving
- Document content extraction
- Scene explanation and analysis
- Expression simplification
- Variable solving

### Video Tasks
- Detailed video content analysis
- Action recognition and description
- Sequential frame understanding

## Performance Optimization

### GPU Memory Management
- Models loaded with float16 precision
- Automatic device detection (CUDA/CPU)
- Efficient memory utilization across multiple models

### Processing Optimization
- Streaming text generation for real-time feedback
- Threaded model inference to prevent UI blocking
- Optimized tokenization and preprocessing

## Limitations

- Video inference performance varies across models
- GPU memory requirements scale with model size
- Processing time depends on input complexity and generation length
- Some models may not perform optimally for all task types

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is open-source and available under the MIT License.

## Acknowledgments

- Hugging Face Transformers library
- Gradio framework for web interface
- Model developers and research teams
- Open-source computer vision community

## Support

For issues and questions:
- Create an issue in the GitHub repository
- Check the Hugging Face Space discussions
- Review model documentation for specific capabilities
