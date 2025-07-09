# **Multimodal-VLMs**

A comprehensive Gradio-based application for multimodal vision-language model inference, supporting image, video, and PDF inputs across multiple state-of-the-art models.

## Features

- **Multi-Input Support**: Process images, videos, and single-page PDFs
- **Multiple Models**: Choose from 5 different vision-language models
- **Real-time Streaming**: Get responses as they're generated
- **Advanced Controls**: Fine-tune generation parameters
- **Export Functionality**: Download results as Markdown files

## Supported Models

| Model | Description |
|-------|-------------|
| Vision-Matters-7B-Math | Visual perturbation framework for enhanced mathematical reasoning |
| ViGaL-7B | Reinforcement learning trained model using simple games like Snake |
| Visionary-R1 | Novel RL framework for robust visual reasoning without CoT annotations |
| R1-Onevision-7B | Enhanced vision-language understanding and reasoning capabilities |
| VLM-R1-Qwen2.5VL-3B-Math-0305 | R1 methodology framework for improved VLM reasoning |

## Installation

### Requirements

```bash
pip install torch torchvision
pip install transformers
pip install gradio
pip install spaces
pip install pillow
pip install opencv-python
pip install pdf2image
pip install numpy
```

### Setup

1. Clone the repository:
```bash
git clone https://github.com/PRITHIVSAKTHIUR/Multimodal-VLMs.git
cd Multimodal-VLMs
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

## Usage

### Image Inference

1. Select the "Image Inference" tab
2. Enter your query in the text box
3. Upload an image file
4. Choose your preferred model
5. Adjust advanced parameters if needed
6. Click "Submit"

### Video Inference

1. Select the "Video Inference" tab
2. Enter your query describing what you want to analyze
3. Upload a video file
4. The system will automatically extract 10 evenly spaced frames
5. Choose your model and submit

### PDF Inference

1. Select the "Single Page PDF Inference" tab
2. Enter your query about the document
3. Upload a single-page PDF file
4. The system converts the PDF to an image for processing
5. Choose your model and submit

## Advanced Parameters

| Parameter | Description | Range | Default |
|-----------|-------------|-------|---------|
| Max New Tokens | Maximum number of tokens to generate | 1-2048 | 1024 |
| Temperature | Controls randomness in generation | 0.1-4.0 | 0.6 |
| Top-p | Nucleus sampling parameter | 0.05-1.0 | 0.9 |
| Top-k | Top-k sampling parameter | 1-1000 | 50 |
| Repetition Penalty | Penalty for repetitive text | 1.0-2.0 | 1.2 |

## System Requirements

- **GPU**: CUDA-compatible GPU recommended for optimal performance
- **RAM**: Minimum 16GB RAM
- **Storage**: At least 50GB free space for model weights
- **Python**: Version 3.8 or higher

## Model Loading

The application automatically loads all 5 models at startup:
- Models are loaded in float16 precision for memory efficiency
- Each model uses its respective processor and tokenizer
- Models are moved to GPU if available, otherwise CPU

## Video Processing

- Videos are downsampled to 10 evenly spaced frames
- Each frame is converted to PIL Image format
- Timestamps are preserved for context
- Supports common video formats (MP4, AVI, MOV)

## PDF Processing

- Only single-page PDFs are supported
- PDFs are converted to images using pdf2image
- First page is processed if multiple pages exist
- Supports standard PDF formats

## Error Handling

The application includes comprehensive error handling for:
- Invalid file formats
- Model loading failures
- GPU memory issues
- File conversion errors
- Network connectivity problems

## Performance Notes

- GPU acceleration is automatically used when available
- Models are kept in memory for faster inference
- Streaming output provides real-time feedback
- Video processing may take longer due to frame extraction

## Limitations

- Video inference performance may vary across models
- Single-page PDF processing only
- Maximum input token length: 4096 tokens
- GPU memory requirements vary by model

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face Transformers library
- Gradio framework
- Individual model creators and researchers
- Open source community contributions

## Support

For issues and questions:
- Open an issue on GitHub
- Check the discussions section
- Review the documentation
