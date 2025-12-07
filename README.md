# Amazon Bedrock Workshop

A comprehensive workshop demonstrating various capabilities of Amazon Bedrock, AWS's fully managed service for building and scaling generative AI applications. This project includes multiple hands-on labs covering text generation, image generation, embeddings, RAG (Retrieval-Augmented Generation), streaming, and more.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Available Labs](#available-labs)
- [Running the Labs](#running-the-labs)
- [Key Features](#key-features)
- [License](#license)

## Prerequisites

Before you begin, ensure you have the following:

1. **AWS Account** with appropriate permissions to access Amazon Bedrock
2. **Python 3.10+** installed
3. **AWS CLI** configured with appropriate credentials
4. **Access to Amazon Bedrock models** - You may need to request access to specific models in the AWS Bedrock console
5. **boto3** and **botocore** installed (included in requirements)

## Key Technologies

1. **Amazon Bedrock**: Fully managed service for foundation models
2. **Streamlit**: Interactive web applications
3. **ChromaDB**: Vector database for embeddings
4. **boto3**: AWS SDK for Python
5. **FAISS**: Vector similarity search

## Key Features

### Models Used

This workshop demonstrates various Amazon Bedrock foundation models:

- **Text Models**:
  - `us.amazon.nova-lite-v1:0` - Amazon Nova Lite
  - `us.anthropic.claude-3-7-sonnet-20250219-v1:0` - Claude 3.7 Sonnet
  - `stability.stable-diffusion-xl-v1` - Stable Diffusion XL

- **Embedding Models**:
  - `amazon.titan-embed-text-v2:0` - Text embeddings
  - `amazon.titan-embed-image-v1` - Multimodal embeddings

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd Amazon-Bedrock-Workshop
```

2. Install the required dependencies:
```bash
pip install -r setup/requirements.txt
```

The requirements include:
- `boto3` and `botocore` - AWS SDK for Python
- `streamlit` - Web framework for building interactive apps
- `chromadb` - Vector database for embeddings
- `faiss-cpu` - Vector similarity search
- `transformers` - Hugging Face transformers library
- `Pillow` - Image processing
- `pypdf` - PDF processing
- `anthropic` - Anthropic SDK (for Claude models)

3. Configure AWS credentials:
```bash
aws configure
```

Ensure your AWS credentials have permissions to:
- Access Amazon Bedrock runtime API
- Use various Bedrock foundation models

## Project Structure

```
Amazon-Bedrock-Workshop/
├── labs/                    # All workshop labs
│   ├── api/                 # Basic API examples
│   ├── chatbot/             # Chatbot applications
│   ├── image*/              # Image generation and manipulation labs
│   ├── embedding/           # Text embeddings
│   ├── embeddings_search/   # Vector search with embeddings
│   ├── rag/                 # Retrieval-Augmented Generation
│   ├── streaming/           # Streaming responses
│   ├── guardrails/          # Content filtering and guardrails
│   └── ...                  # Additional labs
├── setup/                   # Setup scripts and requirements
│   ├── requirements.txt     # Python dependencies
│   └── getproxyurl.py      # SageMaker proxy URL helper
└── README.md               # This file
```

## Available Labs

### Text Generation Labs

#### 1. **Basic API** (`labs/api/`)
- **File**: `bedrock_api.py`
- **Description**: Demonstrates basic usage of Amazon Bedrock API with the Nova Lite model
- **Run**: `python labs/api/bedrock_api.py`
- **Features**: Simple text generation using the messages API format

#### 2. **Text to Text** (`labs/text/`)
- **Files**: `text_app.py`, `text_lib.py`
- **Description**: Streamlit app for text generation
- **Run**: `streamlit run labs/text/text_app.py`
- **Features**: Interactive text generation interface

#### 3. **Chatbot** (`labs/chatbot/`)
- **Files**: `chatbot_app.py`, `chatbot_lib.py`
- **Description**: Interactive chatbot using Claude 3.7 Sonnet
- **Run**: `streamlit run labs/chatbot/chatbot_app.py`
- **Features**: 
  - Conversational interface with message history
  - Uses Bedrock Converse API
  - Maintains conversation context

#### 4. **Converse API** (`labs/converse/`)
- **File**: `converse_api.py`
- **Description**: Direct usage of Bedrock Converse API
- **Run**: `python labs/converse/converse_api.py`

#### 5. **Temperature Control** (`labs/temperature/`)
- **File**: `temperature.py`
- **Description**: Demonstrates how temperature affects model responses
- **Run**: `python labs/temperature/temperature.py`
- **Features**: Shows variations in responses with different temperature settings

#### 6. **Parameters** (`labs/params/`)
- **File**: `params.py`
- **Description**: Exploring inference parameters (temperature, topP, topK, etc.)
- **Run**: `python labs/params/params.py`

### Image Generation Labs

#### 7. **Image Generation** (`labs/image/`)
- **Files**: `image_app.py`, `image_lib.py`
- **Description**: Generate images from text prompts
- **Run**: `streamlit run labs/image/image_app.py`
- **Features**: Text-to-image generation

#### 8. **Image Variation** (`labs/image_variation/`)
- **Files**: `image_variation_app.py`, `image_variation_lib.py`
- **Description**: Generate variations of existing images
- **Run**: `streamlit run labs/image_variation/image_variation_app.py`
- **Features**: Create variations with similarity strength control

#### 9. **Image to Image** (`labs/image_to_image/`)
- **Files**: `image_to_image_app.py`, `image_to_image_lib.py`
- **Description**: Transform images using prompts
- **Run**: `streamlit run labs/image_to_image/image_to_image_app.py`

#### 10. **Image Style Mixing** (`labs/image_style_mixing/`)
- **Files**: `image_style_mixing_app.py`, `image_style_mixing_lib.py`
- **Description**: Mix styles from two images
- **Run**: `streamlit run labs/image_style_mixing/image_style_mixing_app.py`
- **Features**: 
  - Upload two images
  - Mix their styles with customizable similarity strength
  - Example images included

#### 11. **Image Understanding** (`labs/image_understanding/`)
- **Files**: `image_understanding_app.py`, `image_understanding_lib.py`
- **Description**: Analyze and understand image content
- **Run**: `streamlit run labs/image_understanding/image_understanding_app.py`
- **Features**: Vision capabilities for image analysis

#### 12. **Image Background** (`labs/image_background/`)
- **Files**: `image_background_app.py`, `image_background_lib.py`
- **Description**: Modify or extend image backgrounds
- **Run**: `streamlit run labs/image_background/image_background_app.py`

#### 13. **Image Extension** (`labs/image_extension/`)
- **Files**: `image_extension_app.py`, `image_extension_lib.py`
- **Description**: Extend images beyond their original boundaries
- **Run**: `streamlit run labs/image_extension/image_extension_app.py`

#### 14. **Image Replacement** (`labs/image_replacement/`)
- **Files**: `image_replacement_app.py`, `image_replacement_lib.py`
- **Description**: Replace parts of images
- **Run**: `streamlit run labs/image_replacement/image_replacement_app.py`

#### 15. **Image Insertion** (`labs/image_insertion/`)
- **Files**: `image_insertion_app.py`, `image_insertion_lib.py`
- **Description**: Insert objects into images
- **Run**: `streamlit run labs/image_insertion/image_insertion_app.py`

#### 16. **Image Masking** (`labs/image_masking/`)
- **Files**: `image_masking_app.py`, `image_masking_lib.py`
- **Description**: Apply masks to images for selective editing
- **Run**: `streamlit run labs/image_masking/image_masking_app.py`

#### 17. **Image Prompts** (`labs/image_prompts/`)
- **Files**: `image_prompts_app.py`, `image_prompts_lib.py`
- **Description**: Advanced prompt engineering for images
- **Run**: `streamlit run labs/image_prompts/image_prompts_app.py`

### Embeddings and Search Labs

#### 18. **Text Embeddings** (`labs/embedding/`)
- **File**: `bedrock_embedding.py`
- **Description**: Generate text embeddings using Titan Embed Text model
- **Run**: `python labs/embedding/bedrock_embedding.py`
- **Features**: 
  - Cosine similarity calculations
  - Compare text items from `items.txt`

#### 19. **Embeddings Search** (`labs/embeddings_search/`)
- **Files**: `embeddings_search_app.py`, `embeddings_search_lib.py`
- **Description**: Vector search using text embeddings
- **Run**: `streamlit run labs/embeddings_search/embeddings_search_app.py`
- **Features**: Semantic search using ChromaDB

#### 20. **Image Search** (`labs/image_search/`)
- **Files**: `image_search_app.py`, `image_search_lib.py`
- **Description**: Multimodal image search using embeddings
- **Run**: `streamlit run labs/image_search/image_search_app.py`
- **Features**: 
  - Search images by text or by similar images
  - Uses Titan Embed Image model
  - Vector search with ChromaDB

### RAG (Retrieval-Augmented Generation) Labs

#### 21. **RAG** (`labs/rag/`)
- **Files**: `rag_app.py`, `rag_lib.py`
- **Description**: Retrieval-Augmented Generation implementation
- **Run**: `streamlit run labs/rag/rag_app.py`
- **Features**: 
  - Vector search for context retrieval
  - Enhanced text generation with retrieved context

#### 22. **RAG Chatbot** (`labs/rag_chatbot/`)
- **Files**: `rag_chatbot_app.py`, `rag_chatbot_lib.py`
- **Description**: Chatbot with RAG capabilities
- **Run**: `streamlit run labs/rag_chatbot/rag_chatbot_app.py`
- **Features**: Conversational interface with context retrieval

### Advanced Labs

#### 23. **Streaming** (`labs/streaming/`)
- **Files**: `streaming_app.py`, `streaming_lib.py`
- **Description**: Stream responses in real-time
- **Run**: `streamlit run labs/streaming/streaming_app.py`
- **Features**: Real-time token streaming for better UX

#### 24. **Intro Streaming** (`labs/intro_streaming/`)
- **File**: `intro_streaming.py`
- **Description**: Introduction to streaming concepts
- **Run**: `python labs/intro_streaming/intro_streaming.py`

#### 25. **Guardrails** (`labs/guardrails/`)
- **Files**: `guardrails_app.py`, `guardrails_lib.py`, `create_guardrail.py`
- **Description**: Content filtering and safety guardrails
- **Run**: 
  - Create guardrail: `python labs/guardrails/create_guardrail.py`
  - Test guardrails: `streamlit run labs/guardrails/guardrails_app.py`
- **Features**: 
  - Content filtering
  - PII detection
  - Custom guardrail policies

#### 26. **Multimodal Chatbot** (`labs/multimodal_chatbot/`)
- **Files**: `multimodal_chatbot_app.py`, `multimodal_chatbot_lib.py`
- **Description**: Chatbot that can process both text and images
- **Run**: `streamlit run labs/multimodal_chatbot/multimodal_chatbot_app.py`
- **Features**: 
  - Upload images in chat
  - Process images with text prompts
  - Vision-language understanding

#### 27. **Document Summarization** (`labs/summarization/`)
- **Files**: `summarization_app.py`, `summarization_lib.py`
- **Description**: Summarize PDF documents
- **Run**: `streamlit run labs/summarization/summarization_app.py`
- **Features**: PDF processing and summarization

#### 28. **Model Comparison** (`labs/model_comparison/`)
- **Files**: `model_comparison_app.py`, `model_comparison_lib.py`
- **Description**: Compare different foundation models
- **Run**: `streamlit run labs/model_comparison/model_comparison_app.py`

#### 29. **Recommendations** (`labs/recommendations/`)
- **Files**: `recommendations_app.py`, `recommendations_lib.py`
- **Description**: Recommendation system using embeddings
- **Run**: `streamlit run labs/recommendations/recommendations_app.py`

#### 30. **Tool Use** (`labs/tool_use/`)
- **File**: `tool_use.py`
- **Description**: Function calling and tool use capabilities
- **Run**: `python labs/tool_use/tool_use.py`

#### 31. **JSON Mode** (`labs/json/`)
- **Files**: `json_app.py`, `json_lib.py`
- **Description**: Structured JSON output generation
- **Run**: `streamlit run labs/json/json_app.py`

#### 32. **CSV Processing** (`labs/csv/`)
- **Files**: `csv_app.py`, `csv_lib.py`
- **Description**: Process CSV files with AI
- **Run**: `streamlit run labs/csv/csv_app.py`

#### 33. **Showcase** (`labs/showcase/`)
- **Files**: `showcase_app.py`, `showcase_lib.py`, `showcase_examples.py`
- **Description**: Comprehensive showcase of capabilities
- **Run**: `streamlit run labs/showcase/showcase_app.py`

#### 34. **Text Playground** (`labs/text_playground/`)
- **Files**: `text_playground_app.py`, `text_playground_lib.py`
- **Description**: Interactive playground for text generation
- **Run**: `streamlit run labs/text_playground/text_playground_app.py`

#### 35. **Simple Streamlit** (`labs/simple_streamlit/`)
- **File**: `simple_streamlit_app.py`
- **Description**: Minimal Streamlit example
- **Run**: `streamlit run labs/simple_streamlit/simple_streamlit_app.py`

## Running the Labs

### For Streamlit Apps

Most labs use Streamlit for interactive web interfaces. To run a Streamlit app:

```bash
streamlit run labs/<lab_name>/<app_file>.py
```

For example:
```bash
streamlit run labs/chatbot/chatbot_app.py
```

The app will open in your default web browser, typically at `http://localhost:8501`.

### For Python Scripts

Some labs are standalone Python scripts that can be run directly:

```bash
python labs/<lab_name>/<script_file>.py
```

For example:
```bash
python labs/api/bedrock_api.py
python labs/embedding/bedrock_embedding.py
```

### Running on AWS SageMaker

If you're running this on AWS SageMaker, use the `getproxyurl.py` script to get the proxy URL:

```bash
python setup/getproxyurl.py
```

This will display the URL to access your Streamlit app through SageMaker Studio.


## License

This project is licensed under the MIT No Attribution License. See the [LICENSE](LICENSE) file for details.

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

## Additional Resources

- [Amazon Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [Amazon Bedrock User Guide](https://docs.aws.amazon.com/bedrock/latest/userguide/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [ChromaDB Documentation](https://docs.trychroma.com/)


##
<img src="data/images/1.png"/>
<img src="data/images/8.png"/>
<img src="data/images/3.png"/>
<img src="data/images/7.png"/>
<img src="data/images/6.png"/>
<img src="data/images/9.png"/>
<img src="data/images/4.png"/>
<img src="data/images/2.png"/>
<img src="data/images/5.gif"/>
<img src="data/images/10.png"/>
