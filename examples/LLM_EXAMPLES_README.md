# RaKUn2 + Open Source LLMs Examples

This directory contains examples demonstrating how to enhance RaKUn2's keyword extraction results using open-source Large Language Models (LLMs) without requiring any external API keys.

## Overview

RaKUn2 excels at fast, algorithmic keyword extraction, while LLMs can enhance these keywords by:
- Making single words more descriptive
- Adding contextual meaning
- Improving semantic clarity
- Transforming technical terms into readable phrases

## Examples

### 1. HuggingFace Transformers Example
**File**: `huggingface_llm_example.py`

Shows how to use HuggingFace Transformers library with models like GPT-2, DialoGPT, or other open-source models to enhance keywords.

**Requirements**:
```bash
pip install transformers torch
```

**Usage**:
```bash
python huggingface_llm_example.py
```

**Features**:
- Uses free, open-source models from HuggingFace
- Fallback to rule-based enhancement if model unavailable
- CPU-optimized for broader compatibility
- Demonstrates prompt engineering for keyword enhancement

### 2. Ollama Local LLM Example
**File**: `ollama_llm_example.py`

Demonstrates using Ollama to run LLMs locally for complete privacy and no internet dependency.

**Requirements**:
1. Install Ollama: https://ollama.ai/
2. Pull a model: `ollama pull llama2`
3. Install Python client: `pip install ollama`

**Usage**:
```bash
python ollama_llm_example.py
```

**Features**:
- Runs completely locally
- No internet required after setup
- Privacy-friendly
- Supports various models (Llama2, Mistral, CodeLlama, etc.)

### 3. Jupyter Notebook / Google Colab Example
**File**: `rakun2_open_source_llm_examples.ipynb`

Interactive notebook that can be run in Google Colab or locally, showing both approaches with detailed explanations.

**Features**:
- Step-by-step tutorial
- Interactive examples
- Comparison of different methods
- Ready for Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SkBlaz/rakun2/blob/main/examples/rakun2_open_source_llm_examples.ipynb)

## Comparison with Commercial APIs

| Aspect | RaKUn2 + Open Source LLMs | Commercial APIs |
|--------|---------------------------|-----------------|
| **Cost** | Free | Pay per API call |
| **Privacy** | Complete privacy | Data sent to external servers |
| **Speed** | Fast (local processing) | Network dependent |
| **Setup** | One-time model download | API key required |
| **Offline** | Works offline | Requires internet |
| **Customization** | Full control | Limited |

## Quick Start

1. **Install RaKUn2**:
   ```bash
   pip install rakun2
   ```

2. **Choose your LLM approach**:
   - For **cloud/Colab**: Use HuggingFace example
   - For **local/privacy**: Use Ollama example
   - For **learning**: Use the Jupyter notebook

3. **Run the example**:
   ```bash
   python huggingface_llm_example.py
   ```

## Example Output

```
Original RaKUn2 keywords:
 1. artificial intelligence     (score: 0.3939)
 2. machine learning           (score: 0.2740)
 3. neural networks            (score: 0.2456)

Enhanced keywords:
 1. artificial intelligence systems    (score: 0.3939)
 2. machine learning techniques        (score: 0.2740)
 3. neural network architectures       (score: 0.2456)
```

## Model Recommendations

### HuggingFace Models:
- **GPT-2**: Good general-purpose model, fast
- **DialoGPT-medium**: Better conversational understanding
- **CodeBERT**: Excellent for technical/programming content
- **SciBERT**: Optimized for scientific text

### Ollama Models:
- **llama2**: General purpose, good balance
- **mistral**: Fast and capable
- **codellama**: Best for technical content
- **llama2:13b**: Larger model for better quality

## Customization Tips

1. **Adjust prompts** for your domain
2. **Experiment with model parameters** (temperature, max_tokens)
3. **Use domain-specific models** when available
4. **Combine multiple enhancement strategies**

## Troubleshooting

**HuggingFace Issues**:
- Models not downloading: Check internet connection
- Memory errors: Use smaller models or CPU-only mode
- Slow performance: Consider model caching

**Ollama Issues**:
- Connection errors: Ensure Ollama is running (`ollama serve`)
- No models: Pull a model first (`ollama pull llama2`)
- Performance: Consider using smaller models or adjusting parameters

## Contributing

Feel free to contribute additional examples or improvements:
- New model integrations
- Domain-specific enhancements
- Performance optimizations
- Better prompting strategies

## License

These examples are provided under the same license as RaKUn2.