"""
HuggingFace Open-Source LLM Example for Keyword Enhancement with RaKUn2

This example demonstrates how to use RaKUn2 for initial keyword extraction
and then enhance the keywords using open-source LLMs from HuggingFace
Transformers library, without requiring any external API keys.

Requirements:
    pip install transformers torch
    
The example uses a smaller model (GPT-2) for demonstration, but you can
use larger models like CodeLlama, Mistral, or other open-source LLMs.
"""

import warnings
warnings.filterwarnings("ignore")

from rakun2 import RakunKeyphraseDetector
import numpy as np

def install_requirements():
    """Install required dependencies if not available"""
    try:
        import transformers
        import torch
    except ImportError:
        print("Installing required dependencies...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers", "torch"])
        print("Dependencies installed successfully!")

def enhance_keywords_with_huggingface(keywords, text_context="", model_name="microsoft/DialoGPT-medium"):
    """
    Enhance keywords using HuggingFace transformers.
    
    Args:
        keywords: List of [keyword, score] pairs from RaKUn
        text_context: Original text for context (optional)
        model_name: HuggingFace model to use
    
    Returns:
        Enhanced keywords list
    """
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        import torch
    except ImportError:
        print("Installing transformers...")
        install_requirements()
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        import torch
    
    print(f"Loading model: {model_name}")
    print("This may take a moment on first run...")
    
    # Use a text generation pipeline for simplicity
    # Note: For better results, consider using larger models like:
    # - "microsoft/DialoGPT-medium" (conversational)
    # - "gpt2" (general purpose)
    # - "microsoft/CodeBERT-base" (for technical content)
    
    try:
        # Create a text generation pipeline
        generator = pipeline("text-generation", 
                           model=model_name, 
                           device=-1,  # Use CPU to avoid GPU requirements
                           max_length=150,
                           do_sample=True,
                           temperature=0.7,
                           pad_token_id=50256)
        
        enhanced_keywords = []
        
        # Create prompt to enhance keywords
        keywords_str = ", ".join([kw[0] for kw in keywords[:5]])  # Top 5 keywords
        
        prompt = f"""Task: Improve these keywords by making them more descriptive and contextual.
Original keywords: {keywords_str}

Improved keywords:
1."""
        
        # Generate enhanced keywords
        try:
            response = generator(prompt, max_new_tokens=50, num_return_sequences=1)
            generated_text = response[0]['generated_text']
            
            # Extract the generated portion
            enhanced_text = generated_text.replace(prompt, "").strip()
            
            # Parse the response and create enhanced keywords
            lines = enhanced_text.split('\n')
            for i, (original_kw, score) in enumerate(keywords):
                if i < len(lines) and lines[i].strip():
                    # Clean up the generated text
                    enhanced = lines[i].strip()
                    # Remove numbering if present
                    enhanced = enhanced.lstrip('0123456789. ').strip()
                    if enhanced and len(enhanced) > 3:
                        enhanced_keywords.append([enhanced, score])
                    else:
                        enhanced_keywords.append([original_kw, score])
                else:
                    enhanced_keywords.append([original_kw, score])
        
        except Exception as e:
            print(f"Error during generation: {e}")
            print("Using rule-based enhancement as fallback...")
            # Fallback to rule-based enhancement
            enhanced_keywords = rule_based_enhancement(keywords)
            
        return enhanced_keywords
        
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        print("Using rule-based enhancement as fallback...")
        return rule_based_enhancement(keywords)

def rule_based_enhancement(keywords):
    """
    Fallback rule-based keyword enhancement when LLM is not available
    """
    enhanced_keywords = []
    
    # Simple rules to make keywords more descriptive
    enhancement_rules = {
        'learning': 'machine learning techniques',
        'intelligence': 'artificial intelligence systems', 
        'networks': 'neural network architectures',
        'data': 'data processing methods',
        'algorithms': 'computational algorithms',
        'model': 'predictive models',
        'system': 'intelligent systems',
        'methods': 'analytical methods',
        'analysis': 'data analysis techniques',
        'research': 'scientific research approaches'
    }
    
    for keyword, score in keywords:
        enhanced = keyword
        for key, enhancement in enhancement_rules.items():
            if key in keyword.lower():
                enhanced = enhancement
                break
        enhanced_keywords.append([enhanced, score])
    
    return enhanced_keywords

def main():
    """Main example demonstrating HuggingFace LLM enhancement of RaKUn keywords"""
    
    # Sample document - you can replace this with your own text
    example_document = """
    Machine learning is a subset of artificial intelligence that focuses on algorithms 
    that can learn and make decisions from data. Deep learning, a subset of machine learning, 
    uses neural networks with multiple layers to model and understand complex patterns 
    in data. These techniques have revolutionized fields like computer vision, natural 
    language processing, and robotics. Modern AI systems rely heavily on large datasets 
    and computational power to train sophisticated models that can perform tasks like 
    image recognition, language translation, and autonomous driving.
    """
    
    print("=== RaKUn2 + HuggingFace LLM Enhancement Example ===\n")
    
    # Step 1: Extract keywords using RaKUn2
    print("Step 1: Extracting keywords with RaKUn2...")
    
    hyperparameters = {
        "num_keywords": 10,
        "merge_threshold": 1.1,
        "alpha": 0.3,
        "token_prune_len": 3
    }
    
    keyword_detector = RakunKeyphraseDetector(hyperparameters)
    original_keywords = keyword_detector.find_keywords(example_document, input_type="string")
    
    print("Original RaKUn2 keywords:")
    for i, (keyword, score) in enumerate(original_keywords, 1):
        print(f"{i:2d}. {keyword:<25} (score: {score:.4f})")
    
    print("\n" + "="*60)
    
    # Step 2: Enhance keywords using HuggingFace LLM
    print("\nStep 2: Enhancing keywords with HuggingFace LLM...")
    print("Note: This may take some time on first run as model downloads...")
    
    # Try with a smaller, faster model first
    enhanced_keywords = enhance_keywords_with_huggingface(
        original_keywords, 
        text_context=example_document,
        model_name="gpt2"  # Using GPT-2 as it's smaller and faster
    )
    
    print("\nEnhanced keywords:")
    for i, (keyword, score) in enumerate(enhanced_keywords, 1):
        print(f"{i:2d}. {keyword:<35} (score: {score:.4f})")
    
    print("\n" + "="*60)
    print("\nComparison:")
    print(f"{'Original':<25} | {'Enhanced'}")
    print("-" * 60)
    
    for (orig_kw, orig_score), (enh_kw, enh_score) in zip(original_keywords[:5], enhanced_keywords[:5]):
        print(f"{orig_kw:<25} | {enh_kw}")
    
    print("\n=== Example Complete ===")
    print("\nTips for better results:")
    print("1. Use larger models like 'microsoft/DialoGPT-large' for better enhancement")
    print("2. Provide more context in the prompt")
    print("3. Experiment with different models based on your domain")
    print("4. Consider fine-tuned models for specific domains")

if __name__ == "__main__":
    main()