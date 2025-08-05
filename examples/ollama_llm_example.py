"""
Ollama Open-Source LLM Example for Keyword Enhancement with RaKUn2

This example demonstrates how to use RaKUn2 for initial keyword extraction
and then enhance the keywords using Ollama, which runs open-source LLMs
locally without requiring any external API keys.

Requirements:
1. Install Ollama: https://ollama.ai/
2. Pull a model: `ollama pull llama2` or `ollama pull mistral`
3. Install ollama-python: `pip install ollama`

The example shows how to use local LLMs to improve RaKUn2's keyword output.
"""

import warnings
warnings.filterwarnings("ignore")

from rakun2 import RakunKeyphraseDetector
import json
import time

# Configuration parameters for RaKUn2 keyword extraction
RAKUN_HYPERPARAMETERS = {
    "num_keywords": 10,
    "merge_threshold": 1.1,
    "alpha": 0.3,
    "token_prune_len": 3
}

def check_ollama_installation():
    """Check if Ollama is installed and running"""
    try:
        import ollama
        # Test connection
        ollama.list()
        return True
    except ImportError:
        print("Error: ollama-python not installed")
        print("Install with: pip install ollama")
        return False
    except Exception as e:
        print(f"Error: Ollama not running or not installed")
        print("Please install Ollama from https://ollama.ai/")
        print("Then start it and pull a model, e.g.: ollama pull llama2")
        return False

def install_ollama_python():
    """Install ollama-python if not available"""
    try:
        import ollama
        return True
    except ImportError:
        print("Installing ollama-python...")
        import subprocess
        import sys
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "ollama"])
            print("ollama-python installed successfully!")
            import ollama
            return True
        except Exception as e:
            print(f"Failed to install ollama-python: {e}")
            return False

def get_available_models():
    """Get list of available Ollama models"""
    try:
        import ollama
        models = ollama.list()
        return [model['name'] for model in models['models']]
    except Exception as e:
        print(f"Error getting models: {e}")
        return []

def enhance_keywords_with_ollama(keywords, text_context="", model_name="llama2"):
    """
    Enhance keywords using Ollama LLM.
    
    Args:
        keywords: List of [keyword, score] pairs from RaKUn
        text_context: Original text for context (optional)
        model_name: Ollama model to use (e.g., 'llama2', 'mistral', 'codellama')
    
    Returns:
        Enhanced keywords list
    """
    if not install_ollama_python():
        return rule_based_enhancement(keywords)
    
    try:
        import ollama
        
        # Check if model is available
        available_models = get_available_models()
        if not available_models:
            print("No Ollama models found. Please install a model first:")
            print("Example: ollama pull llama2")
            return rule_based_enhancement(keywords)
        
        if model_name not in available_models:
            print(f"Model '{model_name}' not found. Available models: {available_models}")
            model_name = available_models[0]  # Use first available model
            print(f"Using model: {model_name}")
        
        enhanced_keywords = []
        
        # Prepare the prompt for keyword enhancement
        keywords_list = []
        for i, (keyword, score) in enumerate(keywords[:10], 1):  # Top 10 keywords
            keywords_list.append(f"{i}. {keyword} (score: {score:.3f})")
        
        keywords_text = "\n".join(keywords_list)
        
        prompt = f"""You are an expert at improving keyword phrases to make them more descriptive and meaningful. 

I have extracted these keywords from a text using an algorithm:

{keywords_text}

Please improve each keyword by making it more descriptive and contextual while keeping the same meaning. Transform single words into meaningful phrases that better capture the concept.

Respond with a JSON list where each item has "original", "enhanced", and "score" fields. Keep the same scores.

Example format:
[
    {{"original": "learning", "enhanced": "machine learning techniques", "score": 0.045}},
    {{"original": "data", "enhanced": "data processing methods", "score": 0.032}}
]

Response:"""
        
        print(f"Querying Ollama model: {model_name}...")
        print("This may take a moment...")
        
        try:
            # Query Ollama
            response = ollama.chat(model=model_name, messages=[
                {
                    'role': 'user',
                    'content': prompt
                }
            ])
            
            response_text = response['message']['content'].strip()
            
            # Try to parse JSON response
            try:
                # Extract JSON if it's wrapped in text
                start_idx = response_text.find('[')
                end_idx = response_text.rfind(']') + 1
                
                if start_idx != -1 and end_idx != 0:
                    json_text = response_text[start_idx:end_idx]
                    parsed_response = json.loads(json_text)
                    
                    # Convert to our format
                    for item in parsed_response:
                        if 'enhanced' in item and 'score' in item:
                            enhanced_keywords.append([item['enhanced'], float(item['score'])])
                    
                    # If we got results, return them
                    if enhanced_keywords:
                        # Fill remaining keywords if we didn't get all
                        for i in range(len(enhanced_keywords), len(keywords)):
                            enhanced_keywords.append(keywords[i])
                        return enhanced_keywords
                        
            except json.JSONDecodeError:
                print("Could not parse JSON response, trying text parsing...")
                pass
            
            # Fallback: try to extract enhanced keywords from text response
            enhanced_keywords = parse_text_response(response_text, keywords)
            
        except Exception as e:
            print(f"Error during Ollama query: {e}")
            enhanced_keywords = rule_based_enhancement(keywords)
            
        return enhanced_keywords
        
    except Exception as e:
        print(f"Error with Ollama: {e}")
        return rule_based_enhancement(keywords)

def parse_text_response(response_text, original_keywords):
    """Parse text response when JSON parsing fails"""
    enhanced_keywords = []
    lines = response_text.split('\n')
    
    enhanced_found = []
    for line in lines:
        line = line.strip()
        if line and ('.' in line or ':' in line):
            # Try to extract enhanced keyword
            if '"enhanced":' in line:
                start = line.find('"enhanced":') + 11
                end = line.find('"', start + 1)
                if start > 10 and end > start:
                    enhanced = line[start+1:end]
                    enhanced_found.append(enhanced)
            elif '=>' in line or '->' in line:
                # Pattern like "original => enhanced"
                parts = line.split('=>' if '=>' in line else '->')
                if len(parts) == 2:
                    enhanced_found.append(parts[1].strip().strip('"'))
            elif line.startswith(('- ', '* ', str(len(enhanced_found)+1)+'.')):
                # List format
                clean_line = line.lstrip('- *0123456789. ').strip().strip('"')
                if clean_line and len(clean_line) > 3:
                    enhanced_found.append(clean_line)
    
    # Match enhanced keywords with originals
    for i, (original_kw, score) in enumerate(original_keywords):
        if i < len(enhanced_found):
            enhanced_keywords.append([enhanced_found[i], score])
        else:
            enhanced_keywords.append([original_kw, score])
    
    return enhanced_keywords

def rule_based_enhancement(keywords):
    """
    Fallback rule-based keyword enhancement when Ollama is not available
    """
    enhanced_keywords = []
    
    # Simple rules to make keywords more descriptive
    enhancement_rules = {
        'learning': 'machine learning algorithms',
        'intelligence': 'artificial intelligence systems', 
        'networks': 'neural network architectures',
        'data': 'data analysis methods',
        'algorithms': 'computational algorithms',
        'model': 'predictive modeling',
        'system': 'intelligent systems',
        'methods': 'analytical methods',
        'analysis': 'statistical analysis',
        'research': 'scientific research',
        'processing': 'information processing',
        'language': 'natural language processing',
        'computer': 'computer science',
        'vision': 'computer vision'
    }
    
    for keyword, score in keywords:
        enhanced = keyword
        for key, enhancement in enhancement_rules.items():
            if key in keyword.lower():
                enhanced = enhancement
                break
        # If no rule matched, make it more descriptive
        if enhanced == keyword and len(keyword.split()) == 1:
            enhanced = f"{keyword} techniques"
            
        enhanced_keywords.append([enhanced, score])
    
    return enhanced_keywords

def main():
    """Main example demonstrating Ollama LLM enhancement of RaKUn keywords"""
    
    # Sample document - you can replace this with your own text
    example_document = """
    Artificial intelligence has revolutionized many industries through machine learning 
    and deep learning techniques. Natural language processing enables computers to 
    understand and generate human language, while computer vision allows machines to 
    interpret visual information. These technologies rely on neural networks, which 
    are computational models inspired by biological neural networks. Data science 
    combines statistics, programming, and domain expertise to extract insights from 
    large datasets. Modern AI systems use sophisticated algorithms to perform tasks 
    like recommendation systems, autonomous vehicles, and predictive analytics.
    """
    
    print("=== RaKUn2 + Ollama LLM Enhancement Example ===\n")
    
    # Check Ollama availability
    if not check_ollama_installation():
        print("Proceeding with rule-based enhancement as demonstration...\n")
    
    # Step 1: Extract keywords using RaKUn2
    print("Step 1: Extracting keywords with RaKUn2...")
    
    keyword_detector = RakunKeyphraseDetector(RAKUN_HYPERPARAMETERS)
    original_keywords = keyword_detector.find_keywords(example_document, input_type="string")
    
    print("Original RaKUn2 keywords:")
    for i, (keyword, score) in enumerate(original_keywords, 1):
        print(f"{i:2d}. {keyword:<25} (score: {score:.4f})")
    
    print("\n" + "="*60)
    
    # Step 2: Enhance keywords using Ollama LLM
    print("\nStep 2: Enhancing keywords with Ollama LLM...")
    
    # Get available models
    available_models = get_available_models()
    if available_models:
        print(f"Available models: {available_models}")
        # Use the first available model, preferring common ones
        preferred_models = ['llama2', 'mistral', 'codellama', 'llama3']
        model_to_use = None
        for preferred in preferred_models:
            for available in available_models:
                if preferred in available.lower():
                    model_to_use = available
                    break
            if model_to_use:
                break
        
        if not model_to_use:
            model_to_use = available_models[0]
    else:
        model_to_use = 'llama2'  # Default fallback
    
    enhanced_keywords = enhance_keywords_with_ollama(
        original_keywords, 
        text_context=example_document,
        model_name=model_to_use
    )
    
    print("\nEnhanced keywords:")
    for i, (keyword, score) in enumerate(enhanced_keywords, 1):
        print(f"{i:2d}. {keyword:<35} (score: {score:.4f})")
    
    print("\n" + "="*60)
    print("\nComparison:")
    print(f"{'Original':<30} | {'Enhanced'}")
    print("-" * 70)
    
    for (orig_kw, orig_score), (enh_kw, enh_score) in zip(original_keywords[:5], enhanced_keywords[:5]):
        print(f"{orig_kw:<30} | {enh_kw}")
    
    print("\n=== Example Complete ===")
    print("\nSetup Instructions:")
    print("1. Install Ollama: https://ollama.ai/")
    print("2. Pull a model: ollama pull llama2")
    print("3. Install Python client: pip install ollama")
    print("\nTips for better results:")
    print("1. Use larger models like 'llama2:13b' or 'mistral:7b' for better enhancement")
    print("2. Experiment with different models based on your domain")
    print("3. Customize the prompt for your specific use case")
    print("4. Provide more context about your text domain")

if __name__ == "__main__":
    main()