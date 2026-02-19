"""
Base Llama 3.1 8B Model Loader
This script loads the base (pre-trained, not fine-tuned) Llama 3.1 8B model locally.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import config


class BaseLlama31Model:
    """
    Wrapper class for the base Llama 3.1 8B model.
    This is the BASE model without any fine-tuning.
    """
    
    def __init__(self):
        """Initialize the base Llama 3.1 8B model."""
        print(f"Loading base Llama 3.1 8B model: {config.MODEL_NAME}")
        print("Note: This is the BASE model (not fine-tuned)")
        
        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.MODEL_NAME,
            cache_dir=config.MODEL_CACHE_DIR,
            trust_remote_code=True
        )
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with appropriate settings
        print("Loading model...")
        model_kwargs = {
            "cache_dir": config.MODEL_CACHE_DIR,
            "device_map": config.DEVICE,
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            "trust_remote_code": True
        }
        
        # Add quantization if specified
        if config.LOAD_IN_8BIT:
            model_kwargs["load_in_8bit"] = True
        elif config.LOAD_IN_4BIT:
            model_kwargs["load_in_4bit"] = True
        
        self.model = AutoModelForCausalLM.from_pretrained(
            config.MODEL_NAME,
            **model_kwargs
        )
        
        print("Base model loaded successfully!")
        print(f"Device: {next(self.model.parameters()).device}")
    
    def generate(self, prompt, max_new_tokens=None, temperature=None, 
                 top_p=None, top_k=None, do_sample=None):
        """
        Generate text using the base model.
        
        Args:
            prompt (str): Input prompt text
            max_new_tokens (int): Maximum number of tokens to generate
            temperature (float): Sampling temperature
            top_p (float): Nucleus sampling parameter
            top_k (int): Top-k sampling parameter
            do_sample (bool): Whether to use sampling
            
        Returns:
            str: Generated text
        """
        # Use config defaults if not specified
        max_new_tokens = max_new_tokens or config.MAX_NEW_TOKENS
        temperature = temperature or config.TEMPERATURE
        top_p = top_p or config.TOP_P
        top_k = top_k or config.TOP_K
        do_sample = do_sample if do_sample is not None else config.DO_SAMPLE
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode and return
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    
    def chat(self, messages):
        """
        Chat interface for the model.
        
        Args:
            messages (list): List of message dictionaries with 'role' and 'content'
            
        Returns:
            str: Model response
        """
        # Apply chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            # Simple fallback
            prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
            prompt += "\nassistant: "
        
        response = self.generate(prompt)
        
        # Extract only the new response
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        return response


def main():
    """Main function to demonstrate the base model."""
    print("="*60)
    print("Base Llama 3.1 8B Model - Local Deployment")
    print("="*60)
    
    # Initialize model
    model = BaseLlama31Model()
    
    # Example usage
    print("\n" + "="*60)
    print("Example 1: Simple text generation")
    print("="*60)
    
    prompt = "The capital of France is"
    print(f"\nPrompt: {prompt}")
    response = model.generate(prompt, max_new_tokens=50)
    print(f"\nResponse: {response}")
    
    print("\n" + "="*60)
    print("Example 2: Chat-style interaction")
    print("="*60)
    
    messages = [
        {"role": "user", "content": "What is machine learning?"}
    ]
    print(f"\nUser: {messages[0]['content']}")
    response = model.chat(messages)
    print(f"\nAssistant: {response}")


if __name__ == "__main__":
    main()
