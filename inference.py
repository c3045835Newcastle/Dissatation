"""
Simple inference script for testing the base Llama 3.1 8B model.
Run this script to interact with the model.
"""

from llama_base_model import BaseLlama31Model
import sys


def interactive_mode(model):
    """Run the model in interactive mode."""
    print("\n" + "="*60)
    print("Interactive Mode - Base Llama 3.1 8B")
    print("="*60)
    print("Type your prompts below. Type 'quit' or 'exit' to stop.")
    print("="*60 + "\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Generate response
            print("\nGenerating response...\n")
            response = model.generate(user_input, max_new_tokens=256)
            
            # Extract just the generated part (after the prompt)
            if response.startswith(user_input):
                response = response[len(user_input):].strip()
            
            print(f"Assistant: {response}\n")
            print("-"*60 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


def single_prompt_mode(model, prompt):
    """Run a single prompt through the model."""
    print("\n" + "="*60)
    print("Single Prompt Mode - Base Llama 3.1 8B")
    print("="*60)
    print(f"\nPrompt: {prompt}\n")
    
    response = model.generate(prompt)
    print(f"Response:\n{response}\n")


def main():
    """Main function."""
    print("Initializing Base Llama 3.1 8B Model...")
    print("This may take a few moments on first run (downloading model)...")
    print("="*60)
    
    # Initialize model
    model = BaseLlama31Model()
    
    # Check if prompt provided as command line argument
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
        single_prompt_mode(model, prompt)
    else:
        interactive_mode(model)


if __name__ == "__main__":
    main()
