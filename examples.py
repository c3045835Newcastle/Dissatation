"""
Example usage script for the Base Llama 3.1 8B model.
This demonstrates various ways to use the model.
"""

from llama_base_model import BaseLlama31Model
import config


def example_basic_generation():
    """Example 1: Basic text generation."""
    print("\n" + "="*60)
    print("Example 1: Basic Text Generation")
    print("="*60)
    
    model = BaseLlama31Model()
    
    prompts = [
        "The capital of France is",
        "Machine learning is",
        "The theory of relativity states that"
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        response = model.generate(prompt, max_new_tokens=100)
        print(f"Response: {response}\n")
        print("-"*60)


def example_custom_parameters():
    """Example 2: Generation with custom parameters."""
    print("\n" + "="*60)
    print("Example 2: Custom Generation Parameters")
    print("="*60)
    
    model = BaseLlama31Model()
    
    prompt = "Once upon a time in a distant galaxy"
    
    # Creative (high temperature)
    print("\nCreative generation (temperature=1.2):")
    response = model.generate(prompt, max_new_tokens=100, temperature=1.2)
    print(response)
    
    # Deterministic (low temperature)
    print("\nDeterministic generation (temperature=0.1):")
    response = model.generate(prompt, max_new_tokens=100, temperature=0.1)
    print(response)


def example_chat_interface():
    """Example 3: Chat-style interaction."""
    print("\n" + "="*60)
    print("Example 3: Chat Interface")
    print("="*60)
    
    model = BaseLlama31Model()
    
    # Single turn conversation
    messages = [
        {"role": "user", "content": "What is the largest planet in our solar system?"}
    ]
    
    print(f"\nUser: {messages[0]['content']}")
    response = model.chat(messages)
    print(f"Assistant: {response}")
    
    # Multi-turn conversation
    conversation = [
        {"role": "user", "content": "Hi, can you help me with Python?"},
        {"role": "assistant", "content": "Of course! I'd be happy to help with Python. What would you like to know?"},
        {"role": "user", "content": "How do I read a file?"}
    ]
    
    print("\n" + "-"*60)
    print("Multi-turn conversation:")
    for msg in conversation:
        print(f"{msg['role'].capitalize()}: {msg['content']}")
    
    response = model.chat(conversation)
    print(f"Assistant: {response}")


def example_batch_processing():
    """Example 4: Processing multiple prompts."""
    print("\n" + "="*60)
    print("Example 4: Batch Processing")
    print("="*60)
    
    model = BaseLlama31Model()
    
    questions = [
        "What is photosynthesis?",
        "Explain Newton's first law.",
        "What is the water cycle?"
    ]
    
    results = []
    for i, question in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] Processing: {question}")
        response = model.generate(question, max_new_tokens=150)
        results.append({"question": question, "answer": response})
    
    print("\n" + "="*60)
    print("Results Summary:")
    print("="*60)
    for i, result in enumerate(results, 1):
        print(f"\nQ{i}: {result['question']}")
        print(f"A{i}: {result['answer'][:100]}...")  # First 100 chars


def main():
    """Run all examples."""
    print("="*60)
    print("Base Llama 3.1 8B - Usage Examples")
    print("="*60)
    print("\nNote: These examples demonstrate the BASE model.")
    print("First run will download the model (~16GB).")
    print("\nSelect an example to run:")
    print("1. Basic text generation")
    print("2. Custom parameters")
    print("3. Chat interface")
    print("4. Batch processing")
    print("5. Run all examples")
    print("0. Exit")
    
    try:
        choice = input("\nEnter your choice (0-5): ").strip()
        
        if choice == "1":
            example_basic_generation()
        elif choice == "2":
            example_custom_parameters()
        elif choice == "3":
            example_chat_interface()
        elif choice == "4":
            example_batch_processing()
        elif choice == "5":
            example_basic_generation()
            example_custom_parameters()
            example_chat_interface()
            example_batch_processing()
        elif choice == "0":
            print("Goodbye!")
        else:
            print("Invalid choice!")
    
    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye!")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()
