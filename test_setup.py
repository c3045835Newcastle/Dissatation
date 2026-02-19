"""
Test script to validate the base model setup (without actually loading the model).
This tests the code structure and imports.
"""

import sys


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        import config
        print("✓ config.py imported successfully")
    except Exception as e:
        print(f"✗ Failed to import config: {e}")
        return False
    
    try:
        from llama_base_model import BaseLlama31Model
        print("✓ llama_base_model.py imported successfully")
    except Exception as e:
        print(f"✗ Failed to import llama_base_model: {e}")
        return False
    
    try:
        import inference
        print("✓ inference.py imported successfully")
    except Exception as e:
        print(f"✗ Failed to import inference: {e}")
        return False
    
    return True


def test_config():
    """Test configuration values."""
    print("\nTesting configuration...")
    
    import config
    
    # Check that it's the base model
    if "Meta-Llama-3.1-8B" in config.MODEL_NAME:
        print(f"✓ Correct base model configured: {config.MODEL_NAME}")
    else:
        print(f"✗ Unexpected model name: {config.MODEL_NAME}")
        return False
    
    # Check basic parameters
    if config.MAX_NEW_TOKENS > 0:
        print(f"✓ MAX_NEW_TOKENS is valid: {config.MAX_NEW_TOKENS}")
    else:
        print(f"✗ Invalid MAX_NEW_TOKENS: {config.MAX_NEW_TOKENS}")
        return False
    
    if 0 <= config.TEMPERATURE <= 2:
        print(f"✓ TEMPERATURE is valid: {config.TEMPERATURE}")
    else:
        print(f"✗ Invalid TEMPERATURE: {config.TEMPERATURE}")
        return False
    
    return True


def test_dependencies():
    """Test that required packages are importable."""
    print("\nTesting dependencies...")
    
    required = ['torch', 'transformers', 'accelerate']
    all_ok = True
    
    for package in required:
        try:
            __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} is not installed (run: pip install -r requirements.txt)")
            all_ok = False
    
    return all_ok


def main():
    """Run all tests."""
    print("="*60)
    print("Base Llama 3.1 8B Model - Setup Validation")
    print("="*60)
    print()
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Configuration", test_config()))
    results.append(("Dependencies", test_dependencies()))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        symbol = "✓" if passed else "✗"
        print(f"{symbol} {name}: {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n✓ All tests passed! Setup looks good.")
        print("\nNote: This doesn't test actual model loading.")
        print("To test the full model, run: python inference.py")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
