#!/bin/bash
# COG-JEPA Setup Script for macOS M4
# One-shot installation for Apple Silicon

set -e

echo "=========================================="
echo "COG-JEPA Setup for Apple Silicon M4"
echo "=========================================="

# Check Python version
echo "Checking Python version..."
python3 --version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if [ "$(printf '%s\n' "3.11" "$PYTHON_VERSION" | sort -V | head -n1)" != "3.11" ]; then
    echo "Python 3.11+ required. Found: $PYTHON_VERSION"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate venv
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with MPS support
echo "Installing PyTorch with MPS..."
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Install llama-cpp-python with Metal support (optional - fallback available)
echo "Installing llama-cpp-python with Metal (optional)..."
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --no-cache-dir || echo "llama-cpp-python install skipped (fallback mode available)"

# Create directories
echo "Creating directories..."
mkdir -p models
mkdir -p data

# Download GGUF model
echo "Downloading LLM model (this may take a while)..."
if command -v huggingface-cli &> /dev/null; then
    huggingface-cli download bartowski/Llama-3.2-3B-Instruct-GGUF \
        Llama-3.2-3B-Instruct-Q4_K_M.gguf --local-dir models/ || echo "Model download skipped (will use fallback)"
else
    pip install huggingface-hub
    python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='bartowski/Llama-3.2-3B-Instruct-GGUF', filename='Llama-3.2-3B-Instruct-Q4_K_M.gguf', local_dir='models/')" || echo "Model download skipped (will use fallback)"
fi

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Run the following commands:"
echo "  source venv/bin/activate"
echo "  python main.py --mode test      # validates everything works"
echo "  python main.py --mode dashboard # launches the Gradio UI"
echo "  python main.py --mode webcam    # runs on live webcam"
echo ""
