## Developed by Imtiaz Khandoker(Mohsin)

# First, install system dependencies

# Install required audio libraries

brew install libsndfile
brew install sox

# Create virtual environment

python3 -m venv bangla_tts_env
source bangla_tts_env/bin/activate

# Install Python packages (try in this order)

pip install --upgrade pip
pip install streamlit torch transformers numpy scipy

# Try to install soundfile (recommended for macOS)

pip install soundfile

# If soundfile fails, try these alternatives:

pip install librosa

# OR

pip install pydub

# Finally install the rest

pip install huggingface-hub

# If you still have issues, try:

pip install torchaudio --index-url https://download.pytorch.org/whl/cpu
