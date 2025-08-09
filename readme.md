# বাংলা টেক্সট টু স্পিচ (Bengali Text-to-Speech) Web App

### Developed by Imtiaz Khandoker (Mohsin)

![Bangla AI UI](bangla_ai.png)

A Streamlit web application that converts Bengali text to natural-sounding speech using Facebook's MMS-TTS Bengali model.

## Features

- Convert Bengali text to high-quality speech
- Download generated audio as WAV file
- Sample texts for quick testing
- Simple and intuitive interface
- Works entirely in your browser

## Prerequisites

### System Dependencies

**For macOS:**
```bash
brew install libsndfile sox
```

# Clone the repository
git clone https://github.com/yourusername/bengali-tts.git
cd bengali-tts

# Create and activate virtual environment
python3 -m venv bangla_tts_env
source bangla_tts_env/bin/activate  # On Windows: bangla_tts_env\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# run
streamlit run bangla_tts_app.py


bengali-tts/
├── bangla_tts_app.py       # Main application script
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── bangla_ai.png           # Application screenshot
└── bangla_tts_env/         # Virtual environment directory


Usage Instructions
Launch the application

Type or paste Bengali text in the input box

Click "কণ্ঠস্বর তৈরি করুন" (Generate Voice) button

Listen to the generated audio

Download the WAV file if desired


### Requirements

streamlit==1.31.0
torch==2.1.0
numpy==1.26.0
transformers==4.35.0
soundfile==0.12.1
scipy==1.11.3
torchaudio==2.1.0