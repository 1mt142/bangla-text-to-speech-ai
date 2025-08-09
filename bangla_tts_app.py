import streamlit as st
import torch
import numpy as np
from transformers import pipeline
import tempfile
import os
import base64
from io import BytesIO
import warnings
import wave
import struct

warnings.filterwarnings("ignore")

TORCHAUDIO_AVAILABLE = False
SCIPY_AVAILABLE = False
SOUNDFILE_AVAILABLE = False

try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    pass 

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    pass 

try:
    from scipy.io import wavfile
    SCIPY_AVAILABLE = True
except ImportError:
    pass 


st.set_page_config(page_title="বাংলা টেক্সট টু স্পিচ", page_icon="🎙️", layout="wide")

st.markdown(
    """
<style>
    .main-header {
        text-align: center;
        color: #2E8B57;
        font-size: 3em;
        margin-bottom: 20px;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2em;
        margin-bottom: 30px;
    }
    .stTextArea > div > div > textarea {
        font-size: 18px;
        font-family: 'SolaimanLipi', 'Kalpurush', sans-serif;
    }
    .generation-status {
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .success {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .error {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    .info {
        background-color: #d1ecf1;
        color: #0c5460;
        border: 1px solid #bee5eb;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_tts_model():
    """Load the TTS model with caching for better performance"""
    try:
        tts_pipeline = pipeline(
            "text-to-speech",
            model="facebook/mms-tts-ben",
            torch_dtype=torch.float32,
            device=-1,  
        )
        return tts_pipeline, True
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        try:
            from transformers import VitsModel, AutoTokenizer
            model = VitsModel.from_pretrained("facebook/mms-tts-ben")
            tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-ben")

            return (model, tokenizer), True
        except Exception as e2:
            st.error(f"Alternative model loading also failed: {str(e2)}")
            return None, False


def generate_audio(text, model):
    """Generate audio from Bengali text"""
    try:
        if isinstance(model, tuple):
            vits_model, tokenizer = model
            inputs = tokenizer(text, return_tensors="pt")

            with torch.no_grad():
                output = vits_model(inputs["input_ids"])

            audio_data = output.waveform.squeeze().cpu().numpy()
            sampling_rate = vits_model.config.sampling_rate

        else:
            result = model(text)
            audio_data = result["audio"]
            sampling_rate = result["sampling_rate"]

            if torch.is_tensor(audio_data):
                audio_data = audio_data.cpu().numpy()

        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        if audio_data.ndim > 1:
            audio_data = audio_data.flatten()

        return audio_data, sampling_rate, True

    except Exception as e:
        return None, None, False, str(e)


def save_audio(audio_data, sampling_rate, filename="bangla_tts_output"):
    """Save audio data to a temporary file with multiple fallback methods"""
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_path = temp_file.name
        temp_file.close()  

        if torch.is_tensor(audio_data):
            audio_data = audio_data.cpu().numpy()

        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = audio_data / np.max(np.abs(audio_data))

        success = False
        error_msg = ""

        if TORCHAUDIO_AVAILABLE and not success:
            try:
                audio_tensor = torch.from_numpy(audio_data)
                if audio_tensor.dim() == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)
                torchaudio.save(temp_path, audio_tensor, sampling_rate)
                success = True
            except Exception as e:
                error_msg += f"Torchaudio failed: {str(e)}; "

        if SOUNDFILE_AVAILABLE and not success:
            try:
                sf.write(temp_path, audio_data, sampling_rate)
                success = True
            except Exception as e:
                error_msg += f"Soundfile failed: {str(e)}; "

        if SCIPY_AVAILABLE and not success:
            try:
                audio_int16 = (audio_data * 32767).astype(np.int16)
                wavfile.write(temp_path, sampling_rate, audio_int16)
                success = True
            except Exception as e:
                error_msg += f"Scipy failed: {str(e)}; "

        if not success:
            try:
                audio_int16 = (audio_data * 32767).astype(np.int16)

                with wave.open(temp_path, "wb") as wav_file:
                    wav_file.setnchannels(1)  
                    wav_file.setsampwidth(2)  
                    wav_file.setframerate(sampling_rate)
                    wav_file.writeframes(audio_int16.tobytes())

                success = True
            except Exception as e:
                error_msg += f"Manual WAV failed: {str(e)}; "

        if success:
            return temp_path, True
        else:
            st.error(f"All audio saving methods failed: {error_msg}")
            return None, False

    except Exception as e:
        st.error(f"Audio saving error: {str(e)}")
        return None, False


def get_download_link(file_path, filename="bangla_audio.wav"):
    """Generate download link for the audio file"""
    try:
        with open(file_path, "rb") as f:
            audio_bytes = f.read()

        b64_audio = base64.b64encode(audio_bytes).decode()
        download_link = f'<a href="data:audio/wav;base64,{b64_audio}" download="{filename}" style="background-color:#4CAF50;color:white;padding:10px 20px;text-decoration:none;border-radius:5px;display:inline-block;margin-top:10px;">📥 Download Audio File</a>'
        return download_link
    except Exception as e:
        return None


def main():
    st.markdown(
        '<h1 class="main-header">🎙️ বাংলা টেক্সট টু স্পিচ</h1>', unsafe_allow_html=True
    )
    st.markdown(
        '<p class="subtitle">বাংলা টেক্সট থেকে প্রাকৃতিক কণ্ঠস্বর তৈরি করুন</p>',
        unsafe_allow_html=True,
    )

    with st.spinner("মডেল লোড হচ্ছে... (প্রথমবার কিছুটা সময় লাগতে পারে)"):
        tts_model, model_loaded = load_tts_model()

    if not model_loaded:
        st.error("মডেল লোড করতে সমস্যা হয়েছে। পেজ রিফ্রেশ করে আবার চেষ্টা করুন।")
        st.stop()

    st.success("মডেল সফলভাবে লোড হয়েছে!")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader(" বাংলা টেক্সট লিখুন:")

        sample_texts = {
            "নমুনা ১": "আমার নাম ইমতিয়াজ খন্দকার মহসিন।",
            "নমুনা ২": "আজ আবহাওয়া খুবই সুন্দর।",
            "নমুনা ৩": "বাংলাদেশ আমার প্রিয় দেশ।",
            "নমুনা ৪": "প্রযুক্তির যুগে আমরা এগিয়ে চলেছি।",
        }

        selected_sample = st.selectbox(
            "অথবা নমুনা টেক্সট নির্বাচন করুন:",
            ["কাস্টম টেক্সট"] + list(sample_texts.keys()),
        )

        if selected_sample != "কাস্টম টেক্সট":
            default_text = sample_texts[selected_sample]
        else:
            default_text = ""

        bangla_text = st.text_area(
            "এখানে বাংলা টেক্সট লিখুন:",
            value=default_text,
            height=150,
            placeholder="এখানে আপনার বাংলা টেক্সট টাইপ করুন...",
            help="যেকোনো বাংলা টেক্সট লিখুন যা আপনি শুনতে চান",
        )

        generate_button = st.button("🎵 কণ্ঠস্বর তৈরি করুন", type="primary")

    with col2:
        st.subheader("নির্দেশাবলী:")
        st.markdown(
            """
        1. **বাংলা টেক্সট** লিখুন বা নমুনা নির্বাচন করুন
        2. **"কণ্ঠস্বর তৈরি করুন"** বোতামে ক্লিক করুন
        3. **অডিও প্লেয়ার** দিয়ে শুনুন
        4. **ডাউনলোড লিংক** দিয়ে ফাইল সেভ করুন
        """
        )

        st.subheader("⚡ বৈশিষ্ট্য:")
        st.markdown(
            """
        -  **সম্পূর্ণ বিনামূল্যে**
        -  **বাংলা ভাষা সাপোর্ট**
        -  **সহজ ইন্টারফেস**
        -  **WAV ফরম্যাটে ডাউনলোড**
        -  **প্রাকৃতিক কণ্ঠস্বর**
        """
        )

    if generate_button:
        if not bangla_text.strip():
            st.warning("অনুগ্রহ করে কিছু টেক্সট লিখুন!")
        else:
            with st.spinner("কণ্ঠস্বর তৈরি হচ্ছে... অনুগ্রহ করে অপেক্ষা করুন।"):
                try:
                    result = generate_audio(bangla_text, tts_model)

                    if len(result) == 4: 
                        audio_data, sampling_rate, success, error = result
                        st.error(f"অডিও তৈরিতে সমস্যা: {error}")
                    else: 
                        audio_data, sampling_rate, success = result

                        if success:
                            temp_file_path, save_success = save_audio(
                                audio_data, sampling_rate
                            )

                            if save_success:
                                st.success(" কণ্ঠস্বর সফলভাবে তৈরি হয়েছে!")
                                st.subheader("তৈরি হওয়া অডিও:")
                                with open(temp_file_path, "rb") as audio_file:
                                    audio_bytes = audio_file.read()
                                    st.audio(audio_bytes, format="audio/wav")

                                st.subheader("ডাউনলোড:")
                                download_link = get_download_link(temp_file_path)

                                if download_link:
                                    st.markdown(download_link, unsafe_allow_html=True)

                                    with open(temp_file_path, "rb") as file:
                                        st.download_button(
                                            label="ডাউনলোড অডিও",
                                            data=file.read(),
                                            file_name="bangla_tts_output.wav",
                                            mime="audio/wav",
                                            type="secondary",
                                        )

                                st.info(
                                    f"অডিও তথ্য: {len(audio_data)/sampling_rate:.2f} সেকেন্ড, {sampling_rate} Hz"
                                )

                                try:
                                    os.unlink(temp_file_path)
                                except:
                                    pass

                            else:
                                st.error("অডিও ফাইল সেভ করতে সমস্যা হয়েছে।")
                        else:
                            st.error("অডিও তৈরিতে সমস্যা হয়েছে।")

                except Exception as e:
                    st.error(f"একটি অপ্রত্যাশিত ত্রুটি ঘটেছে: {str(e)}")

    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; font-size: 0.9em;'>
        🔧 <strong>বাংলা টেক্সট টু স্পিচ অ্যাপ</strong> | 
        ব্যবহৃত মডেল: <a href='https://huggingface.co/facebook/mms-tts-ben' target='_blank'>Facebook MMS-TTS Bengali</a> | 
        সম্পূর্ণ ওপেন সোর্স
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()