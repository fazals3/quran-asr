---
language:
- ar
base_model:
- tarteel-ai/whisper-base-ar-quran
---
# Ultra Fast Tarteel CoreML

This model is a WhisperKit/CoreML conversion of [OdyAsh/faster-whisper-base-ar-quran](https://huggingface.co/OdyAsh/faster-whisper-base-ar-quran), which itself is a conversion of the original [tarteel-ai/whisper-base-ar-quran](https://huggingface.co/tarteel-ai/whisper-base-ar-quran) model.

## Model Chain
```
tarteel-ai/whisper-base-ar-quran → OdyAsh/faster-whisper-base-ar-quran → fazalshaikh123/ultra-fast-tarteel-coreml
```

## Performance

Tested on **iPhone 15 Pro Max**:
- **Audio Duration**: 1 hour 50 minutes (Surah Al-Baqarah)
- **Transcription Time**: ~200 seconds
- **Real-time Factor**: ~0.03x (33x faster than real-time)

This means the model can transcribe Arabic Quranic audio approximately **33 times faster** than the original audio duration on modern iOS devices.

## Usage

This model is optimized for use with WhisperKit/CoreML on iOS devices and is specifically fine-tuned for Arabic Quranic recitation transcription.

## Credits

- Original model: [Tarteel AI](https://huggingface.co/tarteel-ai/whisper-base-ar-quran)
- Faster-whisper conversion: [OdyAsh](https://huggingface.co/OdyAsh/faster-whisper-base-ar-quran)
- WhisperKit/CoreML conversion: fazalshaikh123

## License

Please refer to the original model licenses for usage terms and conditions.