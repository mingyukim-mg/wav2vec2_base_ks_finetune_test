import torch
import librosa
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

TEST_WAV = "data/horn.wav"   # data/call.wav 또는 data/horn.wav
SAMPLE_RATE = 16000

ORIGINAL_MODEL_PATH = "models/pretrained"
FINETUNED_MODEL_PATH = "models/finetuned"

# 파인튜닝 모델 라벨 매핑
LABEL_MAP = {
    0: "background",
    1: "horn",
    2: "call",
    3: "alarm",
}


# 오디오 로드
def load_audio(path):
    speech, sr = librosa.load(path, sr=SAMPLE_RATE)
    return speech


# 예측 함수
def predict(model_path, speech):
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
    model.eval()

    inputs = feature_extractor(
        speech,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred = torch.argmax(logits, dim=-1).item()

    return pred


# 실행
if __name__ == "__main__":
    speech = load_audio(TEST_WAV)

    # 원본 모델
    original_pred = predict(ORIGINAL_MODEL_PATH, speech)
    print(f"원본 모델 예측 결과: {original_pred}")

    # 파인튜닝 모델
    finetuned_pred = predict(FINETUNED_MODEL_PATH, speech)
    label_name = LABEL_MAP.get(finetuned_pred, "unknown")
    print(f"파인튜닝 모델 예측 결과: {label_name}(label: {finetuned_pred})")
