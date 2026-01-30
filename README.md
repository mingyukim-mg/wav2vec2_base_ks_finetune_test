# Audio Event Inference (wav2vec2 파인튜닝 모델)

`superb/wav2vec2-base-superb-ks` 모델을 파인튜닝한  
오디오 이벤트 분류 모델의 추론(inference) 예제입니다.

모델은 짧은 오디오 클립을 다음 2가지 클래스로 분류합니다.

- horn(1)
- call(2)
아직 튜닝 안 한 라벨
- background(0)
- alarm(3)

---

## 1. 실행 환경

- Python 3.10.xx (Python 3.10.12 기준으로 테스트됨)
- OS: Linux / macOS  
  (Windows에서도 실행 가능)

---

## 2. 설치 방법

가상환경 생성 (권장):
bash
python -m venv venv
source venv/bin/activate

## 3. 필요한 패키지 설치
pip install -r requirements.txt


## 4. 디렉토리 구조
audio-event-infer/
├── data/
│   ├── horn.wav
│   └── call.wav
├── models/
│   ├── finetuned/     # 파인튜닝 모델
│   └── pretrained/    # 기존 pretrained 모델
├── predict.py
├── requirements.txt
└── README.md

## 5. 모델 다운로드
허깅페이스에서 원본 모델과 파인튜닝 모델을 다운로드 합니다.

git lfs install

git clone https://huggingface.co/superb/wav2vec2-base-superb-ks models/pretrained

git clone https://huggingface.co/dbif/wav2vec2-audio-event-tunetest models/finetuned

이후 models/finetuned 폴더 안에 config.json, model.safetensors, preprocessor_config.json파일들이 있어야 합니다.

## 6. 실행 방법
python3 predict.py
