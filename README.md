# 03C CLIP Wordmark Analyzer v1.0

## 실행
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 첫 실행
1. `logo_dataset_68_fixed.zip` 업로드
2. Reference pool: `A only`
3. Permutation: `10000`
4. Reference-font sensitivity: 체크 유지
5. `03C CLIP-wordmark 분석 실행`
6. 결과 ZIP 다운로드 후 ChatGPT에 업로드

첫 실행 시 Hugging Face에서 `openai/clip-vit-base-patch32`를 다운로드합니다.

## 주의
이 앱은 CLIP 결과가 더 잘 나오도록 prompt를 튜닝하는 도구가 아닙니다.
03B FontCLIP과 동일한 조건에서 범용 CLIP의 결과를 비교하는 고정 실험입니다.
