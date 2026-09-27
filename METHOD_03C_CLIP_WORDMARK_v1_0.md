# 03C CLIP Wordmark 비교실험 v1.0

## 목적
03B FontCLIP에서 계산적 Text–Typeface congruence가 확인되지 않았기 때문에,
FontCLIP을 결과가 나올 때까지 조정하는 대신 **동일한 wordmark 자극을 범용 CLIP으로 평가**하여
결과의 모델 의존성을 확인한다.

## 결과 이전에 고정한 조건
- 이미지: 03B와 동일한 1024×1024 흰 배경·검정 영문 wordmark
- CLIP: `openai/clip-vit-base-patch32`
- Aaker: 42 traits → 15 facets → 5 dimensions
- prompt: `{trait} font`
- negative prompt 없음
- primary reference: A only
- A+B: 민감도
- Text comparison: 사전에 실행한 NLI MiniLM과 mDeBERTa 둘 다 비교
- profile similarity: Pearson correlation을 중심 진단
- same-brand vs permuted-brand comparison: 10,000 permutations 기본
- 결과 확인 후 prompt/model을 변경하지 않음

## 핵심 비교
### 1. CLIP ↔ FontCLIP
동일 이미지와 동일 42 trait coordinate에서 두 시각모델의 결과가 얼마나 유사한지 확인한다.

### 2. Text NLI ↔ CLIP
MiniLM, mDeBERTa 두 text profile 각각과 CLIP wordmark profile을 비교한다.

### 3. Reference-font sensitivity
동일 브랜드명을 DejaVu Sans로 렌더링한 control과 실제 wordmark를 비교한다.
DejaVu Sans를 심리적 'neutral font'로 주장하지 않고 **reference font**로만 사용한다.

## 해석
- CLIP에서만 일치가 나타나면, 'CLIP이 맞고 FontCLIP이 틀렸다'고 단정하지 않는다.
- 두 모델의 시각적 의미공간 차이와 측정모델 의존성으로 해석한다.
- FontCLIP과 CLIP 모두 일치가 없으면 wordmark-only 수준에서 계산적 congruence가 약한 결과로 본다.
- 이후 full-logo CLIP은 색상·심볼·형태를 포함하는 별도 03D 분석으로 구분한다.
