  # DiscoverVoice

  ## 프로젝트 배경
  음악은 사람들이 친구나 가족과 함께 즐기며 자신의 감정을 표현하고 공유할 수 있는 방법입니다. 각기 다른 장르와 스타일로 구성된 음악은 개인의 취향에 맞는 곡을 찾는 여정을 돕습니다. 본 프로젝트는 AI 기반의 노래 추천 및 합성 시스템을 통해 개인 맞춤형 음악 경험을 제공하고, 사용자들이 더욱 만족할 수 있는 음악 청취 환경을 만드는 것을 목표로 합니다. 사용자의 음색을 분석하여 어울리는 가수를 추천하고, 사용자의 목소리로 합성된 음악을 제공하는 것을 목표로 합니다.

  ## 데이터셋
  1. **MUSDB18-HQ**: [Link](https://zenodo.org/records/3338373)
      - 음원 분리를 위한 모델 학습 및 평가용
      - 총 150개의 분리된 오디오 데이터셋 (드럼, 베이스, 보컬, 기타 악기로 분리)

  2. **YouTube Music 국내 음원**
      - 총 2909개의 오디오 데이터셋
      - 남성 가수: 50명 (예: 10cm, GOT7, 강다니엘 등)
      - 여성 가수: 36명 (예: IOI, 레드벨벳, 마마무 등)

  3. **다음색 가이드 보컬 데이터**: [Link](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=473)
      - 한국어 wav 파일 사용 (Singer 18, 45, 48)

  ## 사용 모델
  1. HTDemucs
  2. mdx23c
  3. AST
  4. RVC

  ## 평가
  - **오디오 분리 모델 평가**
      - **SDR (Signal to Distortion Ratio)**: 원본 신호와 복원된 신호 간의 차이를 측정하는 지표로, 높은 SDR 값은 신호가 왜곡 없이 원래의 형태를 잘 유지함을 의미합니다.

  - **오디오 분류 모델 평가**
      - **Cross Entropy Loss**: 모델의 예측 확률 분포와 실제 레이블 분포 간의 차이를 측정하는 지표로, 낮은 값은 모델의 예측이 실제 레이블과 가까움을 의미합니다.

  - **오디오 합성 모델 평가**
      - **MEL-Cepstral-Distortion (MCD)**: 음성 신호 처리에서 원래 음성과 생성된 음성 간의 유사성을 측정하는 지표로, 낮은 MCD 값은 생성된 음성이 원래 음성과 매우 유사함을 의미합니다.
      - **UTMOS (Universal Text-to-Speech Mean Opinion Score Predictor)**: 생성된 음성의 품질을 객관적으로 평가하기 위해 사용되는 도구로, 높은 점수는 모델이 생성한 음성이 매우 자연스럽고 실제 음성에 가깝다는 것을 의미합니다.
      - **Human MOS (Mean Opinion Score)**: 실제 청취자들이 생성된 음성을 평가하여 점수를 부여하는 방식으로, 음성의 자연스러움, 품질, 이해 가능성 등을 종합적으로 평가합니다.

  ## 결과
  본 프로젝트는 다양한 모델 및 기법을 활용하여 오디오 데이터의 분류, 합성, 분리 작업을 수행했습니다. Pruning 기법을 통해 불필요한 파라미터를 제거하여 모델의 크기 및 계산 비용을 감소시켰고, Iterative Pruning을 통해 모델의 복잡도를 낮추고 overfitting을 방지했습니다.

  - **SDR 향상**: MDX23C 모델 성능 21.37%, HTDemucs 모델 성능 9% 향상 (각각 SDR 값 9.648 및 12.388)
  - **앙상블 기법**: Vocal을 효과적으로 분리하고 시스템 신뢰성 강화
  - **AST 모델**: Audio Classification에 활용되어 사용자 목소리를 분류하고 추천 시스템의 정확성을 높임
  - **HuBERT 사용**: 적은 양의 데이터로도 학습 가능, 약 10분 내외의 데이터로 모델 구현, 학습 시간 대폭 단축 (1만 스텝을 약 1시간 내외로 학습)

  ![MCD 결과](https://github.com/user-attachments/assets/12256b30-3ba2-45c8-a4da-03697b6989be)
  <p align="center">MCD 결과</p>

  ![MCD 유사도](https://github.com/user-attachments/assets/034052fa-caf1-4c1d-b116-59907430f30d)
  <p align="center">MCD 유사도</p>

  ![Neural MOS Prediction](https://github.com/user-attachments/assets/32465376-0148-4ccd-adc6-534cb2701969)
  <p align="center">Neural MOS Prediction</p>

  ![Human MOS - Singer 18](https://github.com/user-attachments/assets/b488bf56-ca95-4312-b80d-8a35be4253f9)
  <p align="center">Human MOS - Singer 18</p>

  ![Human MOS - Singer 45](https://github.com/user-attachments/assets/08932ac2-963a-4d94-b363-d090a6ac47bb)
  <p align="center">Human MOS - Singer 45</p>

  ![Human MOS - Singer 48](https://github.com/user-attachments/assets/3d6053dc-fe2e-4857-b474-ef614d485ec8)
  <p align="center">Human MOS - Singer 48</p>

  평가 결과, Average MEL-Cepstral-Distortion(MCD)이 3점 내외로 나타났으며, MCD 기반 유사성 평가에서 약 95% 정도의 일치성을 보였습니다. Neural MOS Prediction은 1.5 ~ 1.8점을 기록하였고, Human MOS 평가에서는 세 모델 모두 3.7점 내외를 기록하였습니다. 이는 일부 결함이 있지만 허용 가능한 수준으로 평가되었습니다.
