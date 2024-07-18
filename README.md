# DiscoverVoice

## 프로젝트 배경
음악은 사람들이 친구나 가족과 함께 즐기며 자신의 감정을 표현하고 공유할 수 있는 방법이다. 음악은 각기 다른 장르와 스타일을 가지고 있기 때문에 사람들은 자신의 취향에 맞는 곡을 찾기 위해 노력한다. 본 프로젝트는 개인의 음색과 기호를 반영한 AI 기반 노래 추천 및 합성 시스템을 개발하는 것은 음악을 개인 맞춤형으로 제공하고, 사용자들이 더욱 만족할 수 있는 음악 청취 환경을 제공할 수 있다. 사용자의 음색을 분석하여 그와 어울리는 가수를 추천하고, 더 나아가 사용자의 목소리로 합성된 음악을 제공하는 것을 목표로 한다.

## 데이터셋
1.	MUSDB18-HQ
https://zenodo.org/records/3338373
-	음원 분리를 위한 모델 학습 및 평가를 위한 
-	총 150개의 분리된 오디오 데이터셋으로 drum, bass, vocal, 기타 악기로 분리되어 있음.
2.	YouTube Music 국내 음원
-	총 2909개의 오디오 데이터셋
-	남성 50명, 10cm, GOT7, MC.THE.MAX, 강다니엘, 규현, 김동률, 김범수, 김필, 딘, 마크튭, 먼데이키즈, 멜로망스, 문문, 박원, 박재범, 박효신, 백현, 비, 산들, 성시경, 송민호, 수호, 양다일, 양요섭, 원슈타인, 윤딴딴, 윤종신, 이무진, 이병찬, 이승윤, 이승철, 이승철, 이찬원, 임창정, 자이언티, 잔나비, 장기하, 장범준, 적재, 정승환, 카더가든, 카더가든, 케이윌, 크러쉬, 태일, 폴킴, 하동균, 하현우, 한동근
-	여성 가수 36명
IOI, 권진아, 김나영, 다비치, 레드벨벳, 뉴진스, 마마무, 박보람, 박혜원, 백예린, 백지영, 벤, 블랙핑크, 비비, 빅마마, 선미, 선우정아, 솔지, 아이브, 아이유, 알리, 에스파, 에일리, 이선희, 이소라, 이하이, 정은지, 제시, 조이, 최예나, 케이시, 케플러, 태연, 펀치, 헤이즈, 화사, 휘인
3.	다음색 가이드 보컬 데이터
https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=473
-	한국어 wav 파일 Singer 18, 45, 48를 사용

## 사용 모델
1.	HTDemucs
2.	mdx23c
3.	AST
4.	RVC

## 평가
-	오디오 분리 모델 평가
SDR (Signal to Distortion Ratio):
SDR은 원본 신호와 복원된 신호 간의 차이를 측정하는 지표이다. 높은 SDR 값은 복원된 신호와 원본 신호의 유사성이 높음을 의미한다. 이는 신호가 왜곡 없이 원래의 형태를 잘 유지하고 있음을 나타낸다. 예를 들어, 오디오 분리 모델이 높은 SDR을 기록하면, 이는 원본 오디오 트랙에서 잡음이나 기타 원치 않는 신호를 효과적으로 제거하고, 원하는 신호를 잘 복원했음을 나타낸다. 이 지표는 특히 음악 트랙의 보컬 분리나 악기 분리 작업에서 매우 중요한 역할을 한다.
-	오디오 분류 모델 평가
Cross Entropy Loss:
교차 엔트로피 손실은 모델의 예측 확률 분포와 실제 레이블 분포 간의 차이를 측정하는 지표이다. 이 지표는 분류 모델의 성능을 평가하는 데 널리 사용된다. 낮은 교차 엔트로피 손실 값은 모델의 예측이 실제 레이블과 가까움을 의미한다. 
-	오디오 합성 모델 평가
MEL-Cepstral-Distortion (MCD):
멜-케프스트럴 왜곡(MCD)은 음성 신호 처리에서 원래의 음성과 생성된 음성 간의 유사성을 측정하는 데 사용되는 지표이다. 이 지표는 두 음성 신호의 MFCC(Mel-Frequency Cepstral Coefficients) 간의 차이를 계산한다. 낮은 MCD 값은 생성된 음성이 원래의 음성과 매우 유사함을 의미한다.
UTMOS (Universal Text-to-Speech Mean Opinion Score Predictor):
The singing voice conversion challenge 2023에서 채택한 neural MOS Predictor인 UTMOS를 사용하였다. UTMOS는 생성된 음성의 품질을 객관적으로 평가하기 위해 사용되는 도구로, 인간의 청취 평가를 모방하여 음성의 자연스러움과 품질을 점수화한다. 우리의 오디오 합성 모델은 UTMOS 평가에서 높은 점수를 기록하였으며, 이는 모델이 생성한 음성이 매우 자연스럽고 실제 음성에 가깝다는 것을 의미한다.
Human MOS (Mean Opinion Score):
설문 조사(https://tally.so/r/3XGxyL)를 통해 Human MOS를 평가하였다. Human MOS는 실제 청취자들이 생성된 음성을 평가하여 점수를 부여하는 방식으로, 음성의 자연스러움, 품질, 이해 가능성 등을 종합적으로 평가한다. 우리는 다양한 연령대와 성별의 청취자 40여명을 대상으로 설문 조사를 실시하였다.

## 결과
이번 프로젝트에서는 다양한 모델 및 기법을 활용하여 오디오 데이터의 분류, 합성, 및 분리 작업을 수행하였다. 특히, Pruning(가지치기) 기법을 통해 불필요하고 중요도가 낮은 파라미터를 제거하여 모델의 크기 및 계산 비용을 감소시켰다. Iterative하게 pruning을 수행함으로써 모델의 복잡도를 낮추고, overfitting을 방지하였다. 이로 인해 Signal to Distortion Ratio(SDR)이 증가하였으며, MDX23C과 HtDemucs 모델의 성능은 각각 21.37%, 9% 향상되었다. SDR 값이 9.648 및 12.388로 높아, 음질이 우수하고 음성 인식, 합성 및 분리 작업에 매우 유용한 수치를 기록하였다. 또한, 앙상블 기법을 활용하여 vocal을 효과적으로 분리하였고, 전체 시스템의 신뢰성을 강화하며 다양한 feature를 활용할 수 있었다.
Vision Transformer(ViT)를 활용한 AST 모델은 Audio Classification에 활용되어 사용자 목소리를 분류하고, 어떤 가수와 가장 가까운 목소리인지를 확률적으로 계산하였다. 이 Transformer 기반의 분류 모델은 추천 시스템에 활용 가능하며, 사용자 맞춤형 음악 추천의 정확성을 높일 수 있다.
CNN 및 Diffusion 모델의 학습에는 많은 양의 데이터와 긴 학습 시간이 필요하지만, HuBERT를 통해 이러한 문제를 극복할 수 있었다. Feature를 일부 Masking함으로써 오디오 생성에 중요한 Hidden unit을 모델이 스스로 학습하도록 하였으며, 이를 통해 음성의 효율적 표현을 학습하였다. Diffusion 모델 대비 적은 양의 데이터로도 학습이 가능하며, 약 10분 내외의 데이터로 모델을 구현할 수 있었다. 학습 시간도 대폭 단축되어 1만 스텝을 약 1시간 내외로 학습할 수 있었고, 적은 양의 사용자 목소리를 수집하고 빠르게 모델을 생성하여 사용자 경험을 증대시킬 수 있었다.


![image](https://github.com/user-attachments/assets/12256b30-3ba2-45c8-a4da-03697b6989be)
<p align=center>MCD 결과</p>

![image](https://github.com/user-attachments/assets/034052fa-caf1-4c1d-b116-59907430f30d)
<p align=center>MCD 유사도</p>

![image](https://github.com/user-attachments/assets/32465376-0148-4ccd-adc6-534cb2701969)
<p align=center>Neural MOS Prediction</p>

![image](https://github.com/user-attachments/assets/b488bf56-ca95-4312-b80d-8a35be4253f9)
<p align=center>Human MOS - Singer 18</p>

![image](https://github.com/user-attachments/assets/08932ac2-963a-4d94-b363-d090a6ac47bb)
<p align=center>Human MOS - Singer 45</p>

![image](https://github.com/user-attachments/assets/3d6053dc-fe2e-4857-b474-ef614d485ec8)
<p align=center>Human MOS - Singer 48</p>
