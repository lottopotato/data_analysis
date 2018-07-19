## 데이터 분석 ##

### 개요 ###
- 특정 row 데이터(파형) 에 대한 여러가지 분석기법을 사용해봅니다.

### 매인 파일 ###
- main.py

### 사용언어 및 프레임워크 ###
- python 3.6.4
- matplotlib : 그래프 등 도표
- sklearn : 클러스터링
- tensorflow : 머신러닝

## 0718. 현재 진행사항 ##
1. 오리지날 데이터(평균, 분산, 표준편차) 그래프 
2. K-means 클러스터링
3. Hierarchical-agglomerative 클러스터링
4. 비지도학습 auto-encoder
5. 비지도학습 Generative Adversarial Networks
6. 지도학습 Deep Neural Networks

## 정리 ##
1. 클러스터는 T01 과 T06 데이터에서 좀더 잘 동작함.
2. 비지도 학습 GAN 은 현재 의미 있는 결과를 볼수없음.
3. 지도학습에서 hierarchical city block distance 로 레이블링

## 분석 기법 ##
#### K - means clustering ####
표준편차 값을 입력으로 데이터를 군집화 합니다.

참고 : http://scikitlearn.org/stable/modules/generated/sklearn.cluster.KMeans.html

#### Hierarchical(agglomerative) clustering ####
파형 그대로 각 파형간 거리를 통해 군집화 합니다.
파형간 코사인, 유클리디안, 멘하튼 거리 계산법을 사용합니다.
참고: http://scikitlearn.org/stable/auto_examples/cluster/plot_digits_linkage.html#sphx-glr-auto-examples-cluster-plot-digits-linkage-py

#### simple auto-encoder ####
파형의 대한 피쳐를 줄입니다.
여기서 사용 목적은 각각 다른 틱을 동기화 하려는 것입니다.
각 부품당 틱을 일치화 시킴으로써 이후 머신러닝 네트워크에서 
미니배치학습이나 유연한 네트워크 구조를 사용할수 있을것입니다.
참고 : https://github.com/floydhub/tensorflow-examples/blob/master/3_NeuralNetworks/autoencoder.py

#### simple Generative Adversarial Network ####
파형을 학습하고 새로운 파형을 생성하는 비지도 학습입니다.
현재 잘 작동 하지않으며 원인으로는 투토리얼 수준의 네트워크와
그로 인한 매우 느린 학습률, 또한 파형의 정규화 기법의 부재로 예상됩니다.

*디멘션이나 피쳐를 압축하는 전처리 과정과 최적화된 정규화, 충분한 시간이
있으면 조금더 잘 동작하지 않을까 예상합니다.

참고 : https://github.com/golbin/TensorFlow-Tutorials/blob/master/09%20-%20GAN/01%20-%20GAN.py
https://github.com/carpedm20/DCGAN-tensorflow

#### Deep Neural Network ####
클러스터링으로 레이블을 추출하며 추출된 레이블을 통해
지도 학습을 수행해봅니다.
파손과 정상의 차이가 눈에 보일정도로 뚜렷한 경우 좀더 적은 반복으로 
파손 / 정상 / 마모 를 분류합니다.
이때 파손이라고 생각할 수 있는 파형을 n배 증가시킨 가짜 파형을 생성하여
학습합니다.
## 클러스터링 ##
#### kmeans ####
![1]()

![2]()

![3]()

![4]()

![5]()

![6]()