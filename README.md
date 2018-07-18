## 데이터 분석 ##

### 개요 ###
- 특정 row 데이터(파형) 에 대한 여러가지 분석기법을 사용해봅니다.

### 매인 파일 ###
- main.py

### 사용할 언어 및 프레임워크 ###
- python 3.6.4
- matplotlib
- sklearn
- tensorflow

### 0718. 현재 진행사항 ###
1. 오리지날 데이터(평균, 분산, 표준편차) 그래프 
2. K-means 클러스터링
3. Hierarchical-agglomerative 클러스터링
4. 비지도학습 auto-encoder
5. 비지도학습 Generative Adversarial Networks
6. 지도학습 Deep Neural Networks


### 정리 ###
1. 클러스터는 T01 과 T06 데이터에서 좀더 잘 동작함.
2. 비지도 학습 GAN 은 현재 의미 있는 결과를 볼수없음.
3. 지도학습에서 hierarchical city block distance 로 레이블링

### 클러스터링 ###
![1](https://github.com/lottopotato/data_analysis/blob/test/T01_hierarchical_n_3_2018-07-17%2018_15_32.png)

![2](https://github.com/lottopotato/data_analysis/blob/test/T01_kmeans_n_3_2018-07-17%2018_13_09.png)

![3](https://github.com/lottopotato/data_analysis/blob/test/T06_hierarchical_n_3_2018-07-17%2018_35_42.png)

![4](https://github.com/lottopotato/data_analysis/blob/test/T06_kmeans_n_3_2018-07-17%2018_51_08.png)

![5](https://github.com/lottopotato/data_analysis/blob/test/T07_hierarchical_n_3_2018-07-17%2018_39_02.png)

![6](https://github.com/lottopotato/data_analysis/blob/test/T07_kmeans_n_3_2018-07-18%2010_05_13.png)