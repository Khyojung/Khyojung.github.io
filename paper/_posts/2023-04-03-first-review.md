---
layout: post
title: Generative Cooperative Learning for Unsupervised Video Anomaly Detection review
description: >
    Generative Cooperative Learning for Unsupervised Video Anomaly Detection 리뷰
sitemap: false
hide_last_modified: true
---
Generative Cooperative Learning for Unsupervised Video Anomaly Detection 리뷰

this paper accepted Computer Vision and Pattern Recognition Conference (CVPR) 2022

[Abstract]
Video의 Anomaly Detection은 weakly-supervised와 one-class classification (OCC)에서 잘 관찰된다.
그러나 Unsupervised Anomaly Detection은 매우 드물다 -> anomalies는 자주 발생하지 않고 일반적으로 정의되지 않기 때문이다.
= ground truth supervision과 결합될 때 학습 알고리즘의 성능에 부정적인 영향을 미칠 수 있다.

이 문제를 해결하기 위해 저자들은 Unsupervised Generative Cooperative Learning (GCL) 접근 방식을 제안한다.
-> Generator와 Discriminator 사이의 cross-supervision을 구축하기 위해 anomalies의 low frequency를 이용하는 방식이다.

[Introduction]
In the real world에서 Learning-based Anomaly Detection은 사건의 드문 발생률 때문에 매우 어렵다.
이러한 도전들은 사건들의 제역없는 특성으로 인해 더욱 악화된다.
-> 학습을 다루기 위해 anomalies는 종종 normal 데이터와의 편차로 여겨진다.
-> Anomaly Detection의 인기있는 접근 방식은 normal 훈련 예제만을 사용하여 dominant data representation을 배우는 one-class classifier를 훈련시키는 것이다.

but one-class classification (OCC)의 단점
1) 모든 normalcy variation을 capturing하지 않는다. (제한된 가용성)
2) OCC접근 방식은 여러 클래스와 동적 상황의 복잡한 문제에 적합하지 않다.
=> unseen normal activity는 anomalous한 결과를 초래할 수 있다.

Recently
weakly supervised anomaly detection methods는 video level label을 이용하여 미세한 annotation을 얻는 비용을 줄이는데 상당히 많이 사용되었다.
일부 콘텐츠가 anomalous하고 모든 데이터가 normal 하면 전체 video는 anomalous하게 표시된다. <- 수동검사 필요
-> 상대적으로 비용 효율적이지만, 많은 실제 응용 프로그램에서는 비실용적이다.

annotation cost가 발생하지 않으면 anomaly detection training에 사용할 수 있는 수많은 video 영상이 있다.
but 현재까지는 video anomaly detection을 위해 label이 지정되지 않은 train data를 활용하는 시도는 없다.

In this work,
unsupervised mode for video anomaly detection <- weakly or one-class supervision보다 좀 더 잘 하기 위해 탐구한다.
- 이 문헌에서 'unsupervised'의 용어는 정상적인 train data를 가정하는 OCC 접근 방식을 의미한다.

video에서 unsupervised anomaly detection에 접근할 때, video가 정지 image에 비해 정보가 풍부하고 anomalous한 현상이 적게 일어난다는 사실을 이용하고, 구조화된 방식으로 그러한 도메인 지식을 활용하려고 시도한다.

이를 위해, Generative Cooperative Learning (GCL) 방식을 제안 =  label이 지정되지 않은 video를 입력으로 받아들이고 frame 수준의 anomaly score를 출력으로 예측하는 방법을 배우는 생성 협동 학습
1) Generator와 Discriminator로 이루어져있다.
2) anomaly detection을 위해 상호 협력적(mutually cooperative)방식으로 훈련받는다.

Generator는 normal 반복을 reconstruction(재구성)할 뿐만 아니라 새로운 negative Learning을 사용하여 가능한 high-confidence anomalous representation을 왜곡한다.
Discriminator는 instance가 anomaly일 확률을 추정한다.

Unsupervised anomaly detection을 위해 Generator에서 pseudo-label을 만들고 이를 사용하여 Discriminator를 훈련시킨다.

다음 단계에서 훈련된 Discriminator버전에서 pseudo-label을 만든다음 이를 사용하여 Generator를 개선한다.

Contributions
label이 부착된 train data 없이 복잡한 시나리오에서 anomalous event을 찾아낼 수 있는 anomaly detection 방법을 제안한다.
-> fully unsupervised mode의 video anomaly detection의 첫번째 시도이다.

사용하는 데이터 셋 = UCF-Crime, ShanghaiTech
