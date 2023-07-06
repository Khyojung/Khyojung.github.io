---
layout: post
title: Dimensionality Reduction
description: >
    Dimensionality Reduction
    내용과 그림은 핸즈온머신러닝 책을 참고하였습니다.
sitemap: false
hide_last_modified: true
---

## Curse of dimensionality
- 차원의 저주
- 많은 머신러닝 문제들은 훈련 샘플이 수백만 개의 특성을 가지고 있다. -> 훈련을 느리게 하고 좋은 솔루션을 찾는 것을 어렵게 한다.
- 훈련 세트의 차원이 커질수록 과대적합 위험이 커진다.
- 차원이 커질 수록 필요한 데이터의 양이 기하급수적으로 커진다.
- 같은 데이터지만 1차원에서는 데이터 밀도가 촘촘했던것이 차원이 커질수록 점점 데이터간의 거리가 멀어진다.


- 이러한 차원의 저주 문제를 해결하기 위한 이론적인 해결책은 훈련 샘플의 밀도가 충분히 높아질 때까지 데이터를 모아서 훈련 세트의 크기를 키우는 것.
- 하지만 아쉽게도 **일정 밀도에 도달하기 위해 필요한 훈련 샘플 수는 차원의 수가 커짐에 따라 기하급수적으로 늘어난다.**
    - 예를 들어 크기가 1인 2차원 평면에 0.1 거리 이내에 훈련 샘플을 모두 놓으려면 최소한 10 x 10 개의 샘플이 필요하다. 
    - 이를 100개의 차원으로 확장하면 10^100개의 훈련 샘플이 필요하다. (현실적으로 불가능함)

## 차원을 감소시키는 두 가지 접근법
## 1. 투영 
- 고차원 공간에 있는 훈련 샘플을 저차원 공간으로 그대로 옮기는 것이다.
- 모든 훈련 샘플이 고차원 공간 안의 저차원 부분공간에 놓여있다.
- 예시 : 그림에는 원모양을 띈 3차원 데이터 셋이있다.
<img src="/Users/khj/Desktop/blog/Khyojung.github.io/assets/img/blog/dimensionality_reduction/IMG_20925B427AFA-1.jpeg" width = "50%" height = "50%">
- 모든 훈련 샘플이 거의 평면 형태로 놓여있고, 여기에서 모든 훈련 샘플을 이 부분공간에 수직으로 투영하면 밑의 사진과 같은 데이터셋을 얻을 수 있다.
<img src="/Users/khj/Desktop/blog/Khyojung.github.io/assets/img/blog/dimensionality_reduction/IMG_556BE05C4340-1.jpeg" width = "50%" height = "50%">

- 하지만 스위스롤 같은 데이터가 있을 경우에 투영 기법이 항상 통하는 것은 아니다.
<img src="/Users/khj/Desktop/blog/Khyojung.github.io/assets/img/blog/dimensionality_reduction/1.png" width = "50%" height = "50%">
<img src="/Users/khj/Desktop/blog/Khyojung.github.io/assets/img/blog/dimensionality_reduction/22.png" width = "50%" height = "50%">

- 왼쪽 사진이 스위스롤 데이터를 그냥 투영기법을 사용해서 투영했을 때의 모습이다. 
- 고차원 공간에서 뒤틀리거나 휘어진 2D 모양의 데이터셋을 매니폴드라고 부른다.

## 2. 매니폴드 학습 (manifold learning)
- d차원 매니폴드 : d차원 초평면으로 보일수있는 n차원 공간의 일부(d<n)
- 많은 차원 축소 알고리즘이 훈련 샘플이 놓여있는 매니폴드를 모델링하는 식으로 작동 -> 매니폴드 학습 (manifold learning)
- 매니폴드 학습은 비선형 차원축소법으로 위의 스위스롤처럼 고차원의 꼬여있는 데이터 분포에서 매니폴드를 모델링하는 것이다.

### 매니폴드 학습의 가정
    1. 실제 고차원 데이터 셋이 더 낮은 저차원 매니폴드에 가깝게 놓여 있다.
    2. 처리해야할 작업이 저차원의 매니폴드 공간에 표현되면 더 간단해진다.
    
**매니폴드 학습의 한계**
<img src="/Users/khj/Desktop/blog/Khyojung.github.io/assets/img/blog/dimensionality_reduction/2차원.png" width = "50%" height = "50%">

- 위의 데이터 셋에는 3차원에서는 경계를 나누기 어렵지만 2차원에서는 뚜렷한 경계를 볼 수 있다.

<img src="/Users/khj/Desktop/blog/Khyojung.github.io/assets/img/blog/dimensionality_reduction/3차원.png" width = "50%" height = "50%">

- 하지만 해당 데이터셋에서는 차원 축소를 한 경우 오히려 더 경계를 구분하기 어려운 것을 알 수 있다. 

## 3. 주성분 분석 (principal component analysis, PCA)
- 데이터에 가장 가까운 초평명(hyperplane)을 정의한 다음, 데이터를 이 평면에 투영시킨다.
- 저차원의 초평면에 훈련 세트를 투영하기 전에 먼저 올바른 초평면을 선택해야한다.


<img src="/Users/khj/Desktop/blog/Khyojung.github.io/assets/img/blog/dimensionality_reduction/IMG_99EA0CE5C262-1.jpeg" width = "50%" height = "50%">

- 다른 방향으로 투영하는 것보다 분산이 최대로 보존되는 축을 선택하는 것이 정보가 가장 적게 손실된다. 
- ***원본 데이터셋과 투영된 것 사이의 평균 제곱 거리를 최소화하는 축***

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X2D - pca.fit_transform(X)
```

- PCA모델을 사용해 데이터의 차원을 2로 줄이는 코드
    - 사이킷런의 PCA모델은 자동으로 데이터를 중앙에 맞춰준다.

**적절한 차원의 수 선택하기**
- pca.explained_variance_ratio_ : 변수에 저장된 주성분의 설명된 분산 비율 ( 각 주성분의 축을 따라 있는 데이터 셋의 분산 비율을 나타냄 )

~~~python
from sklearn.decomposition import PCA

pca = PCA()
pca.fit(X_train)
cunsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 2
~~~
- n_components=d로 설정하여 PCA를 다시 실행
- 유지하려는 주성분의 수를 지정하기 보다는 보존하려는 분산의 비율을 n_components에 0.0에서 1.0사이로 설정하는것이 좋음

**PCA 기반 차원 감소의 문제점**

- PCA의 경우 선형 분석 방식으로 값을 사상하기 때문에 차원이 감소되면서 군집화 되어 있는 데이타들이 뭉게져서 제대로 구별할 수 없는 문제를 가지고 있음

**점진적 PCA**

- PCA 구현의 문제는 특이값 분해(Singular Value Decomposition, SVD)알고리즘을 실행하기 위해 전체 훈련세트를 메모리에 올려야 한다는 것
    - 이를 해결하기 위해 점진적 PCA (incremental PCA, IPCA) 알고리즘이 개발
    - 훈련 세트를 미니 배치로 나눈 뒤 IPCA 알고리즘에 한 번에 하나씩 주입

- IPCA는 특정 순간에 배열의 일부만 사용하기 때문에 메모리 부족 문제를 해결할 수 있다.

**지역 선형 임베딩 (locally linear embedding, LLE)**

- 투영에 의존하지 않는 매니폴드 학습
    1. 먼저 각 훈련 샘플이 가장 가까운 이웃에 얼마나 선형적으로 연관되어있는지 측정
    2. 국부적인 관계가 가장 잘 보존되는 훈련 세트의 저차원 표현을 찾음
- 잡음이 많지 않은 경우 꼬인 매니폴드를 펼치는데 잘 작동

<img src="/Users/khj/Desktop/blog/Khyojung.github.io/assets/img/blog/dimensionality_reduction/IMG_58C249A35675-1.jpeg" width = "50%" height = "50%">

## 다른 차원 축소 기법

1. 랜덤 투영 (random projection)
2. 다차원 스케일링 (multidimensional scaling, MDS)
3. Isomap
    - 각 샘플을 가장 가까운 이웃과 연결하는 식으로 그래프를 생성
    - 지오데식 거리를 유지하면서 차원을 축소
4. t-SNE (t-distributed stochastic neighbor embedding)
    - 비슷한 샘플은 가까이, 비슷하지 않은 샘플은 멀리 떨어지도록 하면서 차원을 축소
    - 주로 시각화에 많이 사용
    - 특히 고차원 공간에 있는 샘플의 군집을 시각화 할때 사용
    


```python
import warnings
warnings.filterwarnings('ignore')

from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pandas as pd
iris = datasets.load_iris()

labels = pd.DataFrame(iris.target)
labels.columns=['labels']
data = pd.DataFrame(iris.data,columns=['Sepal length','Sepal width','Petal length','Petal width'])

fig = plt.figure( figsize=(6,6))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
ax.scatter(data['Sepal length'],data['Sepal width'],data['Petal length'],c=labels,alpha=0.5)
ax.set_xlabel('Sepal lenth')
ax.set_ylabel('Sepal width')
ax.set_zlabel('Petal length')
plt.show()
```


![png](/Users/khj/Desktop/blog/Khyojung.github.io/assets/img/blog/dimensionality_reduction/output_9_0.png)



```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

scaler = StandardScaler()

pca = PCA()

pipeline = make_pipeline(scaler,pca)

pipeline.fit(data)

features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.show()
```


![png](/Users/khj/Desktop/blog/Khyojung.github.io/assets/img/blog/dimensionality_reduction/output_10_0.png)



```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

model = PCA(n_components=2)
pca_features = model.fit_transform(data)

xf = pca_features[:,0]
yf = pca_features[:,1]

test = labels['labels'].values.tolist()
plt.scatter(xf,yf,c = test)
plt.show()
```


![png](/Users/khj/Desktop/blog/Khyojung.github.io/assets/img/blog/dimensionality_reduction/output_11_0.png)



```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

model = PCA(n_components=1)
pca_features = model.fit_transform(data)

xf = pca_features[:,0]
yf = len(xf)*[0]
plt.scatter(xf,yf,c=test)
plt.show()
```


![png](/Users/khj/Desktop/blog/Khyojung.github.io/assets/img/blog/dimensionality_reduction/output_12_0.png)



```python
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
import seaborn as sns

# MNIST 데이터 불러오기
data = load_digits()

model = TSNE(2)
features_tsne = model.fit_transform(data.data)

xs = features_tsne[:,0]
ys = features_tsne[:,1]

palette = sns.color_palette("bright", 10)
sns.scatterplot(x = xs, y = ys, hue=data.target, legend='full', palette=palette)
plt.show()
```


![png](/Users/khj/Desktop/blog/Khyojung.github.io/assets/img/blog/dimensionality_reduction/output_13_0.png)



```python
digits_model_pca = PCA(2)
digits_model_pca_features = digits_model_pca.fit_transform(data.data)

xs = digits_model_pca_features[:,0]
ys = digits_model_pca_features[:,1]

palette = sns.color_palette("bright", 10)
sns.scatterplot(x = xs, y = ys, hue=data.target, legend='full', palette=palette)
plt.show()
```


![png](/Users/khj/Desktop/blog/Khyojung.github.io/assets/img/blog/dimensionality_reduction/output_14_0.png)
