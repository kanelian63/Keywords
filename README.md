# Load_map

# Lecture 
딥러닝 홀로서기 - 정주행 1번 완료. 코드구현은 하지 않았음.

테크보이 코딩 배우기

https://www.youtube.com/watch?v=M6kQTpIqpLs&list=PLa7Lj786Q-Gts3-LsBl5I56YQrQb4sHxI


argparse

모듈리스트 시퀀셜

파이토치 모델 하나

우분투

파이토치

다커

깃허브

파이썬

시각화 

웹크롤링

L1

L2 loss

선형대수, 미분방정식, 회귀분석

Value function 이용해서 Bellman equation 푸는 문제

그레인저 인과관계

게임이론

2nd price auction

계량경제학

R

딥러닝이라는게 Neural Network가 중층으로 연결된 구조라는 것, 그 구조 자체가 Regression을 반복적으로 실행하면서 데이터 전처리를 모델 내부에서 해결하는 작업

Autoencoder

Kalman filter

Bayesian 통계학

Logistic regression + Ensemble = Neural net 이라는 개념 이해, Activation function이 사실은 Kernel이라는 이해

Q-Q plot

데이터 안에 비선형의 강한 패턴이 반복될 때만 신경망 모델이 유의미하다는 것을 이해

Markov process와 Markov Decision process의 차이점은 나의 행동이 주변 환경을 변화시키는지 여부

딥러닝 모델을 웹에 올리기

5페이지부터 다 읽어야 돼

https://blog.pabii.co.kr/category/%eb%8d%b0%ec%9d%b4%ed%84%b0%ec%82%ac%ec%9d%b4%ec%96%b8%ec%8a%a4/page/5/



kaggle

트윗 자연어처리 참가

https://www.kaggle.com/c/nlp-getting-started/overview


Correlation 구하는 공식
Network theory
Graph Theory
Newton method
Convex Optimization
Statistical Machine Learning
Tree model
Clustering
Ridge, Lasso
데이터에 알고리즘을 적용하기 전에 먼저 정규분포를 따르는지 아닌지 확인하기 위해 Histogram과 QQ-plot을 그려봤는데요,

1. 회귀모델의 진단
Q. 다중 회귀분석 (Multinomial regression)에서는 모델링 과정에서 정규성, 선형성, 등분산성, 이상치 등을 살펴보기 위해서 Plot을 그려보던데, 머신러닝에서 Gradient descent를 할 때는 Regression에서 하던것처럼 잔차에 대한 여러 가정들을 별도로 확인해보지 않아도 되나요?

A. 수업 중 Ensemble을 다루면서 저 질문에 대한 답변을 좀 더 깊이 있게 다룬다. 통계학에서 Regression은 모델링의 결과값이 Error 최소화, 즉 Variance 최소화에 맞춰져 있다. Variance가 크게 나오면 Regression 모델에서 나온 상관계수들이 0이 아니라고 주장하는 p-value 값을 믿을 수가 없기 때문이다. 그런데, Regression을 (특히 Linear regression을) 적용하는 데이터들은 Error의 패턴이 없는 경우, 즉 데이터의 Error에 해당하는 부분이 랜덤으로 생성된 경우다. 머신러닝을 쓰는 이유는 그런 랜덤이 아닌 부분, 패턴이 있는 부분, 특히 패턴이 단순한 선형 관계 (ax+b)가 아닌 부분을 찾아내기 위해서다.

그런 비선형의 복잡한 패턴을 다항식으로 (무식하게?) 찾아내는데는 한계가 있으니까, 머신러닝의 여러가지 방법론들 (SVM, Decision Tree, Neural network 등등)을 활용하는 것이고, Gradient descent는 그런 탐정놀이 게임 중 수학식으로 쉽게 떨어지지 않는 trial-and-error 계산을 하기 위해 디자인된 계산 방법이다. 비선형 함수를 찾아낸다는 말 자체에서 이미 데이터의 non-random을 가정하고 접근하는 것이고, Gradient descent는 계산 비용을 줄이면서 결과값이 최대한 근사치가 나오도록하는 계산법이다보니, 당연히 잔차의 구조에 대한 고민을 안 하게 될 수 밖에 없다.

 

2. 데이터의 정규성 확인
Q. 데이터에 알고리즘을 적용하기 전에 먼저 정규분포를 따르는지 아닌지 확인하기 위해 Histogram과 QQ-plot을 그려봤는데요, t-test 적용에서와 같이 정규성 검정까진 할 필요가 없는 건가요? 중심극한정리에서는 표본이 충분할 때 “표본분포”가 정규분포의 형태를 따른다고 했는데, 우리가 가진 데이터는 표본 하나인 것과 같으니 단순 데이터 셋의 정규성만 확인해도 충분한 것인지 궁금합니다.

A. 우선 몇 가지 사실 관계를 점검해야할 것 같다.

t-test하는데 정규성 검정이라는 것은, 원래 데이터가 (approximately, more precisely asymptotically) 정규분포를 따르고 있는지에 대한 확인이다. 원래 데이터가 정규분포가 아니라면 학부 수업 때 배운 t-stat말고 다른 계산법이 존재한다. (수학 & 통계학 시간에 Poisson일 경우 어떻게 해야된다고 했는지 노트 참조)

또 중심극한정리 (Central Limit Theorem, CLT)는 표본이 충분하면 분포가 정규분포를 따르는게 아니라, 샘플 추출을 여러번 했을 때, 그 평균값이 정규분포를 따른다는 정리다. 쉽게 예시를 들면, 모집단을 모르고 단순하게 1,000명의 표본집단만 여러번 뽑으면 (ex. 선거 직전 여론 조사), 그 표본집단의 평균값들 (처음엔 평균이 49%, 다음엔 51%, 그 다음엔 48.5% 등등등)이 정규 분포 형태로 고르게 퍼진다는 뜻이다. (수학적으로는 정규분포에 수렴한다고 표현한다.) 데이터가 많으면 무조건 정규분포에 수렴한다고 잘못 알고 있는 사람들이 많은데 (나도 예전엔….), 그러다보니 Underlying distribution이 포아송 분포인데도 t stat을 무조건 정규분포 기반으로 계산해서 모델 테스트를 엉망으로 하고 있다고 수학 & 통계학 시간에 강조한 바 있다.

일단 위의 1번에서 설명한대로, 데이터(의 에러)가 정규분포를 따른다면 Random 이라는 뜻이니까, 단순한 Linear regression으로 충분한 모델링이 되고, 머신러닝에서 배운 여러 테크닉들은 그런 randomness가 깨질 때만 의미가 있다. 따라서 현재 가진 데이터의 정규성을 확인해보는 것만으로 충분하다.

 

3. lm(), glm() 등의 함수와 caret 패키지 사용의 차이
Q. R 코딩 실습 부분에서 저는 평소 caret 패키지를 사용하지 않았었는데요, lm(), glm(), svm() 드으이 함수를 쓰는 것과, caret 패키지에서 method = ‘lm’, ‘glm’, ‘svm’ 등을 지정하는 것은 계산 방식에서 어떤 차이가 있나요? 앞의 함수들은 Normal equation으로 해를 구하고, caret은 gradient descent 알고리즘을 이용하는 건가요?

A. 우선 Normal equation이 작동하는 유일한 영역은 데이터가 정규분포를 따르고, 그 때 Best Unbiased Estimator인 MLE가 OLS estimator와 똑같을 때 밖에 없다. 이게 통계학 수업 시간에 배우는 내용이다.

위에 나온 glm(), svm() 같은 머신러닝 모델을 간략화해놓은 함수들은 Normal equation을 쓰는게 아니라, glm은 Newton’s method를 이용한 approximation, svm은 Duality를 이용한 Group error minimization 계산을 한다.

두 계산 함수 그룹의 차이는, caret은 R 플랫폼의 함수들을 결합해서 위의 계산을 하도록 만든 패키지고, svm, xgboost등의 외부 패키지들은 외부 플랫폼 (C++, Java 등등)의 계산 모듈을 빌려온다. 왜? 계산 속도 자체만 놓고보면 시스템 자원을 효율적으로 쓰지 못하는 R이 압도적으로 느리니까. 참고로 Python, R등의 언어는 C++, Java가 시스템 자원을 활용하는 방식과는 완전히 다른 설계로 짜여져 있기 때문에, 계산 속도가 현격하게 차이날 수 밖에 없다.

Python에서도 scikit-learn을 쓰면 R의 caret과 똑같은 상황이 벌어질 것이다. 진정한 개발용 언어인 Java, C++이 1초만에 계산하는걸 R이나 Python위에서 Java, C++의 계산 모듈을 빌리면 3-4초로 늘어나고, 그냥 caret, scikit-learn 위에서 돌리면 거의 20초 정도 걸린다는 벤치마크도 있다. 데이터를 처리하고 모델링을 고민할 수 있도록 만들어 놓은 프로그램이니만큼, 복잡한 데이터 구조 (ex. n x k x p x q 행렬)를 처리하는데 초점이 맞춰져 있다보니 정작 계산 속도의 효율성 측면에서는 굉장히 나쁜 플랫폼이 되어버리는 것이다.

 

4. Polynomial regression의 경우의 수
Q. 회귀 직선이 주어진 변수만으로 적합이 안 되어서 Polynomial 형태를 적용하게 될 때, 기존 변수가 n개라면 n(n+1)/2의 경우의 수가 생기게 되잖아요. 그럼 회귀 모델 적용할 때 2차 교호작용까지 고려해서 모델을 만드는게 이런 Polynomial 형태를 고려하는 형태의 예라고 생각해도 될까요? 만약 맞다면, 그래서 2차 교호작용 외 다른 경우의 수도 고려하고 싶다면, 실제로 n(n+1)/2 횟수만큼 변수를 바꿔가며 코딩을 해야하나요?

A. 아니다. (걱정하지 마시라ㅋㅋ)

그런 Polynomial regression을 이것저것 모든 조합으로 다 해보는게 무모한 작업이라는 걸 알고 있기 때문에 svm, decision tree, ensemble, neural network 같은 대체재들이 있는 것이다. Polynomial을 쓴다는 것 자체가 이미 데이터가 선형으로 설명되지 않는다는 이해를 깔고 있고, 데이터가 랜덤이라는 가정을 깨고 들어가는 것이다. Non-linear 패턴을 잡아낸다고 했을 때, 인간이 할 수 있는 가장 단순(무식)한 방법이 Polynomial이고, 이걸 좀 더 수학적으로 세련되게 (학문적으로는 Elegant라고 표현함) 만드는 모델이 바로 여러분들이 열심히 머신러닝 테크닉이라고 배우는 svm, ensemble, neural net 같은 모델들이다.

참고로, 지금도 사회학 계열로 연구하는 분들 중에 통계학을 거의 모르는 분들은 stepwise regression이라고 모든 변수를 다 넣어보고 그 중에 설명력 (R-squared)이 제일 좋은 모델을 “단순하게” 고르는 경우도 있다. (저 분도 박사학위가 있는데, 니가 생각하는거처럼 그렇게 절망적으로 통계학을 모르진 않을꺼야…라고? 연구자 취급 못 받을 사람들 진짜로 많다 ㅋㅋㅋ)

 

5. Variable Importance를 통한 변수의 중요도 판단
Q. Random Forest, Boosting 계열의 모델을 보면 Variable importance에 순위를 잡던데, 그 변수가 중요변수가 아닐수도 있다고 하셨던 부분이 어떤 경우였는지 잘 기억이 안 납니다. 

A. Random Forest를 비롯한 Tree 기반의 모델들에서 중요한 변수들 중에, 한 변수가 Tree 안에 여러번 반복이 되기 때문에 중요한 변수로 나타나는 경우가 있다. Tree 구조를 보면 이해가 되겠지만, 그 변수의 여러 구간 (ex. 3-5, 10-15, 30-37 등등, 연속되지 않은 여러 구간)이 y값에 영향을 주는 경우에 Linear regression에서는 연속된 구간 값을 사용하는 탓에 두 모델간 변수의 중요도가 다르게 나올 여지가 있다. 수업 중에서 Variable Importance plot을 그린 다음, 이걸 Regression 기반 모델들에도 무조건 적용하면 문제가 생길 수 있다고 지적하기도 한다.

 

6. 기타 추가 질문
Q. 우연히 ‘머신러닝 피드백을 이용한 게임 지표 분석’ 이라는 글을 보게 되었는데, 제가 실무적으로 활용하면 도움이 될듯한 내용들이라 꼭 적용해보고 싶은데요, 정확히 어떤 개념을 이용하는 것인지 잘 모르겠습니다.

링크의 글을 읽어보면, Vowpal-wabbit을 통해 손실함수를 계산해서 각 독립 변수들이 종속 변수에 어떻게 영향을 주는지 귀납적으로 추론한다는 방식이고, 여기서 독립 변수마다 계산된 연관성 점수 (Relation score)에 따라 positive, negative를 판단하여 독립 변수들의 영향도? 효과? 등을 파악한다는 것인데 여기서 나오는 연관성 점수가 무엇일까요?

A. 두 가지 경우를 생각해보자.

1번. 위의 5번에서 본 Variable importance를 구하는데, +인 경우와 -인 경우를 구분해서 순위를 다시 잡으면 어떻게 될까? 그 다음 단순하게 몇 %의 기여도만 따지는게 아니라, 기여도가 있는 변수들을 다 모아놓고 -100% ~ +100%로 재조정을 한다면?

2번. Regression에서도 가능할까? 물론이다. 모든 변수를 standardize 해놓고, 각각의 상관계수 값에 따라서 어떤 변수가 더 큰 기여도를 가지는지, 어떤 변수가 마이너스 기여도를 가지는지 구분할 수 있다. 그렇게 상관계수들을 모아놓고 등수를 잡거나, -100% ~ +100%로 재조정을 해도 된다.

실제로 Variable importance를 구하는 과정을 보면, 각각의 변수가 cost function 값이 얼마나 변하도록 만드는지에 따라서 변수의 중요도를 따진다. Regression에서 모든 변수를 같은 구간에서 움직이게 묶은 다음 (i.e. 표준화) 상관계수를 보는 것도 같은 컨셉이다.

저 링크의 글을 보면 Vowpal-wabbit이 무슨 엄청나게 기술(!)력 있는 모델인 것처럼 느껴지는데, 정작 위에서 설명한 단순한 중요도 계산 + 재조정 작업에 불과하다.

“우와~ 이거 완전 대박이네!”, “어캐 베끼지?”

보통 공대생, 특히 수학 (계산말고 수학) 공부 많이 안 하는 전공자들에게서 흔히 보는 모습인데, 뭔가 새로운 계산만 나오면 “우와~ 이거 완전 대박이네!”, “어캐 베끼지?” 이런 반응을 보이더라. (뭐, 스티브 잡스도 베끼는거 좋아하긴 했다.) 머신러닝에서 Neural network도 그래서 “떴다”고 생각한다. 그러나, 필자 같은 사람들 눈에는 “저거 XYZABCD로 계산했겠네”가 그냥 눈에 보인다. 평소에도 저런 모델링을 하는걸 머리속으로 고민하고 있으니까. 더더군다나 Vowpal-wabbit은 그렇게 수학적으로 Elegant 한 방법도 아니다. 저거 제대로 돌아가게 하려면 보정작업이 엄청나게 많이 필요할 것이다. (직접 적용해보시라.)



