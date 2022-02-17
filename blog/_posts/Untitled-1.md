---
title : 복잡한 모델 생성
toc : true
toc_sticky : true
---

> "part 3 케라스(Keras) 07 복잡한 모델 생성"에 대한 정리

## 7.1 함수형 API(Functional API)
- Sequential API : 여러 층을 시퀀스 형태로 연결
- Functional API : 복잡한 구조의 모델 정의 가능  
  → 다양한 모델 구조를 구현 가능하다는 장점이 존재

함수형 API 사용 방법  
Sequential API로 생성한 모델과 동일 방식으로 훈련한다. 하지만 그전에 레이어 설정과 모델 구조 설정 과정에서 기존 설정 방법과는 차이를 보인다.  
1. Input 레이어 정의 : 데이터의 입력 shape을 정의
2. 변수 설정 : 레이어별로 반환되는 출력 값을 저장
3. 출력값을 저장한 변수들을 다음 레이어의 입력으로 연결
4. 체인 방식으로 연결한 후 모델 생성(tf.keras.Model() 사용, 입력 레이어와 출력레이어를 정의)

```python
# input layer 정의
input_layer = tf.keras.Input(shape=(28, 28), name='InputLayer')

# 모델의 레이어 -> 체인 구조로 연결
# 레이어마다 name 매개변수로 이름을 지정해줄 수 있다.
x1 = tf.keras.layers.Flatten(name='Flatten')(input_layer)
x2 = tf.keras.layers.Dense(256, activation='relu', name='Dense1')(x1)
x3 = tf.keras.layers.Dense(64, activation='relu', name='Dense1')(x2)
x4 = tf.keras.layers.Dense(10, activation='relu', name='OutputLayer')(x3)

# 모델 생성
func_model = tf.keras.Model(inputs=input_layer, outputs=x4, name='FunctionalModel')

# 모델 요약
func_model.summary()
```
모델의 구조도를 **시각화**하여 알아볼 수 있는 방법이 있다.
케라스의 유틸 패키지에서 제공하는 plot_model 모듈을 활용하면 된다.
```python
from tensorflow.keras.utils import plot_model

# 구조도 시각화
plot_model(func_model, show_shapes=True, show_layer_name=True, to_file='model.png')
```
- show_shapes : 데이터의 입출력 shape을 출력
- show_layer_names : 레이어의 이름을 출력
- to_file : 해당 구조도를 이미지 파일로 저장

```python
# 기존의 방식과 동일한 컴파일, 훈련, 검증 과정
func_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

func_model.fit(x_train, y_train, epochs=3)

loss, acc = func_model.evalutate(x_test, y_test, verbose = 0)
print(f'Functional Model] loss: {loss:.5f}, acc: {acc:.5f}')
```


## 7.2 모델 서브클래싱(Model Subclassing)
텐서플로 케라스에서 자체 제공하는 Model 클래스로 딥러닝 모델을 구현할 수 있다.  
이 클래스를 사용자가 직접 딥러닝 모델을 만들 수 있다. 해당 방법을 모델 서브클래싱이라고 부른다.

책에 의하면 파이썬 클래스 개념이 익숙한 사람들에게 가장 추천하는 방법이라고 한다. 객체 지향과 관련된 개념이 풍부한 사람에게 적합한 접근 방법인 것 같다.

*(그렇다 하더라도 함수형 API로 생성한 모델과 성능 차이는 없다고 한다.)*

모델 서브클래싱으로 모델 인스턴스 생성하기
1. tf.keras.Model을 상속받아 모델 클래스를 구현한다.
   - init 함수에 레이어와 레이어의 하이퍼파라미터를 정의
   - call() 함수를 메소드 오버라이딩으로 구현  
     함수 내부에서 하는 일 : foward propagation(순전파, 모델의 입력~출력까지의 흐름)을 정의하고 모든 레이어를 체인처럼 연결하는 작업을 수행, 최종 출력값을 return 한다.
```python
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        #초기값 설정
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.layers.Dense(256, activation='relu')
        self.dense2 = tf.layers.Dense(64, activation='relu')
        self.dense3 = tf.layers.Dense(10, activation='softmax')

    #method overiding
    #훈련용 함수를 정의한다
    #x는 input
    def call(self, x):
        x=self.flatten(x)
        x=self.dense1(x)
        x=self.dense2(x)
        x=self.dense3(x)
        return x
```

```python
#모델 인스턴스 생성
mymodel = MyModel()
mymodel._name='subclass_model'
mymodel(tf.keras.layers.Input(shape=(28, 28)))
mymodel.summary()

mymodel.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
mymodel.fit(x_train, y_train, epochs=3)
loss, acc=mymodel.evaluate(x_test, y_test, verbose=0)
print(f'Subclassing Model] loss: {loss:.5f}, acc: {acc:.5f}')
```

## 서브클래싱 모델 파라미터를 활용한 생성
Model Subclassing의 장점 : 생성자 파라미터로 모델 내부 레이어의 하이퍼파라미터 지정 가능

```python
class MyModel(tf.keras.Model):

    #생성자 파라미터 추가
    def __init__(self.units.num_classes):
        super(MyModel, self).__init__()
        # 초기값 설정
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units/4, activation='relu')
        self.dense3 = tf.keras.layers.Dense(num_classes, activation='softmax')
    
    #class overiding
    #훈련용 함수 정의, x는 input
    def call(self, x):
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x
    
