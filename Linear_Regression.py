# From https://www.tensorflow.org/get_started/get_started
import tensorflow as tf # tensorflow import한다

# Model parameters
#W,b 각각의 변수를 정의/타입은 32bit float
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)

# Model input and output
#x,y 각각의 값을 넘겨줄 placeholder 정의
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

#hypothesis 정의
linear_model = x * W + b

# loss/cost function 정의, 이 loss를 최소화해야한다
loss = tf.reduce_sum(tf.square(linear_model - y))  # sum of the squares

# loss를 일일이 미분할 필요없이 optimizer를 사용해서 간단하게 구현
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)#위의 최소 loss를 train 변수에 할당

# training data를 정해줌
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

# training loop
init = tf.global_variables_initializer()#변수 초기화 함수 init
sess = tf.Session()#session 열기
sess.run(init)  # 변수 초기화 진행
#학습
for i in range(1000):# 1000번 실행
    sess.run(train, {x: x_train, y: y_train})#train 실행 /x<-x_train,y<-y_train

# evaluate training accuracy

curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})#학습 결과값 할당
print("W: %s b: %s loss: %s" % (curr_W, curr_b, curr_loss))#결과값 출력
