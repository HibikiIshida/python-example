# -*- coding: utf-8 -*-

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training, Variable

class MyChain(chainer.Chain) :
  def __init__(self) :
    super(MyChain, self).__init__()
    with self.init_scope() :
      self.l1 = L.Linear(None, 5)
      self.l2 = L.Linear(None, 5)
      self.l3 = L.Linear(None, 3)
      self.l4 = L.Linear(None, 2)
  def __call__(self, x) :
    h1 = F.relu(self.l1(x))
    h2 = F.relu(self.l2(h1))
    h3 = F.relu(self.l3(h2))
    y = self.l4(h3)
    return y

# データの設定
with open('test_data.txt', 'r') as f :
    lines = f.readlines()
data = []
for l in lines :
    d = l.strip().split()
    data.append(list(map(int, d)))
trainx = np.array(data, dtype = np.float32)

# chainerの設定
# ニューラルネットワークの登録
model = L.Classifier(MyChain(), lossfun = F.softmax_cross_entropy)

chainer.serializers.load_npz("result/out.model", model)

print("XOR論理演算子")

for i in range(len(trainx)) :
    x = chainer.Variable(trainx[i].reshape(1, 2))
    result = F.softmax(model.predictor(x))
    print("input : {}, result : {}".format(trainx[i], result.data.argmax()))