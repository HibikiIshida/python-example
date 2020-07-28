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
      self.l1 = L.Linear(None, 3)
      self.l2 = L.Linear(None, 2)
  def __call__(self, x) :
    h1 = F.relu(self.l1(x))
    y = self.l2(h1)
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

print("AND論理演算子")

for i in range(len(trainx)) :
    x = chainer.Variable(trainx[i].reshape(1, 2))
    result = F.softmax(model.predictor(x))
    print("input : {0}, result : {1}".format(trainx[i], result.data.argmax()))