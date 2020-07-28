# -*- coding: utf-8 -*-

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import chainer.initializers as I
from chainer import training
from chainer.training import extensions

class MyChain(chainer.Chain) :
  def __init__(self) :
    super(MyChain, self).__init__()
    with self.init_scope() :
      self.l1 = L.Linear(None, 3)
      self.l2 = L.Linear(None, 3)
      self.l3 = L.Linear(None, 2)
  def __call__(self, x) :
    h1 = F.relu(self.l1(x))
    h2 = F.relu(self.l2(h1))
    y = self.l3(h2)
    return y

# データの設定
with open('train_data.txt', 'r') as f :
  lines = f.readlines()

data = []
for l in lines :
  d = l.strip().split()
  data.append(list(map(int, d)))

data = np.array(data, dtype = np.int32)
trainx, trainy = np.hsplit(data, [2])
trainy = trainy[:, 0]
trainx = np.array(trainx, dtype = np.float32)
traint = np.array(trainy, dtype = np.int32)

train = chainer.datasets.TupleDataset(trainx, trainy)
test = chainer.datasets.TupleDataset(trainx, trainy)

# chainerの設定
# ニューラルネットワークの登録
model = L.Classifier(MyChain(), lossfun = F.softmax_cross_entropy)
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)

# イテレータの定義
batchsize = 4
train_iter = chainer.iterators.SerialIterator(train, batchsize) # 学習用
test_iter = chainer.iterators.SerialIterator(test, batchsize, repeat=False, shuffle=False) # 評価用

#アップデータの登録
updater = training.StandardUpdater(train_iter, optimizer)

# トレーナの登録
epoch = 500
trainer = training.Trainer(updater, (epoch, 'epoch'))

# 学習状況の表示や保存
trainer.extend(extensions.LogReport()) # ログ
trainer.extend(extensions.Evaluator(test_iter, model)) # エポック数の表示
trainer.extend(extensions.PrintReport([
                                       'epoch', 
                                       'main/loss', 
                                       'validation/main/loss', 
                                       'main/accuracy', 
                                       'validation/main/accuracy', 
                                       'elapsed_time'])) # 計算状況の表示

# 学習開始
trainer.run()

# モデルの保存
chainer.serializers.save_npz("result/out.model", model)
