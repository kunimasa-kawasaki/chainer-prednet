# chainer-prednet
Deep Predictive Coding NetworksをChainerで実装を試みたもの。

#中身
imageフォルダに入っているのは●が移動するサンプルデータです。  
modelフォルダに入っているmodelデータは読み込み方法確認用です。(sampleの0~5000回の学習状態)  

#注意点
ConvLSTMを使用していないため参考文献2のCNN->LSTM->deCNNモデルです。  
PredNetは階層性が特徴ですが、実装は1層のみとなっています。  

#参考文献
  1. Deep Predictive Coding Networks for Video Prediction and Unsupervised Learning(http://arxiv.org/abs/1605.08104)
  3. Unsupervised Learning of Visual Structure using Predictive Generative Networks(http://arxiv.org/abs/1511.06380)
