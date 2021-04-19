# Report

### 作业心得

该作业为通过语音讯号识别说话者，主要考量transformer的运用。

因transformer已有封装包，因此毋需自我实现。助教在作业中安排了对encoder的调参实验。这是通过medium baseline的要求。由于训练时间关系，我并未完成这一过程，而是直接朝strong baseline出发。

通过strong baseline要求尝试conformer。我通过阅读论文*Conformer- Convolution-augmented Transformer for Speech Recognition*，成功实现这一架构，并对文章提到的point-wise与depth-wise卷积操作进行了学习。前者为使用宽1的卷积核对数据进行处理，而后者则是对多通道图片的每一个通道分开计算。在后续提高过程中，我也实现了论文*Self-Attentive Speaker Embeddings for Text-Independent Speaker Verification*提到的attention pooling，性能确有提升。

在训练过程中，public score不断逼近strong baseline（0.95404），但最高只达到0.92879，咨询群友发现d_model这一参数可以适当调大。该参数对应的维度为对数mel图中的特征维（mel图包含特征维和时间维）。将mel图放入一个初始Linear层时，通过矩阵运算将其扩为d_model, length的维度。在网络的初期应尽量保留原始数据参数，才可被后续结构提取出数据特征。助教提供的d_model为80，而mel图原始为40。在改进中，我将d_model调为160，同样的双层conformer下public score暴增到0.94642，说明该方法十分有效。后又将d_model调为256，双层conformer下public score达到了0.95119，但因为colab网络断开，没法继续训练，最终距离baseline咫尺，遗憾收场。

另外，我也尝试实现助教ppt中的hint之additive margin softmax，即通过改造softmax函数，将数据区分的标准变得更分离，从而达到更准确的划分。torch中的分类损失函数CrossEntropyLoss已经封装了softmax，所以只能重新实现损失函数。但最终训练时loss变为负数，只好剔除。由于时间关系无法深入研究，较为可惜。



### 训练日志

04.13

实现Conformer，单层public score为0.83309，双层为0.89880。参数过多，疏于记录。

三层时1w Step的acc为0.4430，2w Step的acc为0.5608，3w Step的acc为0.6177

04.14

尝试加入论文https://www.isca-speech.org/archive/Interspeech_2018/pdfs/1158.pdf提到的attention pooling机制，对encoder出来的数据进行基于attention的pooling。同昨日设置，1w-0.4696，2w-0.5750，3w-0.6385，4w-0.6866，5w-0.7212，6w-0.7434，7w-0.7680，8w-0.7850，9w-0.7948，10w-0.8000。可看出训练已接近尾声，难以下降。此时public score为0.9200！

加入4层conformer，将所有激活函数（除conformer）改成LeakyReLU测试。1w-0.4722，2w-0.5762，3w-0.6526，4w-0.6940，5w-0.7280

网络结构参数为：

```Python
class Conformer(nn.Module):
  def __init__(self, d_model=80):
    super().__init__()

    # in: (batch size, length, d_model)
    # out: (batch size, length, d_model)
    self.feedForward1 = nn.Sequential(
        # out: (length, batch size, d_model)
        nn.LayerNorm(d_model),
        # out: (length, batch size, d_model)
        nn.Linear(d_model, 512), 
        nn.Hardswish(),
        # out: (length, batch size, d_model)
        nn.Dropout(0.3),
        # out: (length, batch size, d_model)
        nn.Linear(512, d_model), 
        # out: (length, batch size, d_model)
        nn.Dropout(0.3)
    )

    # in: (batch size, length, d_model)
    # out: (batch size, length, d_model)
    self.feedForward2 = nn.Sequential(
        # out: (length, batch size, d_model)
        nn.LayerNorm(d_model),
        # out: (length, batch size, d_model)
        nn.Linear(d_model, 256), 
        nn.Hardswish(),
        # out: (length, batch size, d_model)
        nn.Dropout(0.2),
        # out: (length, batch size, d_model)
        nn.Linear(256, d_model), 
        # out: (length, batch size, d_model)
        nn.Dropout(0.2)
    )

    # in: (length, batch size, d_model)
    # out: (length, batch size, d_model)
    self.multiHeaded = nn.Sequential(
        nn.LayerNorm(d_model),
        PositionalEncoding(d_model),
        MHA(d_model, 2),
        nn.Dropout(0.1)
    )

    # in: (batch size, 1, length, d_model)
    # out: (batch size, 1, length, d_model)
    self.convolution = nn.Sequential(
        nn.LayerNorm(d_model),
        # Point-wise conv
        nn.Conv2d(1, 10, 1, 1, 0),
        nn.GLU(dim=1),
        # Depth-wise conv
        nn.Conv2d(in_channels=5, out_channels=6, kernel_size=3, padding=1, stride=1, groups=5),# group means separate kernels for each channel.
        nn.BatchNorm2d(5),
        nn.Hardswish(),
        # Point-wise conv
        nn.Conv2d(6, 1, 1, 1, 0),
        nn.Dropout(0.1)
    )

    self.layerNorm = nn.LayerNorm(d_model)

  def forward(self, inputs):
    """
    args:
      inputs: (batch size, length, d_model)
    return:
      out: (batch size, n_spks)
    """
    # out: (batch size, length, d_model)
    out = inputs + 0.5 * self.feedForward1(inputs)
    # out: (length, batch size, d_model)
    out = out.permute(1, 0, 2)
    # out: (length, batch size, d_model)
    out = out + self.multiHeaded(out)
    # out: (batch size, length, d_model)
    out = out.permute(1, 0, 2)
    out = out.unsqueeze(1)
    # out: (batch size, 1, length, d_model)
    out = out + self.convolution(out)
    # out: (batch size, length, d_model)
    out = out.squeeze(1)
    # out: (batch size, length, d_model)
    out = out + 0.5 * self.feedForward2(out)
    out = self.layerNorm(out)
    return out

class Classifier(nn.Module):
  def __init__(self, d_model=80, n_spks=600, dropout=0.1):
    super().__init__()
    # Project the dimension of features from that of input into d_model.
    self.prenet = nn.Linear(40, d_model)
    # TODO:
    #   Change Transformer to Conformer.
    #   https://arxiv.org/abs/2005.08100
    self.encoder_layer = nn.TransformerEncoderLayer(
      d_model=d_model, dim_feedforward=256, nhead=2
    )
    # self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

    self.conformer1 = Conformer()
    self.conformer2 = Conformer()
    self.conformer3 = Conformer()
    self.conformer4 = Conformer()

    
    # Project the the dimension of features from d_model into speaker nums.
    self.pred_layer = nn.Sequential(
      nn.Linear(d_model, d_model),
      nn.LeakyReLU(),
      nn.Linear(d_model, n_spks),
    )

    # strange attention pooling https://www.isca-speech.org/archive/Interspeech_2018/pdfs/1158.pdf
    # out: (batch size, length, d_model)
    self.attnpool = nn.Sequential(
        nn.Linear(d_model, d_model),
        nn.LeakyReLU(),
        nn.Linear(d_model, 1),
        nn.Softmax(dim=1),
    )

  def forward(self, mels):
    """
    args:
      mels: (batch size, length, 40)
    return:
      out: (batch size, n_spks)
    """
    # out: (batch size, length, d_model)
    out = self.prenet(mels)
    # out: (batch size, length, d_model)
    out = self.conformer1(out)
    # out: (batch size, length, d_model)
    out = self.conformer2(out)
    # out: (batch size, length, d_model)
    out = self.conformer3(out)
    # out: (batch size, length, d_model)
    out = self.conformer4(out)
    
    # # mean pooling
    # stats = out.mean(dim=1)
    # strange attention pooling https://www.isca-speech.org/archive/Interspeech_2018/pdfs/1158.pdf
    # out: (batch size, length, 1)
    stats = self.attnpool(out)

    # out: (batch size, d_model)
    out = torch.bmm(out.permute(0, 2, 1), stats)
    out = torch.squeeze(out, 2)

    # out: (batch, n_spks)
    out = self.pred_layer(out)

    return out
```

4.15

微调结构，增加到4层conformer和8-head。1w-0.4938，2w-0.6043，3w-0.6771，4w-0.7200，5w-0.7395，6w-0.7758，7w-0.7932，8w-0.8113，9w-8193，10w-0.8236。public score0.92857。

```Python
class Conformer(nn.Module):
  def __init__(self, d_model=80):
    super().__init__()

    self.hidden = 256
    # in: (batch size, length, d_model)
    # out: (batch size, length, d_model)
    self.feedForward1 = nn.Sequential(
        # out: (length, batch size, d_model)
        nn.LayerNorm(d_model),
        # out: (length, batch size, d_model)
        nn.Linear(d_model, self.hidden), 
        nn.Hardswish(),
        # out: (length, batch size, d_model)
        nn.Dropout(0.3),
        # out: (length, batch size, d_model)
        nn.Linear(self.hidden, d_model), 
        # out: (length, batch size, d_model)
        nn.Dropout(0.3)
    )

    # in: (batch size, length, d_model)
    # out: (batch size, length, d_model)
    self.feedForward2 = nn.Sequential(
        # out: (length, batch size, d_model)
        nn.LayerNorm(d_model),
        # out: (length, batch size, d_model)
        nn.Linear(d_model, self.hidden), 
        nn.Hardswish(),
        # out: (length, batch size, d_model)
        nn.Dropout(0.3),
        # out: (length, batch size, d_model)
        nn.Linear(self.hidden, d_model), 
        # out: (length, batch size, d_model)
        nn.Dropout(0.3)
    )

    # in: (length, batch size, d_model)
    # out: (length, batch size, d_model)
    self.multiHeaded = nn.Sequential(
        nn.LayerNorm(d_model),
        PositionalEncoding(d_model),
        MHA(d_model, 8),
        nn.Dropout(0.2)
    )

    # in: (batch size, 1, length, d_model)
    # out: (batch size, 1, length, d_model)
    self.convolution = nn.Sequential(
        nn.LayerNorm(d_model),
        # Point-wise conv
        nn.Conv2d(1, 12, 1, 1, 0),
        nn.GLU(dim=1),
        # Depth-wise conv
        nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, padding=1, stride=1, groups=6),# group means separate kernels for each channel.
        nn.BatchNorm2d(6),
        nn.Hardswish(),
        # Point-wise conv
        nn.Conv2d(6, 1, 1, 1, 0),
        nn.Dropout(0.1)
    )

    self.layerNorm = nn.LayerNorm(d_model)

  def forward(self, inputs):
    """
    args:
      inputs: (batch size, length, d_model)
    return:
      out: (batch size, n_spks)
    """
    # out: (batch size, length, )
    out = inputs + 0.5 * self.feedForward1(inputs)
    # out: (length, batch size, d_model)
    out = out.permute(1, 0, 2)
    # out: (length, batch size, d_model)
    out = out + self.multiHeaded(out)
    # out: (batch size, length, d_model)
    out = out.permute(1, 0, 2)
    out = out.unsqueeze(1)
    # out: (batch size, 1, length, d_model)
    out = out + self.convolution(out)
    # out: (batch size, length, d_model)
    out = out.squeeze(1)
    # out: (batch size, length, d_model)
    out = out + 0.5 * self.feedForward2(out)
    out = self.layerNorm(out)
    return out

class Classifier(nn.Module):
  def __init__(self, d_model=80, n_spks=600, dropout=0.1):
    super().__init__()
    # Project the dimension of features from that of input into d_model.
    self.prenet = nn.Linear(40, d_model)
    # TODO:
    #   Change Transformer to Conformer.
    #   https://arxiv.org/abs/2005.08100
    self.encoder_layer = nn.TransformerEncoderLayer(
      d_model=d_model, dim_feedforward=256, nhead=2
    )
    # self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

    self.conformer1 = Conformer()
    self.conformer2 = Conformer()
    self.conformer3 = Conformer()
    self.conformer4 = Conformer()

    
    # Project the the dimension of features from d_model into speaker nums.
    self.pred_layer = nn.Sequential(
      nn.Linear(d_model, d_model),
      nn.LeakyReLU(),
      nn.Linear(d_model, n_spks),
    )

    # strange attention pooling https://www.isca-speech.org/archive/Interspeech_2018/pdfs/1158.pdf
    # out: (batch size, length, d_model)
    self.attnpool = nn.Sequential(
        nn.Linear(d_model, d_model),
        nn.LeakyReLU(),
        nn.Linear(d_model, 1),
        nn.Softmax(dim=1),
    )

  def forward(self, mels):
    """
    args:
      mels: (batch size, length, 40)
    return:
      out: (batch size, n_spks)
    """
    # out: (batch size, length, d_model)
    out = self.prenet(mels)
    # out: (batch size, length, d_model)
    out = self.conformer1(out)
    # out: (batch size, length, d_model)
    out = self.conformer2(out)
    # out: (batch size, length, d_model)
    out = self.conformer3(out)
    # out: (batch size, length, d_model)
    out = self.conformer4(out)
    
    # # mean pooling
    # stats = out.mean(dim=1)
    # strange attention pooling https://www.isca-speech.org/archive/Interspeech_2018/pdfs/1158.pdf
    # out: (batch size, length, 1)
    stats = self.attnpool(out)

    # out: (batch size, d_model)
    out = torch.bmm(out.permute(0, 2, 1), stats)
    out = torch.squeeze(out, 2)

    # out: (batch, n_spks)
    out = self.pred_layer(out)

    return out

```

改变结构，将d_model增加到160，使用单层conformer，1w-0.6077，2w-0.6721，3w-0.7277，4w-0.7533，5w-0.7802，6w-0.8086，7w-0.8273，8w-0.8399网络停了。。

重新训练，使用双层conformer，1w-0.5632，2w-0.6767，3w-0.7218，4w-0.7566，5w-0.7895，6w-0.8177，7w-0.8377，8w-0.8443，9w-0.8557，10w-0.8576。public score为0.94642！

4.16

使用双层conformer，d_model=256，1w-0.5487，2w-0.6802，3w-0.7332，4w-0.7771，5w-0.7905，6w-0.8080，7w-0.8296，8w-0.8465，9w-0.8592，10w-0.8681，11w-0.87xx，网络停了，还好每次都把模型下载了。

public score0.95119！！！但是时间不够了，只得作罢。

