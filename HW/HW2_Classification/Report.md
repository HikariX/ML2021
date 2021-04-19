# Report

## 2.1

### 作业心得

​		HW2.1的学习内容为复杂神经网络的训练、调参，以及一些网络训练技巧的应用。

​		在试验中我发现，虽然数据和特征众多，但一个过深的网络已无法捕捉数据特征，这足以说明，网络自身因为结构问题，存在局限性和训练瓶颈，若要有较大的性能提升，只能寄希望于处理时序关系的神经网络。另外，在调参过程中，我试验了Dropout、Batchnorm和learning rate decay三种技巧。

​		Dropout在训练每一轮控制当前层随机丢弃部分神经元，并用余下神经元输出值加以等比例放大，保证输出尽量不变。随着训练轮次增加，不同的神经元都会被训练到，但训练的神经元组合不尽相同。换言之，我们在足够多的训练轮次下，各种神经元组合方案都会被经过一定训练，从而成为一个有效的小型网络结构。这一系列网络结构的结果组合为最终预测时的网络结构。因此，这一过程降低训练时的准确率，但是测试时准确率会提升。该过程可以将不同网络结构视为不同模型，最终结果是这一系列模型的组合，即“ensemble model”。如此增加模型的稳定性和性能。

​		因为数据众多、网络较深，使用batchnorm技巧可以较好地令网络收敛。对每一层末尾添加batchNorm，则输出结果被移动到均值为0，方差为1的分布下，并送入下一次训练中。我粗略阅读了论文*How Does Batch Normalization Help Optimization?*，其表述bn操作能够改变函数的平滑度，由此，优化过程可以更稳定。另外，因为平滑度改善，所以更大的步长也走在正确的下降方向上，所以优化效果会更好。当然需要注意的是，因为batchnorm针对一批数据进行，所以batchsize不可过小，否则数据分布将不具有代表性，会使得网络性能劣化。

​		Learning rate decay是在训练一段时期后减少学习率，因我们倾向于认为训练后期会走到一个minima，如果它不再下降，那么已可以终止，否则应存在一个较窄的下降区域，此时足够小的学习率能够帮助下探。

### 训练日志

3.16

基准网络已经通过基准测试

3.18

加深网络层有效果，验证集分数0.72，测试结果0.71

```Pyth
self.net = nn.Sequential(
            nn.Linear(429, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),

            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.BatchNorm1d(4096),

            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),

            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),

            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),

            nn.Linear(256, 39),
        )
```

通过在前两层加入dropout，验证集提升到0.73，测试结果提升到0.7179

新网络结构，分数提升到0.735，训练50epoch，batch512，学习率0.001

```Python
self.net = nn.Sequential(
            nn.Linear(429, 1024),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024),

            nn.Linear(1024, 2048),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.BatchNorm1d(2048),

            nn.Linear(2048, 512),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),

            nn.Linear(512, 128),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),

            nn.Linear(128, 39),
        )
```

通过增加batchsize到1024，训练次数到200epoch，学习率0.005，分数提升到0.74309

3.19

bsize1024 大约96epoch达到val set 0.763，但test set分数为0.74375。

```python
self.net = nn.Sequential(
            nn.Linear(429, 1024),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024),

            nn.Linear(1024, 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.BatchNorm1d(4096),
  
            nn.Linear(4096, 2048),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.BatchNorm1d(2048),

            nn.Linear(2048, 512),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),

            nn.Linear(512, 128),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),

            nn.Linear(128, 39),
        )
```

其他设置同上，test set0.74358

```Python
self.net = nn.Sequential(
            nn.Linear(429, 1024),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024),

            nn.Linear(1024, 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.BatchNorm1d(4096),

            nn.Linear(4096, 512),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),

            nn.Linear(512, 128),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),

            nn.Linear(128, 39),
        )
```

bsize 2048，val set比例0.1，test set分数0.747

```Python
self.net = nn.Sequential(
            nn.Linear(429, 2048),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.BatchNorm1d(2048),

            nn.Linear(2048, 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.BatchNorm1d(4096),

            nn.Linear(4096, 1024),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024),

            nn.Linear(1024, 256),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),

            nn.Linear(256, 39),
        )
```

增加bsize到8192，test set分数0.73937

3.20

bsize16384 weightdecay1e-5，score0.74783

```Py
self.net = nn.Sequential(
            nn.Linear(429, 1024),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024),

            nn.Linear(1024, 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.BatchNorm1d(4096),   
            
            nn.Linear(4096, 1024),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024),

            nn.Linear(1024, 256),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),

            nn.Linear(256, 39),
        )
```



bsize 16384 wd1e-5 score 0.74619

```Py
self.net = nn.Sequential(
            nn.Linear(429, 2048),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.BatchNorm1d(2048),

            nn.Linear(2048, 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.BatchNorm1d(4096),
            
            nn.Linear(4096, 1024),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024),

            nn.Linear(1024, 256),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),

            nn.Linear(256, 39),
        )

```



bsize 32768 wd1e-5 score 0.74484

bsize32768 wd1e-5 score 0.74548

bsize 32768 wd1e-5 score 0.74819

```Py	
self.net = nn.Sequential(
            nn.Linear(429, 1024),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024),

            nn.Linear(1024, 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.BatchNorm1d(4096),   
            
            nn.Linear(4096, 1024),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024),

            nn.Linear(1024, 256),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),

            nn.Linear(256, 39),
        )
```

bsize 32768 wd1e-5 score0.74705

```Python
self.net = nn.Sequential(
            nn.Linear(429, 1024),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024),

            nn.Linear(1024, 2048),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.BatchNorm1d(2048),
#
#            nn.Linear(1024, 2048),
#            nn.Dropout(0.5),
#            nn.LeakyReLU(),
#            nn.BatchNorm1d(2048),

            
            nn.Linear(2048, 512),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.BatchNorm1d(512),

#            nn.Linear(1024, 256),
#            nn.Dropout(0.5),
#            nn.LeakyReLU(),
#            nn.BatchNorm1d(256),

            nn.Linear(512, 39),
        )
```



同，网络结构更长，0.74647

```Python
self.net = nn.Sequential(
            nn.Linear(429, 1024),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024),

            nn.Linear(1024, 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.BatchNorm1d(4096),   
            
            nn.Linear(4096, 1024),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024),

            nn.Linear(1024, 256),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),

            nn.Linear(256, 64),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),

            nn.Linear(64, 39),
        )
```



使用全体数据实验，无wd，bsize 65536 test score 0.75185

```Py
self.net = nn.Sequential(
            nn.Linear(429, 1024),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024),

            nn.Linear(1024, 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.BatchNorm1d(4096),   
            
            nn.Linear(4096, 1024),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.BatchNorm1d(1024),

            nn.Linear(1024, 256),
            nn.Dropout(0.5),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),

            nn.Linear(256, 39),
        )
```

使用相同网络对0.1val set和bsize8192实验150epoch，testscore0.74909，为相同网络环境下最高者，因此之后可固定bsize，用全数据实验

在这设置下让bsize变为4096发现效果不好。

bsize为8192时候，200epoch，全数据，test score0.75359，目前最高

PReLU 105epoch testscore0.75138

## 2.2

### 作业心得

​		作业2.2中要求通过给定代码得到梯度值和比值，确定当前优化阶段的状态。通过代码，得到结果如下：

```Python
gradient norm: 0.0007818487647455186, minimum ratio: 0.49609375
```

很显然，此时，gradient norm低于1e-3，所以是题目定义的critical point。而观察ratio小于0.5，说明此时正的特征值较少，所以是一个saddle point。