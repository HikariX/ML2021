# Report

### 作业心得

该作业为食物图片分类，着重对数据增强和半监督学习的实践。

首先，因为是分类问题，并且是食物数据，所以我在一开始做数据增强的时候并不考虑上下翻转或旋转等操作，以防对数据造成污染。最初的数据增强仅限定在了左右翻转，后加入随机裁剪。在测试过程中发现可以随机调节图片的亮度、对比度等，故该增强方法也加入使用。最后，我还是尝试了旋转策略，效果也较好。因为时间关系，无法将所有手段测试完毕，故只使用以上方法进行数据增强。

其次，对于半监督学习，本作业中实践了pseudo-label方法，将无标签数据通过训练的模型判定后打上假标签加入训练。该过程代码由我自行实现。不难发现，该方法需在模型已有较好的辨识水平后才可加入，所以我最终确定在原始数据集的训练集准确率为0.8时才加入pseudo-label。另外，门槛值不可过高。因任务为11类分类，所以分数高于0.09就说明产生了分类的效果。若使用过高门槛，大量数据将无法加入训练。但过低的门槛值也会让模型效果裂化，所以综合门槛设定在了0.6。最后，learning rate schedule 不应该过早启动。因加入pseudo-label后网络还需要调整，所以我设定schedule的计数和执行pseudo-label的次数相关。当达到训练集准确率0.8后，之后的每一次迭代均执行pseudo-label，此时才开始计算schedule。

在实现半监督方法的过程中发现，不同的dataloader，可能返回的数据类型（y标签）不同，从而导致数据集拼接产生问题，一个小坑，不过着实折磨了我半天。

### 训练日志

04.07

对原始网络结构，使用图像变换：

```Python
transforms.RandomHorizontalFlip(p=0.5),
```

此时，public score为0.51851，已经十分优秀，说明合理的图像变换有较好效果。

4.08

使用网络结构

```Python
self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 11)
        )
```

使用水平翻转，使用self-training，设置threshold=0.75，public score-0.53285

已实现pseudo-label方法，目前训练效果一般。

4.09

本日尝试RandomCropResizedCrop方法对图像进行随机裁切，并结合先前实践的翻转操作。另外因为使用BN所以将batch size开到256，但结合pseudo-label时训练效果一般。因此我希望先探索不同的数据增强方式和网络结构，暂时不加入pseudo-label。故重新将batch size设置回128。首先，使用网络结构：

```Python
self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.Sigmoid(),
            nn.MaxPool2d(4, 4, 0),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 8 * 8, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 11)
        )
```

对比是否进行Crop，若执行，public score为0.41696，但若只执行翻转，score降为0.38740，说明该网络效果较差。综合前几次作业经验，可以认为Sigmoid函数目前性能已不佳，因此调节网络的时候应尽量避免。

将所有激活函数改为LeakyReLU：

```Python
 self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(4, 4, 0),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 8 * 8, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 11)
        )
```

仅使用随机水平翻转，80epoch的训练准确率能达到100%，验证集准确率0.57839，public score为0.53405；使用水平和随机裁剪，80epoch的训练准确率为0.913，验证准确率0.547，poblic score为0.57825，结果说明随机裁剪也起作用，另外说明泛化能力强的模型效果更好。当然目前还处在提升验证集准确率的阶段。

保持使用两种图像增强，若加入resnet18，训练集准确率0.940，验证集准确率0.447，public score为0.51553，说明网络存在过拟合。综合来看，感觉可能需要试验更多的图像增强方法，而网络反而不是瓶颈。

4.10

保持昨日最终网络结构，将随机裁剪的比例从(0.5, 1)变为(0.2, 1)，训练集0.75531，验证集0.50807，public score0.58423，表明数据增强还有空间。将batch_size改为64，训练集0.76562，验证集0.57841，public score0.57526。

保持以上数据变换操作和bsize64，网络改为densenet161，训练集0.91167，验证集0.57699，public score0.59259，表明合适的网络还有提升空间。当epoch从80到300时，加上lr_scheduler，设定50epoch后lr*0.1，训练集0.97353，验证集0.57045，poblic score爬升到0.61350。综合来看，后续确实需要将val acc提升到较高水平才有比较的意义。

4.11

使用训练完毕的DenseNet161，数据变换为

```Python
transforms.RandomOrder([
        # You may add some transforms here.
        transforms.RandomResizedCrop(size=128, scale=(0.3,1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)]),
```

设定在开始pseudo label的情况下每50epoch下降90%的学习率，且pseudo label的门槛为0.6。当训练集分数超过0.8时才允许加入pseudo label。经过200epoch后，训练集0.97024，验证集0.67557，public score0.64456。

同上设置，门槛0.5，效果较差，略过。

同上设置，门槛0.7，训练集0.98061，验证集0.70085，public score为0.64336

4.12

同上设置，门槛0.6，训练集0.96616，验证集0.70085，public score为0.66367

4.13

同上设置，门槛0.4，训练集0.93788，验证集0.68807，public score为0.65292