# Report

### 作业心得

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

将所有激活函数改为LeakyReLU，仅使用随机水平翻转，80epoch的训练准确率能达到100%，验证集准确率0.57839，public score为0.53405；使用水平和随机裁剪，80epoch的训练准确率为0.913，验证准确率0.547，poblic score为0.57825，结果说明随机裁剪也起作用，另外说明泛化能力强的模型效果更好。当然目前还处在提升验证集准确率的阶段。

保持使用两种图像增强，加入resnet18，训练集准确率0.940，验证集准确率0.447，public score为0.51553，说明网络存在过拟合。综合来看，感觉可能需要试验更多的图像增强方法，而网络反而不是瓶颈。

