# Report

### 作业心得

本次作业为基于网络模型，计算得到能够让模型误识别的图片。本次任务主要要求实现Fast Gradient Sign Method（FGSM）和Iterative FGSM方法。通过medium baseline的要求为实现IFGSM，通过strong和boss baseline的要求为使用Model ensemble attack。因为使用JudgeBoi（内部评估软件），所以我无法上传进行评判。

首先，FGSM是指在允许的图像变化范围内走最大的步长。此时假设允许的变化范围是$\epsilon$，那么最大的步长就是$1*\epsilon$。所以对于一张图片，对输入$x$求偏导，每个位置方向的梯度取sign函数，加到原始图像上，即可得到变化后的图像。对每张图片进行attack，则每个像素位置均移动了一定的长度，导致图像数据本身发生变化；但由于移动的程度对人眼而言可以忽略，所以人无法察觉。

<img src="image-20210608153001289.png" alt="image-20210608153001289" style="zoom:25%;" />

在此基础上，如果移动的步长收小，并规定每次移动后超过$\epsilon$范围的步长截断，多次迭代就得到了IFGSM方法，实现不难。

最后，课上提到Ensemble方法。Ensemble的技巧在前几次作业中均提到。因为在本次课中介绍较为详细，我进行了深入实践。研究表明，对于识别同一类图片（在同种数据集上训练）的分类器，其能容忍的图片改变而类别不变的数据空间相似，所以将同样数据集上得到的几种不同模型结合起来，用它们的结果对目标模型进行攻击，可以得到更均衡、更科学的结果。具体实现的时候，我参考了论文*DELVING INTO TRANSFERABLE ADVERSARIAL EX- AMPLES AND BLACK-BOX ATTACKS*，主要呈现为如下公式：
$$
\mathbf{argmin}_{x^*}-log((\sum_{i=1}^k\alpha_iJ_i(x^*))\cdot\mathbf{1}_{y^*})+\lambda d(x,x^*)
$$
其中$x^*$代表最终的图形，$\alpha_i$为第$i$个模型的权重，一般权重总和为1。$J_i(\cdot)$表示第$i$模型输出的softmax分数。对于有目标攻击，需要指定$y^*$，则上式意为“对指定为$y^*$标签类的图片，其攻击后产生的softmax分数必须在该类别上最大”。但我们是无目标攻击，所以换种方式理解，可以将(1)式的$\mathbf{1}_{y^*}$换成$\mathbf{1}_{y}$，即被攻击图片的原始标签。此时优化目标不是最小化，而是最大化以上式，意为“让模型们对正确标签输出的分数尽可能低”。不需要管到底能攻击到哪个标签，只要让原始的标签被攻击到即可。

### 训练日志

07.16

首先跑了一遍DaNN基础代码，200epoch，同时我设置$\lambda$为0.3，public score为0.58448，差一点超过。为了节省时间，我直接开始尝试boss baseline的实现。经过论文挑选，我选择了大致能理解透彻的MCD方法，并构想了其实现。但构思完成后我注意到已有代码，于是直接使用其实现，经过修改后移植。但最终public为0.44508。无奈之下，我重回DaNN，设置300epoch，$\lambda$为0.4，意外地发现结果为0.63382，成功超过medium。
