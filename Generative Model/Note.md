# Generative Model

在传统的神经网络任务中，输入不同的x，给出不同的y。然而该网络无法处理对于确定输入产生多个输出的问题。例如，对于预测游戏画面，角色在某分叉点可能执行不同的下一步操作，但对应的前序训练资料可能相同，从而网络会产生误解。此时对不同的分叉结果赋予不同的采样概率，即可解决这一问题。如下图所示，当输入x时，同时输入一个分布的采样。对于不同的x选取同一个分布的不同采样，从而输出的y也开始遵循某一个分布。

<img src="image-20210421103225290.png" alt="image-20210421103225290" style="zoom:25%;" />

需要注意的是，用于采样的z分布必须足够简单（能够了解其分布），以便于采样。

***为什么需要分布信息？***

如上所述，当单输入可能对应多输出时，网络将需要数据的更多信息以处理。拟人化讲，这是一种“创造性”的要求。所以对于画面生成、聊天机器人等任务会更多地要求这一特性。

<img src="image-20210421103846487.png" alt="image-20210421103846487" style="zoom:25%;" />

### GAN: Basic

首先以动漫人脸生成任务为例。输入不同的分布向量，网络将会学习并输出对应的复杂分布输出y。

<img src="image-20210506112101758.png" alt="image-20210506112101758" style="zoom:25%;" />

在这一过程中，产生了**Discriminator**的概念。Discriminator，即对某输入进行判定，并输出一标量值。通常定义其输入0-1间的实数，数值越大表明结果越接近。这里的接近是以我们的任务目标来判定的。例如对于动漫人脸生成，如果机器生成的图和目标的动漫人物图风格相似，那么分数将会较高。

<img src="image-20210421110835898.png" alt="image-20210421110835898" style="zoom:25%;" />

#### 基本思路

***仿生学角度：演化***

通过generator进行生成，再通过discriminator进行判别筛选，实际上是一种仿生学过程。首先discriminator判定生成的东西好坏，接下来generator通过这一判定来改进自己的生成结果，再用discriminator进行判别，如此循环。这也是GAN中对抗**adversarial**的由来。但在机器学习任务里是一种合作关系。

<img src="image-20210421111142097.png" alt="image-20210421111142097" style="zoom:25%;" />

#### 训练流程

<img src="image-20210421112205688.png" alt="image-20210421112205688" style="zoom:25%;" />

GAN的训练流程为：首先固定generator，训练discriminator对generator生成的内容和目标内容进行判定。此时discriminator对从数据库中选取的目标内容给予高分，而对生成内容给出低分；接下来固定该discriminator，将generator更新，其发现discriminator的偏好，修改自己的参数，以使得生成结果能够获得高分。如此往复多次，最终完成训练。

在新世代，GAN的效果可以以假乱真，不仅是针对动漫图片，对真实人脸也可达到类似效果。另外，对于某两角度不同的人脸，甚至可以输出其中间的变化：从左脸转动到右脸间的所有图片。

<img src="image-20210421115422649.png" alt="image-20210421115422649" style="zoom:25%;" />

### Theory behind GAN

GAN的目标是让generator生成的内容与真实数据的分布尽量接近：

<img src="image-20210423220403892.png" alt="image-20210423220403892" style="zoom:25%;" />

而这一系列分布间的衡量方式就是所谓的散度**divergence**。如果无法写出原始数据分布$P_{data}$和生成数据分布$P_G$，从对散度的定义来说，取出这些分布中的数据进行计算同样可以衡量分布的相似性。

我们先从discriminator的分类任务看起。其所做即为将来自两组不同数据集的抽样尽量分开。对于这样一个discriminator（记作D），要求输入的数据来自原始数据集时有较大分数，来自生成数据集时分数变小。最终定义出衡量数据相似性的目标函数$V(G, D)$，训练目标变为最大化这一函数。

<img src="image-20210423221018513.png" alt="image-20210423221018513" style="zoom:25%;" />

由需要可以定义出其中一种目标函数式如上，其实际上和二元分类的交叉熵定义相同。实际上$\max_DV(G, D)$和JS散度相关（证明忽略），所以求最大值后这一结果就变为了我们需要的散度。

既然得到了散度的结果，将这一结果代入先前对generator的优化函数里头，我们就得到了如下的式子：

<img src="image-20210423221502325.png" alt="image-20210423221502325" style="zoom:25%;" />

可以发现，这是一个minimax问题。所以需要先让D的目标函数最大，即红框部分最大。接下来固定D，更新G，即为在D的基础上优化G的目标函数。这实际上来自于minimax优化问题的思想，即先求出最大值，再对这个最大值求最小。当然其中还涉及到对偶理论的问题，就不再赘述了。需要注意的是，由这一训练过程，我推算出GAN的训练过程在每一轮对抗中是对同一批数据进行的。比如，将某一批真数据和某一批生成数据送入D，并让D的目标值最大；再将这一批生成数据和标签送入G，进行梯度下降修改G的参数。由此才可完成一轮训练。

### Tips for GAN

因为特殊的优化问题存在，所以GAN其实很难训练。

#### 目标函数的问题

实际上，在高维空间下，原始图像分布$P_{data}$和生成器分布$P_G$的重叠度并没有想象的高。随着维度增长，这类分布可能存在于高维空间中的子空间，从而难以重合（比如二维空间里头的两条线）。另外，抽样时可能无法充分，从而导致分布重叠的部分并未被完全考虑。

使用前述的JS散度时，低重叠度带来的恶果是，如果两个分布完全不重叠，那么计算得到的散度永远是固定常数。按照网络训练的想法，应该从两个不相似的分布逐渐靠近，这一过程在目标函数上体现为数值变化。然而当分布没有重叠（却在互相接近）的过程中，JS散度不变从而梯度下降没有任何帮助。

<img src="image-20210424210049583.png" alt="image-20210424210049583" style="zoom:25%;" />

**Wasserstein Distance**

面对这一问题，改进目标函数成为当务之急。一种做法是定义衡量两个分布距离的函数。其中，将某一分布P移动到另一分布Q所需要的距离（开销）被定义为wasserstein distance。

<img src="image-20210424210648203.png" alt="image-20210424210648203" style="zoom:25%;" />

实际上因为数据的不同，这一分布有多重计算方法。为计算方便，定义将两个分布移动到相似位置的最小距离为所谓的wasserstein distance。这样一来，将两个分布从远处拉近，这一过程就可以体现在目标函数之中。当然在实现中，实际上只是真实数据集的均值和生成数据集的均值差。

#### WGAN

使用上述距离，WGAN就产生了。

<img src="image-20210424210928361.png" alt="image-20210424210928361" style="zoom:25%;" />

WGAN的目标函数定义和前述JS散度的类似，对真实数据目标分数最大化，生成器数据分数最小化。注意到这里的判别函数D有$1-Lipschitz$的限制，即表示有平滑度的要求。当两个不同数据集接近时，为表现出分数差别，D的取值会达到$\pm\infin$，这就违背了平滑度要求。这一限制可以使得两个数据集接近时，D必须较平滑，从而对两个数据集输出的分数相类似；而数据集远时，D可以有较大的波动。

为实现这一限制，一般有如下三种手段：

<img src="image-20210424211506386.png" alt="image-20210424211506386" style="zoom:25%;" />

1.截尾。通过对权重限制，使得函数空间不会过于曲折，但效果一般；

2.梯度惩罚。让两个不同数据集间的判别函数梯度近似为1，也是一种平滑方式，当数据集分布接近时，该函数将被强制平滑，从而对两个数据集输出类似的值；

3.谱正则化。让梯度的范数处于某个限制。

***由于GAN的目标是在生成器和判别器之间对抗，所以一旦生成器无法骗过对方/判别器无法找到生成的内容，模型将不再能够更新。所以它的训练过程仍然充满困难。***

#### GAN与序列生成

在序列生成任务中，decoder相当于一个generator，其输出的内容被送入discriminator进行判定。但此时，生成结果是通过概率挑选后留下的one-hot向量，因此在decoder上的微小改变仅对其输出的概率值产生影响，对最终输出的one-hot值并不会产生影响，从而该过程无法进行微分。要解决这一问题，只能通过设计输出函数或使用强化学习的方法来解决。

<img src="image-20210506110610890.png" alt="image-20210506110610890" style="zoom:25%;" />

### Evaluation of GAN

#### Quanlity of Image

对GAN的评估一直是老大难问题。最开始的论文和实验均为人眼比较：这个模型的图片生成效果比其他人的好，所以它就好。但这种方法显然不科学。

<img src="image-20210506113734133.png" alt="image-20210506113734133" style="zoom:25%;" />

如果生成的任务是不同类别图，那么可以用某强大的影像分类器，观察输出的标签分布。如果生成图片的标签分布比较集中，说明生成的结果是确定的，即模型输出清晰可靠。反之，若标签分布平均，则有可能生成了“四不像”。

#### Diversity

与quality不同，diversity不关心某张图片的类别是否突出，而关心生成的不同图片是否能够类别充足。如果对于不同的输入分布，生成的图片类型分布均值是一种平均分布，即说明生成的结果多样化，否则就说明网络只记住了一种骗过判别器的图片并大量生成，我们在下面详细介绍。

<img src="image-20210506114801260.png" alt="image-20210506114801260" style="zoom:25%;" />

**Model Collapse**

因为生成器只需要骗过判别器，所以对于某输入，其只要能够产生够相近的输出就足够。这样一来，生成器一旦发现某些图片能骗过判别器，就有可能不思进取，只生成类似的图片。这一情况在输出结果中可以被观测到。

<img src="image-20210506114350521.png" alt="image-20210506114350521" style="zoom:25%;" />

**Mode Dropping**

如果生成的数据分布和真实数据分布接近，但不代表全集，就会产生mode dropping的问题：

<img src="image-20210506114550311.png" alt="image-20210506114550311" style="zoom:25%;" />

如图，某轮的生成图片看似无害，但下一轮发现肤色发生了改变。这些图片欺骗人眼已经足够，它们是一种分布的不完全代表，且难以通过输出的方式识别。

**FID**

动漫人脸和图片生成的类别相比更难产生上述的不同class标签。一种更精进的生成质量判别方法是FID。其主要思想为提取生成过程的图片隐变量与真实图片进行分数衡量。

<img src="image-20210506115541359.png" alt="image-20210506115541359" style="zoom:25%;" />

***We don't want memroy GAN***

在生成的过程中，模型会发现简单地记住训练图片或者将其翻转就可以获得类似的FID分数或骗过上述的判别标准。这一问题仍然有待研究。

### Conditional GAN

Conditional GAN的生成器接受输入为分布z和某种条件x，令输出的结果为符合该条件x的y。例如对于动漫人脸生成，输入人脸描述x和分布z，输出的图片y需要符合描述x的内容：

<img src="image-20210506112322253.png" alt="image-20210506112322253" style="zoom:25%;" />

Conditional GAN的情况下，不仅生成器需要额外输入条件x，判别器同样需要x的帮助，以使得生成器的结果y能够被判定好坏（否则虽然生成的y比较真实，却不一定符合条件x）。

<img src="image-20210506112455959.png" alt="image-20210506112455959" style="zoom:25%;" />

事实上这种额外条件x类似于一种监督学习。而普通的监督学习在输入分布z时，因为对应的输出多样，可能产生模糊的结果（如模糊的图片）。但使用GAN时其结果又因为GAN的创造力从而变得“天马行空”，所以还可以将GAN和监督学习结合，以消除这一创意性。



<img src="image-20210506112731608.png" alt="image-20210506112731608" style="zoom:25%;" />

如图所示，这类通过某设计图生成真实图片的模型虽然需要创意，但需要的是忠于原始设计图，创意体现于从设计图到真实图的过程，而非对真实的图片胡乱产生。因此使用监督学习的方法来限制它的能力，以使得结果真实可靠。

另外一个有意思的例子是声音-图片生成。给定一段水声，生成流水图片。通过调大声音，发现水花变大，说明模型并非单纯记住这一段声音（标签）和图片（输出），而是真正学习到了水声和图片的关系。

<img src="image-20210506113032922.png" alt="image-20210506113032922" style="zoom:25%;" />

### Learning from Unpaired Data

现实中有一大类任务不存在标签。如从真实人脸到二次元人脸的映射，真实人脸显然不可能存在对应的二次元人脸（标签），此时需要利用GAN的思想对数据分布进行建模：

<img src="image-20210506152703219.png" alt="image-20210506152703219" style="zoom:25%;" />

和先前的任务类似，但此时输入的分布并不是某个数学分布，而是人脸图片向量。在此，其被抽象为人脸这一数据的分布。同样地，输出的分布就是二次元头像图片向量。

此时同样存在问题：没有考虑输入的分布。如果网络认为输入就是一个噪声，输出一个二次元头像，这个头像与输入的人脸没有关联，但同样能够骗过discriminator。

#### CycleGAN

为解决这一问题，提出Cycle GAN。顾名思义，该模型的基本思路为组织一个循环的GAN，即输入人脸→转为二次元头像→反向GAN转为原始的输入人脸。最终衡量损失时对反向转换的人脸和输入的人脸进行相似度对比。

<img src="image-20210506153158882.png" alt="image-20210506153158882" style="zoom:25%;" />

对这一思想的朴素解释是，二次元头像有一定的特征时才能被转换回原始人脸。通过这一方式，模型真正对人脸数据进行了考虑。并且这一方法对人脸→头像的任务可做，对头像→人脸的任务同样可做，因为拥有两个生成器和判别器。所以此时它们构成了一个cycle。

**Text Style Transfer**

利用这一思路，GAN同样可以进行文法风格迁移的任务。例如将负面句子通过转换映射为正面的句子。

<img src="image-20210506153649257.png" alt="image-20210506153649257" style="zoom:25%;" />

类似地，无监督的GAN还可以进行文章→摘要转换、无监督翻译、无监督语音转文字等任务。

