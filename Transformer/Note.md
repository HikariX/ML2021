# Transformer

## Quick Introduction of Batch Normalization

该节实际上是Deep Learning一课中的训练技巧介绍，因课程需要在此处讲授。

### Changing Landscape

以二维参数的网络为例，两个参数组合的error surface可以是任意形状，例如椭圆。该情况说明输入$x_1$较小，移动其并不会对损失函数$L$造成极大影响，而输入$x_2$值较大，则同样微小的$\Delta w_2$会令产生的$\Delta L$较大，因此产生一种两个坐标轴尺度不同的error surface，如下图。

<img src="image-20210409201237963.png" alt="image-20210409201237963" style="zoom:25%;" />

如果将输入的尺度规范到同类区间，则其对error surface的改善体现为绿色等高线图，此时两个方向的参数优化均有同等水平的影响，从而不会出现不均衡的下降。

### Feature Normalization

对于序列输入，将它们同样位置的值求均值与方差而进行标准化，即称为feature normalization。根据以上分析可知，这种标准化过程能够让梯度下降过程更快收敛。

<img src="image-20210409201751534.png" alt="image-20210409201751534" style="zoom:25%;" />

### Normalization in Deep Learning

在深度学习中，因为数据量众多，因此一般不可能对所有数据进行标准化。实际中是针对一批训练数据进行该操作。当批数据量合适的时候，它们的均值和方差分布被认为可以代表全体数据。这就是batch normalization （BN）的由来。

DL中的BN不仅需要对数据输入$x_i$进行，其对应的某层输出$z_i$也可以被看作下一层的输入，而它们在经过神经网络层后已经产生了偏移，所以对它们执行BN同样有利于下一层网络的训练。BN操作可以放在激活函数前/后，实际效果均较好。虽然大部分激活函数在0附近才有激活的效果，但是实测发现BN的顺序不太影响效果。

另外，虽然将数据标准化，但我们仍希望让网络来决定这一程度。所以BN后的数据实际上是标准化数据和超参数$\gamma$、$\beta$的结合，前者控制方差，后者控制均值。注意它们的维度和输入是一致的。

<img src="image-20210409202001138.png" alt="image-20210409202001138" style="zoom:25%;" />

#### 测试集怎么办

在进行测试的时候，不可能等到一批数据后再输出结果，所以在训练时BN就已经将得到的均值与方差进行了滑动平均：

<img src="image-20210409202932501.png" alt="image-20210409202932501" style="zoom:25%;" />

通过来自大量训练集的参数，可以为测试集提供同样好的结果。

#### 效果

因为BN改变了error surface，所以梯度下降的过程能够得到加速。实验证明加上BN可以显著提升训练速度。另外，因为平滑的error surface，较大的学习率同样能够有效训练，如下图蓝色实线（然而蓝色虚线学习率比它低反而效果好，深度学习的不可解释性- -）和红色虚线的对比。此途中粉色线为Sigmoid函数，就算搭配BN也无法较好训练，所以实际中尽量少用。

<img src="image-20210409203049505.png" alt="image-20210409203049505" style="zoom:25%;" />

### Internal Covariate Shift

DL中还有一个概念称为Internal Covariate Shift，即，当优化网络参数的时候，比如对矩阵A进行更新为A'，且对B更新为B‘。但存在一种可能：B'的更新是以A为基准的，所以A'的产生会让B‘的更新“不够匹配”。有人认为，BN的存在让中间输入/输出$\boldsymbol{a}$和$\boldsymbol{a}'$的分布接近，从而缓解了参数矩阵优化时候的偏移。

<img src="image-20210409203517323.png" alt="image-20210409203517323" style="zoom:25%;" />

然而在*How Does Batch Normalization Help Optimization?*（https://arxiv.org/abs/1805.11604）一文中，作者对BN的原理进行了分析和实验，输出中间变量发现该现象并不存在。即添加BN不会导致$\boldsymbol{a}$和$\boldsymbol{a}'$的分布有多接近，所以BN的效果好可以说是一种意外之喜。另外，该文章在理论和实验上都证明了BN能够平滑化error surface。

## Transformer

### Seq2Seq

提到Transformer，无法绕开的就是sequence to sequence（Seq2Seq）模型。该模型的输出长度由模型自己决定，这一特性恰好符合语音辨识、机器翻译、语音翻译之类任务。另外有大量额外的工作皆可由它解决。

<img src="image-20210409222430280.png" alt="image-20210409222430280" style="zoom:25%;" />

**Speech**

老师给出其实验室的一个范例：台语-中文辨识。台语因为没有文字，因此从语音辨识再到机器翻译的做法并不可取，因此需要尝试直接从语音到翻译。使用网上的肥皂剧（配音台语，字幕中文），即可获取大量训练资料。此处不关心输入语音的背景杂音、配乐，不关心字幕和配音的时间对齐问题，最终确实得到一个可用的模型。虽然这看起来像是强行训练，但也验证了端到端的DL模型之强大。

<img src="image-20210409222723188.png" alt="image-20210409222723188" style="zoom:25%;" />

将台语与中文的映射顺序颠倒，后者当输入，前者做输出，就可以得到文字转语音合成系统，同样出自该实验室。

**Chatbot**

聊天机器人（chatbot）通过接收上下文对话，可以学习一种应答机制。这种输出长度不限的问题就可以通过seq2seq解决。

***Question Answering***

实际上，该类问题被称为问答。例如翻译、文章摘要提炼、文章内容属性判断等。最极端下只有一个输出，但seq2seq均能够胜任。

<img src="image-20210409223655604.png" alt="image-20210409223655604" style="zoom:25%;" />

**Syntactic Parsing**

对于语句，其可以被拆解成语法树的结构。如果将其写作一串字符如下，那么我们同样获得了对应的输出序列，从而可以去训练一种语法分析器。

<img src="image-20210409223836022.png" alt="image-20210409223836022" style="zoom:25%;" />

实际上这串字符（树）的产生类似于编译原理的分析过程，此处不详述。

**Multi-label Classification**

对于多标签问题，某个输入（如文档）可能拥有不同的类别，可以由seq2seq模型自行决定划定的标签数量。

<img src="image-20210409224012886.png" alt="image-20210409224012886" style="zoom:33%;" />

实际上目标检测问题也是一种多标签问题。

### Encoder

Seq2seq模型分为encoder和decoder两部分。

<img src="image-20210409224211570.png" alt="image-20210409224211570" style="zoom:25%;" />

Encoder为给定长度序列，输出相同长度序列的机构，致力于提取足够多的输入序列信息。

<img src="image-20210409224327420.png" alt="image-20210409224327420" style="zoom:25%;" />

Encoder的结构中细分为多个block，每个block内含self-attention模块，因此它能够在block的堆叠中积累序列前后的数据信息。

<img src="image-20210409224444165.png" alt="image-20210409224444165" style="zoom:25%;" />

实际上，在attention层后，输出会和其对应的输入作加，这也叫做**残差（residual）**机制。接下来该合成输出会经过layer norm，对输入序列这一维度（而不是序列间）作标准化，即对每一个序列的不同维度合起来标准化。另外，输出后的结果进入全连接层后，同样要经过residual和layer norm。这就是Transformer中的Add&Norm层。

<img src="image-20210409224617091.png" alt="image-20210409224617091" style="zoom:25%;" />