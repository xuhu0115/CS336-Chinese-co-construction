# 第十二章：评估与基准测试

评估的核心问题是**给定一个固定的模型，它到底有多“好”？** 这看似是一个简单的打分问题，实则是一个深刻且复杂的系统性工程。评估不仅决定了我们如何衡量当前模型的性能，更在根本上塑造了未来模型的发展方向。本章将从你所见的表象出发，深入剖析评估的本质、方法、挑战与陷阱。


## 12.1 简介

当你打开任何一个大语言模型（LLM）的评测网站或论文时，你首先看到的是什么？

### 基准测试分数 (Benchmark Scores)

这是最直观、最普遍的评估形式。各大模型发布时，都会在一系列标准化基准上报告其得分。例如：

<div align="center">
   <img src="images/12-1-Deepseek-R1的基准性能.png" />
   <p>图12.1 Deepseek-R1的基准性能</p>
 </div>

[Deepseek-R1论文 Figure 1 报告的基准性能](https://arxiv.org/pdf/2501.12948)，展示了 Deepseek-R1 在AIME 2024、Codeforces、GPQA Diamond、MATH 500等基准上的性能

<div align="center">
   <img src="images/12-2-llama4的基准性能.png" />
   <p>图12.2 llama4的基准性能</p>
 </div>

[Llama 4 Behemoth 指令微调版本的基准性能](https://ai.meta.com/blog/llama-4-multimodal-intelligence/)，在其技术报告里展示了其在 MMLU-Pro、MATH500、GPQA 等多个基准上的表现。

<div align="center">
   <img src="images/12-3-OLMo的基准性能.png" />
   <p>图12.3 OLMo的基准性能</p>
 </div>

[Allen AI 团队在技术报告中展示的 OLMo-2-32B 模型的基准性能](https://allenai.org/blog/olmo2-32B)，主要在 MATH、MMLU、DROP 和 GSM8k 等基准上进行评估。

大部分语言模型都会在大致相同的基准上进行评估，但它们并非完全一致。那么，这些基准究竟是什么？这些数字又意味着什么？

下面是一个来自于 [HELM](https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard)网站的例子，他展示了不同模型在多个基准测试上的性能排行：

<div align="center">
   <img src="images/12-4-HELM-capibities.png" />
   <p>图12.4 不同模型在 HELM 网站上的性能排行榜</p>
 </div>

 除了上面的介绍之外，国内也有很多类似的大模型榜单。OpenCompass是由上海人工智能实验室（上海AI实验室)于2023年8月正式推出的大模型开放评测体系：

<div align="center">
   <img src="images/12-5-上海人工智能实验室opencompass.png" />
   <p>图12.5 不同模型在思南 OpenCompass 网站上的性能排行榜</p>
 </div>

[SuperCLUE](https://superclueai.com/homepage) 是国内权威独立第三方 AI 测评分析机构，源于2019年10月发起的中文测评基准 CLUE 开源社区。

<div align="center">
   <img src="images/12-6-SuperCLUE.png" />
   <p>图12.6 不同模型在 SuperCLUE 网站上的性能排行榜</p>
 </div>


### 成本与性价比

当然，评估不能只看能力，成本和推理速度也是评价的关键维度。另外一个例子是来自于 [Artificial Analysis](https://artificialanalysis.ai/) 网站，从智能、推理速度、价格三个角度测评了不同模型：

<div align="center">
   <img src="images/12-7-Artificial Analysis.png" />
   <p>图12.7 不同模型在 Artificial Analysis 网站上的性能排行榜</p>
 </div>

像 Artificial Analysis 这样的网站，就将模型的性能与每 token 的成本结合起来，绘制出帕累托前沿（Pareto Frontier）。这揭示了一个现实：顶级模型（如 O3）虽然强大，但价格昂贵；而一些排名稍后的模型，可能在性能和成本之间取得了更好的平衡。智能纵坐标的智能指数包含以下测试： MMLU-Pro、GPQA Diamond、Humanity's Last Exam、LiveCodeBench、SciCode、AIME 2025、IFBench、AA-LCR、Terminal-Bench Hard 和 𝜏²-Bench Telecom 。

<div align="center">
   <img src="images/12-8-Intelligence vs Cost to Run Artificial Analysis Intelligence Index (12 Dec &apos;25) .png" />
   <p>图12.8 不同模型在 Artificial Analysis 网站上的性能 vs 成本对比</p>
 </div>

###  用户选择与市场反馈 

另一种“评估”是看用户实际选择了谁。[OpenRouter](https://openrouter.ai/rankings) 网站通过路由流量到不同的模型，积累了大量关于用户偏好的数据。根据发送到每个模型的 token 数量，可以生成一个“使用量”排行榜。这个榜单显示，OpenAI、Anthropic、Google 以及国内的 Deepseek 和 Qwen 模型目前占据主导地位。这表明，在真实世界中，“被广泛使用”本身就是一个强有力的“好”的指标。

<div align="center">
   <img src="images/12-9-不同厂商token使用排行榜.png" />
   <p>图12.9 不同厂商token使用排行榜</p>
 </div>

除以上之外，还有一个很受欢迎的评测榜单是[Chatbot Arena](https://huggingface.co/spaces/lmarena-ai/lmarena-leaderboard)。它是由国际开放研究组织 LMSYS Org 推出的大模型竞技评测平台，最大特点是采用**匿名盲测**机制随机配对模型，通过**用户投票**结合 Elo 评分系统量化模型能力。该平台由加州大学伯克利分校、圣地亚哥分校和卡内基梅隆大学研究人员联合开发，支持多轮对话与针对性榜单评测。

<div align="center">
   <img src="images/12-10-chatbot-arera.png" />
   <p>图12.8 不同模型在 SuperCLUE 网站上的性能排行榜</p>
 </div>

### 主观评价与社区口碑

最后，还有来自社交媒体（如 X 平台）的“感觉”（vibes）。人们会分享模型令人惊叹的案例：“看，它居然能这样做！”这些非正式的、感性的评价，也是评估模型能力的一个侧面来源。

[社交媒体上关于 Gemini 2.5 Pro Preview的使用分享](https://x.com/demishassabis/status/1919779362980692364)

<video src="https://x.com/demishassabis/status/1919779362980692364"></video>

然而，正如 [Andrej Karpathy](https://x.com/karpathy/status/1896266683301659068?s=20) 所指出的，当前我们正面临一场“评估危机”。

<div align="center">
   <img src="images/12-11-Andrej Karpathy关于当前评估问题的看法.png" />
   <p>图12.11 Andrej Karpathy关于当前评估问题在x平台上发表的看法</p>
 </div>

> 翻译：我的感受是，当前正面临一场评估危机。我现在真的不知道该看哪些指标了。
MMLU 在过去几年里是个不错且有用的基准，但那段时期早已结束。
SWE-Bench Verified（基于真实、实用且经过验证的问题）我非常喜欢，也很出色，但它本身的覆盖范围又太窄了。
>Chatbot Arena 被过度关注（部分或许是我的责任？），以至于大模型实验室开始严重过拟合它——手段包括从 API 请求中挖掘提示词（prompt mining）、用大量私有评估集反复测试，更糟糕的是，甚至直接将排行榜排名用作训练监督信号。我觉得它目前勉强还能用，毕竟还没有明显“更好”的替代方案，但其信号质量似乎正在下滑。
>现在出现了一些私有的评估集，将它们组合成一个集成评估体系，或许是一条有希望的前进方向。
在缺乏优秀综合性评估手段的情况下，我曾尝试转向“感觉检查”（vibe checks），但现在我担心这种做法具有误导性——容易带来确认偏误、样本量太小等问题，实在不够可靠。
>
>简而言之，我的感受是：我现在真的不太清楚这些模型到底有多好了。

许多主流基准（如 MMLU）之前看起来不错，但现在人们认为要么已经饱和，要不被刷榜了，或者两者兼有。Chatbot Arena 也因商业利益而受到质疑。因此，我们手握海量数据和排名，却缺乏一套清晰、可靠、公认的评估标准。

## 12.2 如何看待评估

评估绝非一个机械化的脚本过程（输入提示词 -> 得到输出 -> 计算平均分）。它是一个需要深思熟虑的框架，其设计必须服务于一个明确的目标。

> 核心原则：没有唯一的“正确”评估。

评估的意义取决于你想回答的问题：

- 对于终端用户/企业：我应该采购哪个模型来满足我的具体需求（如客户服务聊天机器人）？
- 对于研究人员：我们是否在模型的原始能力（如“智能”）上取得了科学进步？
- 对于政策制定者/安全机构：当前模型带来的益处和风险分别是什么？现状如何？
- 对于模型开发者：我该如何改进我的模型？哪些干预措施有效？

为了将抽象目标转化为具体评估方案，我们需要思考四个关键环节：

#### 输入是什么？

- 提示词从何而来？覆盖了哪些使用场景？
- 我们是否在长尾分布（即那些出现频率很低、不常见但真实存在的输入情况，如在语言模型中，可能包括罕见领域的问题、复杂推理任务、边缘语言现象、对抗性或歧义性较强的语句等）中包含了具有代表性的困难输入样本？
- 输入是否需要适配模型？例如，在多轮对话中，用户的下一条输入依赖于模型的上一条回复，这使得评估变得动态化。

#### 如何调用模型？

- 如何提示模型？是零样本、少样本还是思维链（CoT）？
- 是否允许模型使用外部工具（如代码解释器、RAG 检索增强生成）？
- 我们是在评估语言模型本身，还是评估一个由模型和代理框架（agent scaffolding）组成的完整系统？开发者关心前者，而用户关心后者。

#### 如何评估输出？

- 参考答案是否准确无误？
- 使用什么指标？对于代码生成，是 pass@1（模型生成一个回答，输出回答和标准答案一样即为正确）还是 pass@10（模型一次生成十个回答，只要十个输出回答里有一个和标准答案一样即为正确））？
- 如何考量成本？模型性能不应只看准确率，还要考虑计算成本。例如：一个更大但仅略优的模型，是否值得其高推理延迟或训练开销？在资源受限场景（如移动端、低延迟服务），成本是关键约束。
- 如何处理不对称错误？某些错误的代价远高于其他错误。在医疗、金融、法律等高风险领域中，幻觉（hallucination）的代价远高于其他错误。
- 如何评估开放域生成任务？当没有标准答案时（如“写一个关于斯坦福的精彩故事”），评估变得极其困难。常见做法：
  - 使用人工评估（成本高但可靠）；
  - 用 LLM-as-a-judge（让另一个大模型打分）；
  - 基于多样性、流畅性、相关性等维度设计 Proxy metrics（代理指标，是指在无法直接测量目标量（target quantity）时，用来间接反映或近似该目标的可测量指标）；
  - 通过用户反馈或 A/B 测试衡量实际效用。

#### 如何解读结果？

* 一个 91% 的分数意味着什么？它是否足够好，可以部署给真实用户？
* 如何判断模型是否真正掌握了某种泛化能力，而不是仅仅记住了训练数据？
* 最重要的是，我们到底在评估什么？是最终的产品模型、一个完整的系统，还是底层的研究方法？

忽略这些问题，仅凭一个分数做判断，是评估中最大的误区。

## 12.3 困惑度

### 12.3.1 什么是 Perplexity（困惑度）？

在深入具体的下游任务基准之前，我们必须理解一个基础且重要的度量标准：困惑度（Perplexity）。

语言模型的本质是一个概率分布 p(x)，它对任意一段 token 序列 x 给出一个概率值 —— 即这个序列“有多大概率被模型生成”。困惑度衡量了模型对某个数据集分配高概率的能力。在预训练阶段，模型的目标就是最小化训练集上的困惑度。数值越小越好 —— 表示模型认为这些数据“很自然”，预测能力强。

$$
\text{Perplexity} = \left( \frac{1}{p(D)} \right)^{1/N}
$$

其中 $p(D)$ 是模型给整个数据集 $D$ 中所有句子联合概率的乘积（通常取 log 求和再指数还原）；$|D|$ 是数据集中 token 的总数量。

### 12.3.2 什么要用测试集？

在预训练阶段，我们通过最小化训练集上的困惑度来优化模型参数。但为了评估模型是否真的“学会了语言”，我们必须在 **未见过的数据（测试集）** 上测量困惑度，这才是模型泛化能力的体现。

> ⚠️ 注意：不能只看训练集困惑度，否则容易过拟合！

### 12.3.3 经典基准数据集（Standard datasets）

这些是历史上用于评估语言模型的经典数据集：

| 数据集 | 特点 |
|--------|------|
| **Penn Treebank (PTB)** | 小规模，华尔街日报语料，常用于早期 RNN/LSTM 实验 |
| **WikiText-103** | 基于维基百科的大规模英文语料，更贴近真实语言分布 |
| **One Billion Word Benchmark (1BW)** | 来自机器翻译数据集（欧洲议会、联合国文件、新闻），词汇量大，挑战性强 |

> 🎯 这些数据集曾是衡量模型性能的“黄金标准”。


### 12.3.2 历史与演变

2016 年，Jozefowicz 等人在[《Exploring the Limits of Language Modeling》](https://arxiv.org/abs/1602.02410) 论文里，用纯 CNN + LSTM 架构，在 十亿（1B） Word Benchmark 上把困惑度从 51.3 降到 30.0 —— 当时是非常大的突破。

<div align="center">
   <img src="images/12-12-CNN+LSTM 架构显著降低困惑度.png" />
   <p>图12.11 CNN+LSTM 架构显著降低困惑度</p>
 </div>

OpenAI 在 [《Language Models are Unsupervised Multitask Learners》](https://cdn.openai.com/better-language-models/) 论文中发布的 GPT-2，在 WebText（来自 Reddit 链接的网页文本，约 40GB）上训练，然后直接在标准数据集上做 zero-shot（零样本）评估。
这属于“分布外评估”（out-of-distribution），因为训练数据和测试数据来源不同。但它表现很好，说明大规模、多样化的训练数据能带来强大的泛化能力。

<div align="center">
   <img src="images/12-13-GPT-2的zero-shot（零样本）评估.png" />
   <p>图12.13 GPT-2的zero-shot（零样本）评估</p>
 </div>

这是 GPT-2 论文中的关键表格，展示了不同规模模型在多个任务上的表现：

🔍 发现：
- 随着模型变大，困惑度持续下降 → 表明更大的模型能更好地拟合语言分布。
- 在某些任务上（如 LAMBADA），准确率也随模型增大而提高 → 说明语言建模能力迁移到下游任务是有益的。
- 但在 1BW 上，大模型改进幅度较小 → 说明该数据集难度更高或已接近极限。

### 12.3.3 为什么困惑度仍重要？

自 GPT-2 和 GPT-3 以来，语言建模论文的研究方向已经更多地转向下游任务的准确性。但困惑仍然有用的原因如下：

- 更平滑，适合拟合缩放定律（scaling laws）.下游任务准确率往往波动大、有天花板，而困惑度变化连续，便于研究模型性能如何随参数量/数据量增长。
- 通用性强，适用于任何文本.不像特定任务需要标注数据，困惑度可以在任何文本上计算 —— 更适合作为“基础能力”的度量。
- 可扩展到条件困惑度（conditional perplexity）.比如在问答任务中，固定问题部分，只计算答案部分的困惑度 —— 也能反映模型在该任务上的语言建模能力。

📌 所以，困惑度不仅是训练目标，也是分析模型内在能力的重要工具。

### 12.3.4 使用困惑度的陷阱

如果你要做一个排行榜（leaderboard），评估任务准确率很容易：只需要模型输出结果，人工或自动打分即可。但评估困惑度则要求模型必须输出每个 token 的概率分布，并且这些概率加起来必须等于 1（归一化）。这依赖于模型内部机制，存在信任问题：
- 如果模型偷偷作弊（比如输出虚假概率），你就无法检测。
- 早期模型遇到未知词（UNK）时处理不当，也会导致概率不准确。

🧩 所以，困惑度更适合研究者自己控制模型架构和训练过程的场景，不适合完全黑盒的公开评测。

### 12.3.5 “困惑度最大主义者”的哲学观点

这是一种理想主义/理论派的观点，他们假设真实世界语言分布是 t，你的模型是 p。最优困惑度就是真实分布的熵 H(t) —— 只有当 p = t 时才能达到。如果你完美建模了真实分布 t，那你就能解决所有语言相关任务（因为任务只是分布的一部分）。因此，`不断降低困惑度 → 最终逼近真实分布 → 实现通用人工智能（AGI）`。

这听起来很酷，但值得注意的是，现实中，我们并不需要完美建模整个语言分布，很多“边缘”或“无关紧要”的部分对人类任务没用。一味追求最低困惑度可能是资源浪费。

### 12.3.6 一些能够反映模型是否真正‘理解’了语言的下游任务

既然困惑度衡量的是模型对语言分布的拟合能力，那么有没有一些具体的、有挑战性的下游任务，能反映模型是否真正‘理解’了语言？

#### 完形填空类任务

完形填空类任务就是给定一段上下文（context），让模型预测其中一个被遮盖的词（target word）。这个任务要求模型不仅要理解局部语法，还要捕捉长距离依赖、语义连贯性、世界知识等。只关注某个关键位置的预测质量，而不是整个句子。

因此，完形填空类任务可以看作是“局部困惑度”。



来自 [LAMBADA](https://arxiv.org/abs/1606.06031) 数据集的三个具体例子 

<div align="center">
   <img src="images/12-14-LAMBADA完形填空任务的三个样例.png" />
   <p>图12.14 LAMBADA完形填空任务的三个样例</p>
 </div>


✅ 案例 1
> 原文上下文：“Yes, I thought I was going to lose the baby.” … “This baby wasn’t exactly planned for.”
目标句： “Do you honestly think that I would want you to have a _____ ?”
正确答案：miscarriage（流产）

> 中文翻译与理解：“是的，我当时以为我会流产。”……“这个孩子其实并不是计划中的。”
目标句：“你真的觉得我会希望你经历一次 _____ 吗？”
正确答案：流产

✅ 案例 2
> 原文上下文：“Why?” … “He was a great craftsman,” said Heather. “That he was,” said Flannery.
目标句： “And Polish, to boot,” said ______.
正确答案：Gabriel

> 中文翻译与理解：“为什么？”……“他是个了不起的工匠，”海瑟说。“确实如此，”弗兰纳里说。
目标句：“而且，他还是波兰人呢，”______ 说道。
正确答案：加布里埃尔（Gabriel）

✅ 案例 3

> 原文上下文：Preston had been the last person to wear those chains...
目标句： Sergei looked at me, surprised by my low, raspy please, but he put down the _____.
正确答案：chains（锁链）

> 中文翻译与理解：普雷斯顿是最后一个戴上那些锁链的人……
目标句：谢尔盖看着我，对我那低沉沙哑的恳求感到惊讶，但他还是放下了那条 _____。
正确答案：锁链

#### 常识推理任务

[HellaSwag](https://arxiv.org/pdf/1905.07830)是一个常识推理任务。通常是给定一个视频片段或一段文字描述（premise），然后给出四个选项（A, B, C, D），要求选出最符合常识、最自然、最可能发生的后续动作。它强调的是日常生活的常识推理（commonsense reasoning）和行为合理性判断。

<div align="center">
   <img src="images/12-15-HellaSwag常识推理样例.png" />
   <p>图12.15 HellaSwag常识推理样例</p>
 </div>

✅ 案例 1：给狗洗澡

> 一位女士正站在室外，手里拿着一只水桶，身边还有一只狗。那只狗四处奔跑，试图躲过洗澡。她……
>
> A. 用肥皂冲洗水桶，然后用吹风机吹干狗的头。
> B. 用水管防止它被弄湿。
> C. 把狗弄湿了，结果它又跑掉了。 ← 正确
> D. 和狗一起钻进浴缸里。

解析：现实中，狗通常讨厌洗澡，主人一泼水它就逃跑；所以“先弄湿 → 狗逃跑”是最常见、最自然的流程；A、B、D 要么违反常识（狗不可能乖乖吹头），要么操作不合理（用水管“防止”弄湿？）

✅ 案例 2：交通规则

> 在停车标志或红灯处必须完全停下……如果你在红灯前停下，请在灯变绿后继续行驶……
> 
> A. 停车时间不超过两秒……
> B. 完全停下后，关闭你的转向灯……
> C. 远离对向车流……
> D. 如果你所在车道有白色停止线，请在这条线前停下。等所有车辆通过后再穿过路口。 ← 正确

解析：D 描述的是标准交通规则：停在白线前，确认安全再通行；A 错误（红灯必须停够时间）；B 无关（转向灯在此场景不适用）；C 模糊（“远离对向车流”不是红灯场景的核心操作）。

HellaSwag 可以看作是“情境下的困惑度”，模型不需要输出概率，但它的选择应该反映出它对“哪个结局最可能”的内在概率估计。

## 12.4 知识类基准 

这类基准旨在衡量模型所掌握的事实性知识。

### MMLU (Massive Multitask Language Understanding)

[MMLU](https://arxiv.org/pdf/2009.03300.pdf) 包含 57 个学科（从数学、历史到法律、伦理）的多项选择题。问题源自网络，由学生收集。它更侧重于知识而非语言理解。

最初用 GPT-3 的少样本提示进行评估，最大型号的 X-Large 只能取得不到 0.5 的分数，现在最强的 LLM 在 MMLU 上最高分已达90+。

<div align="center">
   <img src="images/12-16-GPT-3在MMLU上的少样本提示.png" />
   <p>图12.16 GPT-3在MMLU上的少样本提示</p>
 </div>


当前顶尖模型（如 Geimini 3 Pro Preview）在 [HELM](https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard/mmlu_pro) 上可以看到，已达到 90.3% 的准确率。

### MMLU-Pro

[MMLU-Pro](https://arxiv.org/abs/2406.01574) 是对 MMLU 的改进版，移除了噪声大或过于简单的问题，并将选项从 4 个增加到 10 个。模型准确率显著下降，缓解了饱和问题。通常结合思维链（CoT）进行评估。

<div align="center">
   <img src="images/12-17-MMLU-pro对比MMLU.png" />
   <p>图12.17 MMLU-pro对比MMLU</p>
 </div>

### GPQA (Graduate-Level Google-Proof Q&A)

[GPQA](https://arxiv.org/abs/2311.12022) 由 61 名 PhD 通过 Upwork 平台设计的高难度问题。目标是创建“防谷歌”问题，即非专家即使花 30 分钟用谷歌搜索也难以解答。

- 博士级专家实现了 65% 的 accuracy
- 非专家在可访问谷歌的条件下，在 30 分钟内达到了 34% 的accuracy
- GPT-4 实现了 39%

<div align="center">
   <img src="images/12-18-MMLU-pro对比MMLU.png" />
   <p>图12.18 MMLU-pro对比MMLU</p>
 </div>

当前顶尖模型（如 Geimini 3 Pro Preview）在 [HELM](https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard/gpqa) 上可以看到，已达到 80.3% 的准确率，显示出巨大进展。

### Humanity's Last Exam

[Humanity's Last Exam](https://arxiv.org/abs/2501.14249) 是一个雄心勃勃的项目，包含 2500 道多模态、多学科的选择题和简答题。通过五十万美元奖金池激励社区贡献，并用前沿 LLM 筛选掉过于简单的问题，再经过多阶段审查。局限在于问题征集过程可能存在严重的选择偏差，且问题类型仍局限于有标准答案的“考试”形式。

<div align="center">
   <img src="images/12-19-Humanity&apos;s Last Exam 收集筛选流程.png" />
   <p>图12.19 Humanity's Last Exam 收集筛选流程</p>
 </div>

当前顶尖模型（如 Geimini 3 Pro Preview）在  https://agi.safe.ai/ 上可以看到，已达到 38.3% 的准确率。



## 12.5 指令遵循基准

这类基准评估模型是否“听话”，能否按照用户的要求进行输出。

### Chatbot Arena

[Chatbot Arena](https://arxiv.org/abs/2403.04132) 采用“盲测”和 ELO 评分系统。真实用户提交提示，同时收到两个匿名模型的回复，并选择更优者。优点是输入动态、能容纳新模型。问题在于评估者是网站访客，样本可能存在偏差；ELO 分数可能被策略性操纵。

<div align="center">
   <img src="images/12-20-Chatbot Arena 分数排行榜单.png" />
   <p>图12.20 Chatbot Arena 分数排行榜单</p>
 </div>


在其[排行榜单](https://huggingface.co/spaces/lmarena-ai/chatbot-arena-leaderboard)里我们可以看到，当前表现最好的是 Geimini 3 Pro ，Arena Score 达到了 1492 分。Arena Score 是一个经过统计校准的能力分数，其数值差异直接对应模型在人类偏好盲测中的胜率。

### IFEval (Instruction-Following Eval)

[IFEval](https://arxiv.org/abs/2311.07911) 使用可自动验证的约束（如“回答必须包含至少5句话”）来测试模型。优点是自动化程度高。局限是只评估约束遵守情况，不评估语义质量，且约束本身可能过于人工化。

<div align="center">
   <img src="images/12-21-IFEval包含指令的详细描述.png" />
   <p>图12.21 IFEval包含指令的详细描述</p>
 </div>

![Alt text](image-10.png)

在 HELM 的 [Leaderboard IFEval](https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard/ifeval) 上我们可以看到，当前前沿模型在IFEval 能达到 0.951 的准确度。 

<div align="center">
   <img src="images/12-22-HELM上不同模型在IFEval的排行榜.png" />
   <p>图12.22 HELM上不同模型在IFEval的排行榜</p>
 </div>

### AlpacaEval

[AlpacaEval](https://tatsu-lab.github.io/alpaca_eval/) 包括各种来源的 805 条指令。使用一个强大的 LLM（如 GPT-4）作为裁判，判断候选模型的回答是否优于 GPT-4 自身的回答，并计算胜率。优点是自动化程度高，能处理开放域回答。问题在于存在裁判偏见，且早期版本易被回答长度等表面特征欺骗。

<div align="center">
   <img src="images/12-23-不同模型在AlpacaEval上的排行榜.png" />
   <p>图12.23 不同模型在AlpacaEval上的排行榜</p>
 </div>


### WildBench

[WildBench](https://arxiv.org/pdf/2406.04770) 从100多万条真实人机对话中采样构建评估集，并使用 GPT-4-Turbo 作为裁判，配合检查清单确保评估维度的全面性。其结果与 Chatbot Arena 高度相关（相关系数 0.95），被视为新基准有效性的“事实上的”检验标准。

<div align="center">
   <img src="images/12-24-WildBench构建流程.png" />
   <p>图12.24 WildBench构建流程</p>
 </div>

在 HELM 的 [Leaderboard WildBench](https://crfm.stanford.edu/helm/capabilities/latest/#/leaderboard/wildbench) 上我们可以看到，当前前沿模型在IFEval 能达到 0.866 的准确度。 

<div align="center">
   <img src="images/12-25-HELM上不同模型在WildBench的排行榜.png" />
   <p>图12.25 HELM上不同模型在WildBench的排行榜</p>
 </div>

## 12.6 智能体基准

这类基准评估模型作为智能体（Agent） 的能力，即在复杂环境中通过工具调用和迭代规划完成任务。

### SWEBench

[SWEBench](https://arxiv.org/abs/2310.06770) 包含 12 个 Python 代码库中的 2294 个任务，根据 GitHub issue 描述，提交一个能通过单元测试的 Pull Request (PR)。评估直接运行单元测试来验证修复是否成功。

<div align="center">
   <img src="images/12-26-SWEBench评测流程示意图.png" />
   <p>图12.26 SWEBench评测流程示意图</p>
 </div>

### CyBench

[CyBench](https://arxiv.org/abs/2408.08926) 完成 40 个网络安全领域的“夺旗”（CTF）挑战。任务难度通过人类“首次解决时间”来度量，有些任务对人类而言也极具挑战性（耗时可达24小时）。

<div align="center">
   <img src="images/12-27-CyBench评测流程示意图.png" />
   <p>图12.27 CyBench评测流程示意图</p>
 </div>

<div align="center">
   <img src="images/12-28-不同模型在CyBench的排行榜.png" />
   <p>图12.28 不同模型在CyBench的排行榜</p>
 </div>

### MLEBenc:

[MLEBench](https://arxiv.org/abs/2410.07095) 自动化参与 75 个 Kaggle 机器学习竞赛，包括数据处理、模型训练、超参调优和结果提交。当前最佳模型获得任何奖牌的准确率也低于20%。

<div align="center">
   <img src="images/12-29-MLEBench评测流程示意图.png" />
   <p>图12.29 MLEBench评测流程示意图</p>
 </div>

<div align="center">
   <img src="images/12-30-不同模型在MLEBench的排行榜.png" />
   <p>图12.30 不同模型在MLEBench的排行榜</p>
 </div>

这些基准代表了 LLM 能力的最前沿，但当前模型的成功率普遍较低，表明它们距离真正的自主智能体仍有巨大差距。

## 12.7 纯推理基准

这类基准试图剥离知识，仅评估模型的抽象推理能力。

### ARC-AGI (Abstraction and Reasoning Corpus for Artificial General Intelligence)

[ARC-AGI](https://arcprize.org/arc-agi) 提供一系列视觉化的输入-输出网格对，要求模型推断出变换规则并应用于新的输入。整个过程不涉及任何语言。

ARC-AGI-1:

<div align="center">
   <img src="images/12-31-ARC-AGI-1评测示意图.png" />
   <p>图12.31 ARC-AGI-1评测示意图</p>
 </div>

<div align="center">
   <img src="images/12-32-不同模型在ARC-AGI-1的得分.png" />
   <p>图12.32 不同模型在ARC-AGI-1的得分</p>
 </div>

更难的 ARC-AGI-2:

<div align="center">
   <img src="images/12-33-ARC-AGI-2评测示意图.png" />
   <p>图12.33 ARC-AGI-2评测示意图</p>
 </div>
 
它捕捉了一种更纯粹的、类似人类的模式识别和泛化能力，是早期 AGI 研究的重要基准。传统 LLM 在此任务上表现极差，但最新模型已展现出一定能力。

## 12.8 安全基准

对于 AI 来说，安全意味着什么？

<div align="center">
   <img src="images/12-34-什么是安全.png" />
   <p>图12.34 什么是安全</p>
 </div>


安全评估旨在衡量模型执行有害行为的倾向。在 HELM 的 [Leaderboard Safety Scenarios](https://crfm.stanford.edu/helm/safety/latest/#/leaderboard) 展示了当前 LLM 在安全领域的排行榜。 

<div align="center">
   <img src="images/12-35-HELM上LLM在安全领域的排行榜.png" />
   <p>图12.35 HELM上LLM在安全领域的排行榜</p>
</div>

### HarmBench

[HarmBench](https://arxiv.org/abs/2402.04249) 定义了 510 种违反法律或社会规范的有害行为，通过提示词测试模型是否会执行，并评估其拒绝率。

在 HELM 的 [Leaderboard armBench](https://crfm.stanford.edu/helm/safety/latest/#/leaderboard) 展示了当前 LLM 在 HarmBench 基准上的排行。 

<div align="center">
   <img src="images/12-36-HELM上LLM在HarmBench基准上的排行.png" />
   <p>图12.36 HELM上LLM在HarmBench 基准上的排行</p>
</div>

### AIR-Bench

[AIR-Bench](https://arxiv.org/abs/2407.17436) 基于全球监管框架和公司政策，构建了一个包含 314 个风险类别和 5694 个提示的系统性评测集。

<div align="center">
   <img src="images/12-37-AIR-Bench评测集概览.png" />
   <p>图12.37 AIR-Bench评测集概览</p>
</div>

在 HELM 的 [Leaderboard AIR-Bench](https://crfm.stanford.edu/helm/air-bench/latest/#/leaderboard) 展示了当前 LLM 在 AIR-Bench 基准上的排行。 

<div align="center">
   <img src="images/12-38-HELM上LLM在AIR-Bench基准上的排行.png" />
   <p>图12.38 HELM上LLM在AIR-Bench基准上的排行</p>
</div>

### 越狱（Jailbreaking）:


即使模型被训练为拒绝有害请求，[攻击者也能通过自动优化提示（如 Greedy Coordinate Gradient, GCG）来绕过安全防护](https://arxiv.org/pdf/2307.15043)。这类攻击甚至能从开源模型迁移到闭源模型。

<div align="center">
   <img src="images/12-39-模型越狱案例.png" />
   <p>图12.39 模型越狱案例</p>
</div>

### 部署前测试

美国安全研究所和英国人工智能安全研究所携手合作，公司在发布前向安全机构提供模型访问权限（目前为自愿性质），安全机构进行评估并向公司提交[报告](https://www.nist.gov/system/files/documents/2024/12/18/US_UK_AI%20Safety%20Institute_%20December_Publication-OpenAIo1.pdf)。

### 但安全究竟是什么？

安全不仅关乎“拒绝”，也关乎“能力”。在医疗等高风险场景，减少幻觉本身就是提升安全性和能力。对于闭源 API 模型，倾向性（propensity） 是关键；而对于开源基础模型，能力（capability） 本身就是风险。

## 12.9 真实性 

语言模型在实践中被广泛应用：

<div align="center">
   <img src="images/12-40-OpenAI模型被广泛使用.png" />
   <img src="images/12-40-基于LLM构建的Cursor编程工具被广泛应用.png" />
   <p>图12.40 LLM被广泛使用的两个案例</p>
</div>


然而，大多数现有的基准测试（如 MMLU）是标准化的“考试”，与现实世界的使用场景相去甚远。真实用户的提示更多是“求助型”（asking）而非“测验型”（quizzing）。

### Clio (Anthropic)

在 [Clio: Privacy-Preserving Insights into Real-World AI Use
](https://arxiv.org/abs/2412.13678) 论文里，提出使用语言模型分析真实用户数据，分享人们提问的一般模式：


<div align="center">
   <img src="images/12-41-Clip使用语言模型分析真实用户数据.png" />
   <p>图12.41 Clip使用语言模型分析真实用户数据</p>
</div>

### MedHELM

为解决此问题，出现了如 [MedHELM](https://arxiv.org/abs/2412.13678) 这样的基准，它由 29 名临床医生贡献了 121 个真实的临床任务，更贴近实际医疗应用场景。然而，真实性与隐私之间存在天然矛盾，许多真实数据（如患者病历）无法公开。

<div align="center">
   <img src="images/12-42-MedHELM构建流程.png" />
   <p>图12.42 Clip使用语言模型分析真实用户数据</p>
</div>

## 12.10 有效性 

评估的有效性面临两大核心挑战：

### 训练-测试集重叠（Train-Test Overlap）:

在预训练数据即整个互联网的时代，确保测试集未被模型“见过”变得极其困难。这会导致评估结果虚高。解决方法包括开发工具推断重叠情况和推动行业规范，要求模型提供商主动报告重叠检测结果。

机器学习入门时我们就学过不要用测试集进行训练，在之前的基础模型（ImageNet、SQuAD）我们的训练集和测试集划分明确，但是如今语言模型的训练通常使用大规模的多种来源的语料，大部分机构不会公开他们的使用的详细数据。

#### 路线 1：尝试从模型中推断训练集和测试集的重叠部分

[PROVING TEST SET CONTAMINATION IN BLACK BOX LANGUAGE MODELS](https://arxiv.org/pdf/2310.17623)利用数据点的可交换性尝试从模型中推断训练集和测试集的重叠部分：

<div align="center">
   <img src="images/12-43-利用数据点的可交换性尝试从模型中推断训练集和测试集的重叠部分.png" />
   <p>图12.43 利用数据点的可交换性尝试从模型中推断训练集和测试集的重叠部分</p>
</div>

#### 路线 2：鼓励报告规范

[Language model developers should report train-test overlap](https://arxiv.org/abs/2410.08385) 提倡模型提供者应报告训练集与测试集的重叠情况

### 数据集质量

[许多基准存在标注错误或噪声](https://arxiv.org/abs/2502.03461)。例如，[修正后的 SWE-Bench Verified](https://openai.com/index/introducing-swe-bench-verified/) 版本显示，原始分数可能因错误而被低估。

<div align="center">
   <img src="images/12-44-许多基准存在标注错误或噪声.png" />
   <p>图12.44 许多基准存在标注错误或噪声</p>
</div>


## 12.11 我们到底在评估什么？

我们到底在评估什么？换句话说，游戏的规则是什么？这是一个根本性的问题。

- 过去：在 ImageNet 时代，我们评估的是方法（method），即在固定数据集和训练协议下，新算法的优劣。
- 现在：我们更多评估的是模型/系统（model/system），即“端到端”的最终产品，开发者可以使用任何数据、任何技巧。

这两种范式各有价值：评估方法能促进算法创新；评估系统对下游用户更有意义。但必须清晰界定评估的“游戏规则”，否则比较将失去意义。

