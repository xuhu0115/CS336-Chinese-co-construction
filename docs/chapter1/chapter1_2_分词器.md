# 分词器
分词器是连接人类自然语言与机器计算的关键，负责将非结构化的文本高效地编码为模型可理解的数字序列（Tokens）。其地位至关重要，它决定了模型的输入形式。我们可以将大语言模型（LLM）视为一位阅读者，分词器则执行了文本的“认知拆解”即将连续的文本流切分为具备最小语义或功能意义的基本单元，类似于人类阅读时对词汇或音节的本能识别。本章节将剖析分词器实现文本“数字化”的核心原理，并探讨BPE等主流分词算法的实际应用。

<div align="center">
   <img width="800" height="500" alt="1" src="https://github.com/user-attachments/assets/bc838c76-7eff-4479-a760-ef404fc48e89" />
   <p>图1 分词器与LLM</p>
 </div>
 
## 1训练分词器
在我们把海量数据喂给大模型之前，必须先经过一道关键工序——分词。分词器常被视为LLM的一部分，但它其实拥有独立的训练生命周期。我们需要利用正则表达式对原始文本进行预处理，并统计构建出一套高效的**词元——数字**离散序列转化映射表（vocab），这个映射过程决定了模型眼中的世界是由字、词还是更碎的片段组成的，直接影响后续模型对语义的理解效率。也正因如此，[分词器](https://tiktokenizer.vercel.app/?model=deepseek-ai%2FDeepSeek-R1)虽独立训练却与LLM保持着“强耦合”的关系。

```
训练一个用于现代大型语言模型的分词器可以拆成四步：准备语料 → 初始化基础单元 → 统计并迭代合并 → 输出产物并用于编码、解码。
```

### 第一步 准备语料

1. 在准备语料阶段，应尽量收集覆盖目标应用场景的多样化文本，以便训练出的词表对下游任务具有良好泛化能力。

   - 准备不同类型的文本信息比如小说、散文、诗歌等不同描述风格的信息。
   - 多种语言的文本信息比如中文、英文、韩语、法语等。
     
2. 对原始文本进行清洗和标准化是必须的步骤，包含去除或屏蔽无关元数据、修正或删除乱码与非法字符、统一字符编码为UTF-8，并对重复或近重复样本进行去重以减少训练偏移。
3. 对带有敏感信息或隐私的语料要提前进行脱敏处理与合规检查，明确哪些信息不可用于训练并记录数据来源与许可。
      处理示例基于python实现：
   
   ```python
   import re
   
   def desensitize(text):
       # 1.常见中文姓名（简单规则：2~3字，全中文）
       text = re.sub(r'([\u4e00-\u9fa5]{2,3})(的)', r'[NAME]\2', text)
   
       # 2.手机号脱敏（11位数字）
       text = re.sub(r'1[3-9]\d{9}', '[PHONE]', text)
   
       # 3.地址脱敏（如 “居住于重庆xx…地方”）
       # 捕获（居住于、现居住于、现居于）字段后面的“重庆”“北京”“上海”“广州” 等 +任意字符
       text = re.sub(r'(居住于|现居住于|现居于)([\u4e00-\u9fa5A-Za-z0-9]+)', r'\1[PLACE]', text)
       return text
   
   # 测试
   text="小明的联系电话是13312311111，现在居住于重庆两江新区某小区。"
   print(f"处理前:{text}")
   print(f"脱敏后：{desensitize(text)}")
   ```
   
   处理前

   >小明的联系电话是13312311111，现在居住于重庆两江新区某小区。
   
   脱敏后

   >[NAME]的联系电话是[PHONE]，现居住于[PLACE]。
   
   如果句子中出现姓名、电话号码、住址等特定信息进行脱敏处理。值得注意的是这样数据脱敏不仅是保护隐私与合规的需要，同时也能让tokenizer的统计过程更干净、更稳定。大量独特且高基数的特定信息如*姓名、电话号码、身份证号等*如果不处理，会在语料中以几乎不重复的形式出现，使分词器在训练时被这些“只出现一次的随机字符串”干扰，从而产生大量低价值的token片段，通过脱敏替换这类信息后模型能够更专注于学习真正高频、有规律的语言结构，使词表更加精简分词效果更一致，泛化能力也更强。不过，如果下游任务本身需要识别真实实体（如信息抽取等），过度脱敏会削弱训练信号。因此需要在**保护隐私**与**保留关键语义信息**之间进行合理的策略选择与权衡。
   
>token是LLM的基本输入单位，由分词器根据统计规则把文本拆成的子词、字符或字节，再映射成数字ID。

4. 在多语言或混合语料场景中，应当统计每种语言的占比并考虑是否对低资源语言做过采样或专门保留，以避免词表被高频语言主导导致低频语言表现差即语料类型、语言不平衡会导致`token`碎片化、占用更多`token`、增加Transformer计算成本并降低低资源语言性能。
   
   例如准备一种可以支持四种语言的分词器，这里假设提前收集到的原始未经过第7步处理的各语言原始语料占比如下：
   
| 语言 | 语料量 |
| :--- | ---: |
| 中文 | 200 GB |
| 英文 | 150 GB |
| 法语 | 10 GB |
| 韩文 | 5 GB |

>这是一个典型的多语言语料不平衡场景。若将上述语料不经处理直接混合训练分词器，其统计过程会被中文和英文主导，导致法语与韩文的常见字串在合并阶段难以进入高频统计，从而无法占据足够的词表空间，最终在`vocab`中会出现大量被切得过碎的`token`，形成严重碎片化，下游LLM在法语与韩文任务上会因此表现显著劣化。

因此在准备语料的第4步应先按语言统计语料占比，并根据目标能力设定合理的采样策略。例如将语料比例调整为 中文:英文:法语:韩文=4:4:1:1或者采用完全均衡策略。通过对高资源语言下采样或对低资源语言过采样、增强，可以获得更符合目标分布的训练语料，再使用验证集评估各语言的token覆盖率、平均其碎片化程度，以确保最终词表在多语言任务中具备稳定且均衡的表示能力。
    
5. 建议保留一小部分未参与训练的验证语料比如训练集:验证集＝99:1，用来在训练过程中评估分词器对真实文本的编码效率与平均token长度等统计指标。

### 第二步 初始化基础单元

1. 预分词的主要任务是将原始文本切分成可统计、可合并的基础单元，例如字符、字节或Unicode片段。常见策略包括基于空格和标点的切分、按Unicode类别划分，或直接采用字节级切分。需要注意的是并不是所有的分词器都需要用户显式进行预分词。例如基于SentencePiece的分词器将标准化和预分词逻辑内置，因此无需在外部额外执行预分词步骤。

   - 基于空格和标点的切分策略：一个完整的句子中遇到空格或者标点（.,!?[]{}...）可以分为独立的tokens，该方法适用于大多数预分词处理过程。
     
   ```python
   
   # 基于空格和标点切分的实现示例
   import re
   
   def part(text):
       # 将标点符号单独拆开，并按照空格进行分割
       text = re.sub(r'([.,!?;:()"\'\[\]{}])', r' \1 ', text)
       tokens = text.split()
       return tokens
   
   # 测试
   s = "I like DataWhale."
   print(part(s))

   ```

   输入
   >I like DataWhale.
   
   输出token划分
   >['I', 'like', 'DataWhale', '.']

   - Unicode类别划分策略：根据字符类型比如字母、数字、标点、中文、特殊字符等自动切分token，不同Unicode类别会属于不同token块，这种方法天然适用于多种语言混合的文本，提供了可靠的基线切分。
        
     ```python
        # Unicode类别划分token
      import unicodedata
      def unicode_category_type(ch):
          """根据Unicode类别将字符划为：中文、字母、数字、其他"""
          if '\u4e00' <= ch <= '\u9fff':
              return "CJK"
          if ch.isdigit():
              return "DIGIT"
          if ch.isalpha():
              return "ALPHA"
          return "OTHER"
      
      def tokenize_unicode_category(text):
          if not text:
              return []
      
          tokens = []
          current = text[0]
          current_type = unicode_category_type(current)
      
          for ch in text[1:]:
              ch_type = unicode_category_type(ch)
              if ch_type == current_type and ch_type != "OTHER":
                  # 同类字符，继续合并
                  current += ch
              else:
                  # Unicode不同类 → 切分
                  tokens.append(current)
                  current = ch
                  current_type = ch_type
          tokens.append(current)
      
          # 最后再把"OTHER"类型（标点等）拆开
          final_tokens = []
          for t in tokens:
              if unicode_category_type(t[0]) == "OTHER":
                  final_tokens.extend(list(t))
              else:
                  final_tokens.append(t)
          return final_tokens
      
      # 测试
      s = "Hello，DataWhale成立于2018年12月6日！至今已有7年的历史了~"
      print(tokenize_unicode_category(s))

     ```
     
     输入
     >Hello，DataWhale成立于2018年12月6日！至今已有7年的历史了~

     输出token划分
     >['Hello', '，', 'DataWhale', '成立于', '2018', '年', '12', '月', '6', '日', '！', '至今已有', '7', '年的历史了', '~']

   - 字节级切分策略：先将每个字符拆成[UTF-8字节序列](https://datatracker.ietf.org/doc/html/rfc3629)，不依赖语言种类、字符，按照单个字节序列得到一个独立的token。

     ```python
     
      def tokenize_byte_level(text):
          tokens = []
          for ch in text:
              # 字符对应的UTF-8字节序列
              utf8_bytes = ch.encode("utf-8")
              hex_bytes = [f"{b:02X}" for b in utf8_bytes]
      
              # 打印转换过程
              print(f"{ch} 转化为UTF-8字节序列：{hex_bytes}")
      
              # 加入token列表
              tokens.extend(hex_bytes)
          return tokens
      
      # 测试
      s = "All for learners！"
      print(tokenize_byte_level(s))

     ```

     输入
     >All for learners！

     输出token划分
     >['41', '6C', '6C', '20', '66', '6F', '72', '20', '6C', '65', '61', '72', '6E', '65', '72', '73', 'EF', 'BC', '81']

     
**Uincode与UTF-8的联系：**

   Unicode就像给全世界所有字符发的“身份证号”，不管是英文A、汉字“中”、还是emoji 😄等不同类型的字符都在Unicode里有一个唯一编号比如A是U+0041，“中”是U+4E2D。但“身份证号”本身只是一个抽象编号，电脑不能直接存储。

   UTF-8就像把这个字符对应的“身份证号”写进电脑的具体方式。它规定这个字符的编号应该用几个字节、按什么规则写下来。英文常用字符在UTF-8中只需要1个字节，而中文通常需要3个字节。不管是Unicode还是UTF-8都可以表示不同类别的字符，两者配合起来让自然语言可以被计算机准确存储、传输和解析，是人机交互之间的“桥梁”。

在LLM的token划分中，常见策略包括基于规则的预分词按空格以及标点切分、按Unicode类别的分段如连续汉字、连续拉丁字母或数字...以及更底层的UTF-8字节级切分。前两类方法在处理缺乏显式分隔符或长段同类字符<ins>例如连续的中文长句、拼接的代码标识符或压缩后的字符串时</ins>存在局限，预处理阶段难以有效断句，分词器可能被迫退化为字符级切分，从而把文本映射成更长的token ID序列，增加Transformer的计算和内存负担，其中注意力的复杂度近似为 $O(n^2)$ 。

相比之下，UTF-8字节级策略具有最强的通用性，它把任意文本统一拆为字节序列，从而从根本上减少未登录词（OOV）问题并覆盖任意字符集。但因为它以最细粒度开始，训练时通常需要更多轮的共现统计与合并来把零散字节压缩为紧凑且具语义的token，才能在Transformer计算效率与语义表征之间取得平衡。

>未登录词是当LLM模型在处理新的、实际应用的文本时，如果遇到一个词汇表中没有的Token，那么这个Token就被视为一个OOV。

2. 对大多数以空格为词边界的语言，可先用正则表达式按单词边界和标点进行初步切割，而对中文、日文等不以空格为词界的语言则通常采用逐字符或基于字的初始单元来保证覆盖性。
   
   字节级预分词的好处：
   - token利用率更高，提高BPE合并token的自由度以及尽可能合并共现频率高的单个字符，提高文本信息压缩率。
   - 可兼容处理多种语言。
   - 学到更多高频片段，减少未登录词的出现情况，模型推理更快（token数少）。
     
>文本压缩率：指一段文字被转换成token（数字化）后，用多少token来表示内容的紧凑程度，同样内容使用的token越少，压缩率就越高。

3. 预分词生成的基础单元序列将作为后续统计合并的输入，务必保存该序列与对应位置信息以便在训练过程反复高效更新。
   
   实现示例：
   
   ```python
   def btp_hex_list(text):
       """
       UTF-8字节级预分词，返回：
           1. tokens: 每个字符的字节序列+位置信息
           2. t: 所有字节的十六进制字符串列表
       """
       tokens = []
       t = []
       for idx, char in enumerate(text):
           utf8_bytes = char.encode('utf-8')
           hex_bytes = ' '.join(f"{b:02X}" for b in utf8_bytes)
           tokens.append({
               'char': char,
               'bytes': hex_bytes, # 单个字符对应的UTF-8字节序列
               'start': idx,   # 文本信息起始位置
               'end': idx + 1   # 文本信息结束位置
           })
           # 将每个字节拆成单个十六进制字符串
           t.extend([f"{b:02X}" for b in utf8_bytes])
       return tokens, t
   
   # 测试
   text = "Hi，你好🐋"
   tokens, t = btp_hex_list(text)
   for i in tokens:
       print(i)
   print(t)
   ```

### 第三步 统计并迭代更新 ——> 核心步骤

1. **子词候选统计**

遍历语料以收集用于后续决策的统计信息，具体方法随算法不同而异：

   - BPE：统计当前符号字符、子词序列中相邻对的出现频次，每次贪心合并出现频次最高的相邻对，迭代构建词表——其决策仅基于频率统计。
   - WordPiece：评估合并或保留某些子词对对语料似然即语言模型性能的贡献，选择能显著提升语料拟合度的合并操作。
   - Unigram：从一个过大的种子词表出发，通过迭代估计子词概率并重新分词，从而逆向优化出一个紧凑、高效的词表。
       
| 算法 | 适用场景 | 常见分词器结合 | 典型LLM框架 |
| --- | --- | --- | --- |
| [BPE](https://arxiv.org/pdf/1508.07909)| 快速、简单，高频子词压缩好；<br>多语言、大语料适用 | 字节级BPE<br>或代码点级BPE| GPT系列、RoBERTa|
| [WordPiece](https://huggingface.co/learn/llm-course/en/chapter6/6)| 平衡OOV与词表大小；<br>预训练语言模型常用 | 代码点、字符级实现<br>Google BERT原始实现 | [BERT](https://arxiv.org/abs/1810.04805)、DistilBERT、RoBERTa |
| [Unigram](https://arxiv.org/abs/1808.06226)| 灵活适应语料分布；<br>多语言、低频词友好 | SentencePiece Unigram模式<br>支持byte-fallback | T5、mT5、其他使用 SentencePiece 的模型 |
| [SentencePiece](https://arxiv.org/pdf/1804.10959)| 跨语言、通用tokenizer；<br>可移植、多语种适用 | 代码点级BPE、Unigram<br>byte-fallback | T5、mT5、各种开源模型和自定义LLM |

总体来说，以上四种子词分词算法各有特点，没有哪一种是绝对最好的。选择算法时应根据具体的文本内容（语料分布）、任务类型（理解或生成）、词表规模以及是否需要处理多语言来决定，这样才能让训练出来的LLM模型发挥最佳性能。

2. **迭代**
BPE、WordPiece、Unigram、SentencePiece这四种迭代算法简要分析：

   - BPE算法：可以借助第一步子词候选统计数据作为初始化数据，进行单个token合并形成新的token，然后在多次迭代过程中动态统计共现次数，得到新的token。
   - WordPiece算法：迭代过程中动态计算词汇表中所有相邻子词对的出现频次。其关键创新在于合并标准它不合并最频繁的对，而是合并能最大化以下得分的对：
  
     $$
     score = freq(A, B) / [freq(A) * freq(B)]
     $$
     
     这个得分近似于P(B|A) / P(B)，即合并AB相较于独立出现的可能性。它选择合并后最可能“粘”在一起的对，从而优先形成有语言学意义的单元。
     
这个过程中需要保持特殊控制token（如`<PAD>`、`<UNK>`、`<CLS>`、`<MASK>`等），在分词器迭代更新过程中不参与修改，这样可以确保它们的词——数字映射保持固定，编码后的离散数字序列能够准确还原为原始文本。同时，这些token不会在统计合并或概率优化中被拆分或覆盖，从而有效减少碎片化token的出现。无论使用BPE、WordPiece、SentencePiece还是Unigram等算法，这一策略都适用有助于保护关键token的完整性，保证模型训练和推理的一致性。

4. **通用终止条件**
   不同算法可能采用不同策略，但核心目标都是得到一个可控大小且高效的词表：

   - **词表大小限制**：当词表达到预设大小如32k、50k、100k、256k...时停止。
   - **迭代步数限制**：达到最大合并或训练步数（合并次数、EM迭代次数）时停止。
   - **低贡献候选终止**：

     - 对BPE、WordPiece：最高合并频率低于阈值时停止。
     - 对Unigram、SPM：子词概率低于阈值，删除后剩余词表稳定即可停止。
   
   - **训练、验证集指标收敛**：可监控OOV率、平均token长度、压缩率等指标，直到达到预设目标或收敛。

5. **大规模语料优化**

   - 分布式统计或分片训练。
   - 精确、近似计数。
   - 确定性排序+固定随机种子保证训练可复现。

6. **监控与评估指标**

   - 平均token长度即每词或每字符、未登录词率（OOV）。
   - 高频token示例合理性。
   - 压缩率（总token数 / 总字符数或总字节数）。

> 总结：分词器训练的核心是<ins>迭代更新候选子词 → 控制词表大小或收敛指标 → 监控质量指标</ins>，不同算法仅在“候选生成方式”和“迭代更新策略”上有差异。


### 第四步 输出产物并用于编码与解码

1. **导出核心产物**
   训练完成后需要导出至少两个关键文件：

   - **vocab文件**：记录所有token及其对应的id，是编码器和解码器的核心索引。
   - **merges文件**：按顺序记录所有子词合并规则或概率模型。二者共同决定tokenizer的编码与解码逻辑，并确保编码的可逆。

2. **下游使用前的验证与评估**

   - 将tokenizer应用于一部分验证集后，建议统计以下关键指标：**平均token数与最大长度分布**，直接影响显存占用、训练速度和推理效率；**碎片化情况**，检查关键实体、专业术语是否被拆得过碎，避免影响模型理解；**跨语言token平衡度**，多语言任务中需确保不同语言的常见模式都有足够的token支持。

   - 如果后续需要扩表如加入新领域术语、专业词或品牌名等，建议优先采用这些方式，而非完全重训 okenizer：**增量训练**、**加入新的 merges 项**、**清理极低频token**。

   - 扩表后应进行一次**回归测试**，确保：与旧模型保持兼容且根据数字化编码可以还原回到最开始的输入文本、不发生token分配冲突或token耗尽问题。

3. **版本管理与可复现性保证**
   词表与merges文件应纳入严格的版本控制，包括：语义化版本号（如1.2.0 ）、每次修改的变更日志、在训练脚本和推理 pipeline中显式固定tokenizer版本，防止模型训练阶段与推理部署阶段使用不同tokenizer，导致结果不可复现或性能下降。

## 2常用的分词器
在NLP的发展历程中，分词策略经历了几次重要的演变。我们主要关注四种最典型的范式：字符、字节、词级、BPE分词器，以及结合课程[lecture1](https://stanford-cs336.github.io/spring2025-lectures/?trace=var/traces/lecture_01.json)各个分词器伪代码转化为python代码实践。

### 2.1 字符分词器

#### 原理介绍

这是最直观、最简单的分词方式，它将文本拆解为最小的字符单位即单个字符形如英语中的字母 a, b, c或者中文里的单字你，好。

  - **优点：**
      - **词表极小：** 英语只需包含26个字母+符号；中文只需包含常用汉字（约几千个）。
      - **无OOV问题：** 任何生僻词都是由基础字符组成的，不会出现“未知词”。
  - **缺点：**
      - **序列过长：** 一句话变成字符后，长度会增加数倍，大大消耗LLM宝贵的上下文窗口，从而加大LLM的transformer计算现存消耗。
      - **语义稀疏：** 单个字符（如t）通常不具备独立的语义，模型需要更深的网络层数来组合出意义。

   实现示例：
   ```python
      # 字符Tokenizer
   class CharacterTokenizer:
       def __init__(self):
           pass  # 不需要额外参数，直接用ord、chr
   
       def encode(self, text):
           """
           将字符串编码为字符索引列表（Unicode code points）
           """
           return [ord(ch) for ch in text]
   
       def decode(self, indices):
           """
           将索引列表解码为字符串
           """
           return ''.join([chr(i) for i in indices])
   
   # 测试代码
   if __name__ == "__main__":
       tokenizer = CharacterTokenizer()
       string = "hi，很好的，terrific！🐋"  # 测试字符串
   
       # 编码
       indices = tokenizer.encode(string)
       print("编码ID:", indices)
   
       # 解码
       reconstructed_string = tokenizer.decode(indices)
       print("解码:", reconstructed_string)
   
       # 验证是否可逆
       assert string == reconstructed_string, "字符编码、解码不一致!"
   
       # 计算词汇量（最大Unicode code point+1）
       vocabulary_size = max(indices) + 1
       print("词汇量（上限）", vocabulary_size)
   
       # 简单压缩率计算
       def get_compression_ratio(text, indices):
           # 压缩率 = 原字符串字节数/编码索引字节数
           import sys
           original_bytes = len(text.encode('utf-8'))
           encoded_bytes = len(indices) * 4  # 假设每个Unicode code point用4字节存储
           return original_bytes / encoded_bytes
   
       compression_ratio = get_compression_ratio(string, indices)
       print("压缩比率:", compression_ratio)
   ```

   
输入
> hi，很好的，terrific！🐋

输出
> 编码ID: [104, 105, 65292, 24456, 22909, 30340, 65292, 116, 101, 114, 114, 105, 102, 105, 99, 65281, 128011]
> 
> 压缩比率: 0.47058823529411764


### 2.2 字节分词器

#### 原理介绍

计算机底层存储文本本质上都是**字节**，在UTF-8编码中，英文通常占1个字节，汉字通常占3个字节。字节分词器直接对二进制字节进行操作。

  - **核心逻辑：** 不再维护“字符”的词表，而是维护一个大小为256的基础词表（0x00到0xFF）。
  - **应用：** 现代LLM如GPT-4, Llama通常不单独使用纯字节分词，而是将字节作为BPE的基础单位*即BBPE，这样可以彻底解决跨语言和特殊符号如emoji 🌍等的编码问题。

   ```python
   # Byte-level Tokenizer实现
   class ByteTokenizer:
       def __init__(self):
           # vocab就是0~255的256个值
           self.vocab_size = 256
   
       def encode(self, text: str):
           # 将字符串编码为UTF-8字节序列 转为int列表
           return list(text.encode("utf-8"))
   
       def decode(self, indices):
           # 将int列表→bytes→UTF-8 字符串
           return bytes(indices).decode("utf-8")
   
   
   # 计算压缩率
   def get_compression_ratio(text: str, indices):
       input_byte_len = len(text.encode("utf-8"))  # 原始字节序列长度
       token_len = len(indices)                   # token数量
       return input_byte_len / token_len if token_len > 0 else 1
   
   # 测试
   if __name__ == "__main__":
   
       print("以下测试单字节与多字节 UTF-8 字符：")
       assert bytes("a", encoding="utf-8") == b"a"
       assert bytes("🌍", encoding="utf-8") == b"\xf0\x9f\x8c\x8d"
       print("测试通过：UTF-8单字节与多字节验证完毕\n")
   
       tokenizer = ByteTokenizer()
       string = "Hello, 🌍! 你好!"
       print("原始字符串：", string)
       indices = tokenizer.encode(string)
       print("编码后的byte token序列：", indices)
   
       reconstructed_string = tokenizer.decode(indices)
       print("解码结果：", reconstructed_string)
   
       assert string == reconstructed_string
       print("\nRound-trip测试通过")
   
       vocabulary_size = tokenizer.vocab_size
       print("\n词表大小:", vocabulary_size)
   
       compression_ratio = get_compression_ratio(string, indices)
       print("压缩率compression_ratio:", compression_ratio)
   
       assert compression_ratio == 1
       print("压缩率测试通过（byte tokenizer无压缩）。")
   ```

输入
> Hello, 🌍! 你好!

输出
> 原始字符串： Hello, 🌍! 你好!
> 
>编码后的byte token序列： [72, 101, 108, 108, 111, 44, 32, 240, 159, 140, 141, 33, 32, 228, 189, 160, 229, 165, 189, 33]
> 
>解码结果： Hello, 🌍! 你好!

值得注意的是**字节级分词器的压缩比恒等于 1**，原因在于：

- 输入文本中单个字符首先被编码为UTF-8字节序列；
- 字节级分词器将每一个UTF-8字节（0~255）直接作为一个token；
- 因此`token数量=UTF-8字节数`。

所以

$$
compression_{ratio}
= \frac{\text{UTF-8 字节长度}}{\text{token 数量}}
= \frac{N}{N}
= 1
$$

**也就是说，字节级分词器完全不具备压缩能力即每个字节对应一个token，不会产生更长或更短的词片段。**

#### 2.3 词级分词器

#### 原理介绍

在深度学习早期（如RNN时代）这是最主流的方法。它基于空格（英文）或分词算法（中文）将文本切分为具备独立语义的“词”。

  - **优点：** Token保留了完整的语义信息比如"apple" 直接对应一个Token ID...。
  - **缺点：**
      - **词表爆炸：** 英语中 `look, looks, looked, looking` 会被视为4个完全不同的ID，导致词表巨大几十万甚至上百万。
      - **OOV 问题严重：** 遇到没见过的词如人名、新造词等，只能标记为 `<UNK>` ，导致信息丢，从而影响LLM的表现能力。

   实现示例：
   ```python
   import regex
   
   # deepseek tokenizer中使用的经典正则表达式（简化版）
   TOKENIZER_REGEX =  r"\p{L}+|\p{N}+|[^\p{L}\p{N}\s]+|\s+"
   
   # 压缩率计算
   def get_compression_ratio(text: str, segments):
       byte_len = len(text.encode("utf-8"))
       token_count = len(segments)
       return byte_len / token_count if token_count > 0 else 1
   
   
   # Word-level Tokenizer实现
   class WordTokenizer:
       def __init__(self, pattern=r"\w+|."):
           """
           pattern: 正则表达式（默认基础版：把连续字母数字合成一个词）
           """
           self.pattern = pattern
           self.word2id = {}
           self.id2word = {}
   
       def build_vocab(self, texts):
           """
           根据训练文本列表建立词表
           """
           vocab = set()
           for text in texts:
               segments = regex.findall(self.pattern, text)
               vocab.update(segments)
   
           vocab = sorted(vocab)
           self.word2id = {w: i for i, w in enumerate(vocab)}
           self.id2word = {i: w for w, i in self.word2id.items()}
   
       def encode(self, text):
           """
           文本 → 字符串片段 → token id列表
           未登录词 UNK = -1
           """
           segments = regex.findall(self.pattern, text)
           return [self.word2id.get(seg, -1) for seg in segments], segments
   
       def decode(self, ids):
           """
           token ID → 原始片段 → 拼成字符串
           """
           return "".join(self.id2word.get(i, "<UNK>") for i in ids)
   
   # 测试
   if __name__ == "__main__":
   
       string = "It's so supercalifragilisticexpialidocious!👋👋"
       print("原始字符串：", string)
   
       # 使用基础正则分词（基于空格和标点切分）
       basic_segments = regex.findall(r"\w+|.", string)
       print("基础正则分词结果：")
       print(basic_segments)
   
       # 使用deepseek风格正则
       segments = regex.findall(TOKENIZER_REGEX, string)
       print(f"deepseek风格分词结果：{segments}")
   
       # 构建词表
       tokenizer = WordTokenizer(pattern=TOKENIZER_REGEX)
       tokenizer.build_vocab([string])
   
       print("词表大小：", len(tokenizer.word2id))
   
       # 编码
       ids, segs = tokenizer.encode(string)
       print(f"编码token IDs：{ids}")
   
       # 字节序列
       byte_tokens = [b for b in string.encode("utf-8")]
       print(f"UTF-8字节序列：{byte_tokens}")
   
       print(f"编码segments：{segs}")
   
       # 解码
       decoded = tokenizer.decode(ids)
       print("解码结果：", decoded)
   
       # 压缩率
       ratio = get_compression_ratio(string, segs)
       print("压缩率：", ratio)
   ```
输入   
>It's so supercalifragilisticexpialidocious!👋👋

输出
>基础正则分词结果：
>
>['It', "'", 's', ' ', 'so', ' ', 'supercalifragilisticexpialidocious', '!', '👋', '👋']
>
>deepseek风格分词结果：['It', "'", 's', ' ', 'so', ' ', 'supercalifragilisticexpialidocious', '!👋👋']
>
>词表大小： 7
>
>编码token IDs：[3, 2, 4, 0, 5, 0, 6, 1]
>
>压缩率： 6.375

### 2.4 BPE分词器

#### 原理介绍

这是目前LLM（GPT, BERT, Llama等）最主流的分词算法，BPE是一种试图在<ins>字符级（粒度太细）</ins>和<ins>词级（粒度太粗）</ins>之间找到平衡。

  - **核心思想：** 统计语料中相邻字符对出现的频率，迭代地将**最频繁出现的字符对**合并成一个新的Token。

  - **过程：**
    1. 初始化：将单词拆成字符序列。
    2. 统计：计算所有相邻字符对的频率（如'e' 和's'经常一起出现）。
    3. 合并：将频率最高的对（'e', 's'）合并为新 Token ('es')。
    4. 循环：重复上述步骤，直到达到预设的词表大小。

   实现实例：简易版BPE训练过程
   
   ```python
   import regex
   from collections import Counter
   
   # DeepSeek风格正则
   DEEPSEEK_REGEX = r"\p{L}+|\p{N}+|[^\p{L}\p{N}\s]+|\s+"
   
   # 使用grapheme cluster保持emoji不被拆分
   def split_graphemes(token):
       return tuple(regex.findall(r'\X', token))
   
   # BPE训练函数
   def train_bpe(texts, num_merges=50):
       """
       texts: 文本列表（用于训练BPE）
       num_merges: BPE 迭代合并的次数
       """
       # 1.构建初始vocab（字符级+</w>结束符）
       vocab = Counter()
       for text in texts:
           tokens = regex.findall(DEEPSEEK_REGEX, text)
           for token in tokens:
               chars = split_graphemes(token) + ('</w>',)
               vocab[chars] += 1
       merges = []
       for _ in range(num_merges):
           # 统计相邻pair出现次数
           pairs = Counter()
           for word, freq in vocab.items():
               for i in range(len(word)-1):
                   pairs[(word[i], word[i+1])] += freq
           if not pairs:
               break
   
           # 找到最常见pair
           best_pair = max(pairs, key=pairs.get)
           merges.append(best_pair)
   
           # 合并所有vocab中的该pair
           new_vocab = {}
           for word, freq in vocab.items():
               w = []
               i = 0
               while i < len(word):
                   if i < len(word)-1 and (word[i], word[i+1]) == best_pair:
                       w.append(word[i]+word[i+1])
                       i += 2
                   else:
                       w.append(word[i])
                       i += 1
               new_vocab[tuple(w)] = freq
           vocab = new_vocab
       return merges, vocab
   
   # BPE Tokenizer类
   class BPETokenizer:
       def __init__(self, merges):
           self.merges = merges
   
       def encode_word(self, token):
           # 初始分成字符+</w>
           word = list(split_graphemes(token)) + ['</w>']
           # 按merge顺序依次合并
           for pair in self.merges:
               i = 0
               new_word = []
               while i < len(word):
                   if i < len(word)-1 and (word[i], word[i+1]) == pair:
                       new_word.append(word[i]+word[i+1])
                       i += 2
                   else:
                       new_word.append(word[i])
                       i += 1
               word = new_word
           return word
   
       def encode(self, text):
           tokens = regex.findall(DEEPSEEK_REGEX, text)
           bpe_tokens = []
           for t in tokens:
               bpe_tokens.extend(self.encode_word(t))
           return bpe_tokens
   
       def decode(self, tokens):
           # 拼接tokens并去掉结尾</w>
           text = ''.join(tokens).replace('</w>', '')
           return text
   
   # 测试
   if __name__ == "__main__":
       train_texts = ["这只猫🐈很可爱", "the quick brown fox jumps over the lazy 🐕‍🦺"]
       merges, vocab = train_bpe(train_texts, num_merges=20)
       print("BPE合并:", merges)
       tokenizer = BPETokenizer(merges)
       test_text = "敏捷的棕色狐狸🦊"
       encoded = tokenizer.encode(test_text)
       print("编码:", encoded)
       decoded = tokenizer.decode(encoded)
       print("解码:", decoded)
   ```

输入
>test_text = "敏捷的棕色狐狸🦊"

输出
>BPE合并: [(' ', '</w>'), ('t', 'h'), ('th', 'e'), ('the', '</w>'), ('这', '只'), ('这只', '猫'), ('这只猫', '</w>'), ('🐈', '</w>'), ('很', '可'), ('很可', '爱'), ('很可爱', '</w>'), ('q', 'u'), ('qu', 'i'), ('qui', 'c'), ('quic', 'k'), ('quick', '</w>'), ('b', 'r'), ('br', 'o'), ('bro', 'w'), ('brow', 'n')]
>
>编码: ['敏', '捷', '的', '棕', '色', '狐', '狸', '</w>', '🦊', '</w>']

在BPE编码阶段，如果没有`</w>`算法可能把`the`错误地拆成'th'、'e'或在后续合并时与其他token错误合并。加上`</w>`后，`the`会被表示为['t', 'h', 'e', '</w>']，BPE就知道这是一个完整单词的结尾不会跨单词错误合并，那么解码阶段去掉`</w>`就能把token拼回`the`，保证原文恢复正确。

**因此，`</w>`的核心作用是保证单词完整性，并让编码可逆即可以从相应的数字序列转化为原文。**
#### 四种分词器对比表

| 分词器类型 | 粒度 | 词表大小 | 未登录词 (OOV) | 序列长度 | 代表模型 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 字符级 | 细 | 小 (100-5k) | 无 | 非常长 | Char-RNN |
| 词级 | 粗 | 极大 (>100k) | 严重 | 短 | Word2Vec, GloVe |
| **BPE** | **中 (自适应)** | **适中 (30k-100k)** | **极少** | **适中** | **GPT-4, Llama 3** |

除了分词器的选择与训练语料直接影响LLM的输入稀疏度与表示效率。用大规模、高质量且多样的语料训练分词器通常会减少token碎片化即生成更常见、更稳定的子词单元，使得同一段文字被编码为更少的token。由于Transformer的自注意力与许多操作的复杂度依赖于序列长度例如自注意力为 O(n²)，token数下降会直接减少计算与内存开销。同时，在固定的上下文窗口长度下单位token承载更多实际信息，这意味着模型能够在有限窗口内“看到”更多内容——从而在一定程度上缓解因上下文长度受限引起的信息丢失。

>注意上述情况这依赖于语料的覆盖与质量；若语料偏颇或过度合并罕见词，反而可能损害少数语言或专业术语的表示能力。

## 3分析DeepSeek的分词器
DeepSeek模型尤其是Coder系列，对代码和中英文都进行了高度优化，我们将加载DeepSeek Coder模型的官方分词器。
### 准备工作 加载DeepSeek Tokenizer
请确保transformers库已安装
```
# 安装transformers库
pip install transformers torch
```
我们将加载`deepseek-ai/deepseek-coder-6.7b-instruct`的分词器。
```python
from transformers import AutoTokenizer
# 使用DeepSeek Coder系列模型的分词器
MODEL_NAME = "deepseek-ai/deepseek-coder-6.7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"成功加载模型: {MODEL_NAME} 的分词器。")
print(f"分词器词表大小V: {len(tokenizer.get_vocab())}")
```
### 实例分析 DeepSeek分词器的处理逻辑
DeepSeek的分词器基于BPE在处理中、英文和代码时具有其独特的策略。

举例分析: 中文文本处理
观察DeepSeek如何处理中文短语，通常它也会使用子词或单个汉字Token来提高效率。
```python
chinese_text = "注意力机制是AI的核心技术。 🚀 🚀"
# 编码
encoded_ids = tokenizer.encode(chinese_text, add_special_tokens=False)
# 解码回Token字符串 (用于观察子词)
tokens = tokenizer.convert_ids_to_tokens(encoded_ids)
print(f"\n原文: {chinese_text}")
print(f"编码: {tokens}")
print(f"IDs:{encoded_ids}")
```

最后得到的token可能在显示上与原文有所差异这并不是编码本身出错，而是因为LLM所用的词表在训练过程中对某些字符或子词的覆盖不足（例如BPE训练不够充分），导致模型无法生成对应的token，从而在可读形式上看起来像“乱码”。通过增加训练语料量或进行充分的BPE训练，可以学习到更完整的token映射表，从而解决这个问题，使中文、英文、emoji等字符都能被正确编码和解码。接下来是相应的解决办法即训练BPE：

```python
"""
DeepSeek-V3 Tokenizer 实现示例
（核心包含：字节级BPE+DeepSeek风格正则预分词）
"""
import regex as re
from collections import Counter
from typing import List, Tuple, Dict, Iterable
import json
import base64


# 配置：DeepSeek 正则模式（预分词）
# \p{L}+   连续字母（中文、英文、所有 Unicode 字母）
# \p{N}+   连续数字
# [^\p{L}\p{N}\s]+  非字母数字空白的字符（如标点、emoji）
# \s+      连续空白符
DEEPSEEK_REGEX = r"\p{L}+|\p{N}+|[^\p{L}\p{N}\s]+|\s+"


# 基础函数：预分词与字节处理
def pretokenize(text:str):
    """按DeepSeek风格的正则进行预分词"""
    return re.findall(DEEPSEEK_REGEX, text)

def bytes2tokens(b:bytes):
    """
    将UTF-8字节序列转为latin1可表示的token列表。
    每个字节0–255都能被latin1 接映射到字符。
    """
    return [bytes([x]).decode('latin1') for x in b]

def tokens2bytes(tokens):
    """将 latin1 token 列表重新转回原始 bytes"""
    return b''.join([t.encode('latin1') for t in tokens])


# BPE训练相关
def build_corpus(texts):
    """
    构建byte-level语料。
    步骤：预分词 → UTF-8 编码 → 分解为单字节 → 作为初始 token 序列。
    """
    corpus = []
    for text in texts:
        for chunk in pretokenize(text):
            corpus.append(bytes2tokens(chunk.encode('utf-8')))
    return corpus

def pair_freq(corpus: List[List[str]]):
    """统计所有token序列中相邻token pair的出现频率"""
    pairs = Counter()
    for word in corpus:
        for i in range(len(word)-1):
            pairs[(word[i], word[i+1])] += 1
    return pairs

def merge_pair(word: List[str], pair: Tuple[str,str]):
    """将指定的token pair合并成一个token"""
    a, b = pair
    merged = []
    i = 0
    while i < len(word):
        if i < len(word)-1 and word[i]==a and word[i+1]==b:
            merged.append(a+b)   # 合并为一个新token
            i += 2
        else:
            merged.append(word[i])
            i += 1
    return merged

def train_bpe(texts: Iterable[str], vocab_size: int=5000, num_merges: int=None) -> Tuple[List[Tuple[str,str]], List[str]]:
    """
    训练字节级BPE
    """
    corpus = build_corpus(texts)
    base_tokens = [bytes([i]).decode('latin1') for i in range(256)]
    merges: List[Tuple[str,str]] = []
    merged_set = set()
    cur_vocab_size = 256

    # 若未指定合并次数，则由target vocab来决定
    merge_steps = num_merges or (vocab_size - 256)

    for _ in range(merge_steps):
        pfreq = pair_freq(corpus)
        if not pfreq:
            break

        # 找到出现频率最高的pair
        best_pair, _ = pfreq.most_common(1)[0]

        if cur_vocab_size + 1 > vocab_size:
            break

        merges.append(best_pair)

        # 对整个语料进行合并替换
        corpus = [merge_pair(word, best_pair) for word in corpus]

        # 将新token记入词表
        merged_set.add(best_pair[0]+best_pair[1])
        cur_vocab_size += 1

    # 追加特殊token
    special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]

    # vocab = 特殊token+ 256 byte token +BPE合并的新token
    vocab_tokens = special_tokens + base_tokens + sorted(merged_set)

    return merges, vocab_tokens



# Tokenizer类
class DeepSeekV3Tokenizer:
    def __init__(self, merges: List[Tuple[str,str]], vocab_tokens: List[str]):
        self.merges = merges
        self.vocab_tokens = vocab_tokens

        # token ↔ id映射
        self.token2id = {tok:i for i, tok in enumerate(vocab_tokens)}
        self.id2token = {i:tok for tok,i in self.token2id.items()}

        # merges pair → 排序index
        self.ranks = {pair:i for i,pair in enumerate(merges)}

        # 特殊token
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"

    def encode_chunk(self, chunk: str) -> List[str]:
        """
        对一个预分词做BPE编码：
        - 转字节token
        - 逐步应用merges
        - 处理OOV：未知token拆回字节或标记为 <unk>
        """
        tokens = bytes2tokens(chunk.encode('utf-8'))

        # 应用 PE 并规则
        for pair in self.merges:
            new_tokens = []
            i = 0
            a,b = pair
            while i < len(tokens):
                if i<len(tokens)-1 and tokens[i]==a and tokens[i+1]==b:
                    new_tokens.append(a+b)
                    i+=2
                else:
                    new_tokens.append(tokens[i])
                    i+=1
            tokens = new_tokens

        # OOV token拆回字节
        out = []
        for t in tokens:
            if t in self.token2id:
                out.append(t)
            else:
                # 拆分成字节token；如果字节token也不在词表 → <unk>
                out.extend([ch if ch in self.token2id else self.unk_token for ch in t])
        return out

    def encode(self, text: str, add_bos=False, add_eos=False, print_chunks=False):
        """
        编码完整文本：
        - 先预分词
        - 再逐chunk编码
        - 可选打印中间过程
        """
        ids = []

        if add_bos:
            ids.append(self.token2id[self.bos_token])
            if print_chunks: print(f"[Special] <bos> -> {self.token2id[self.bos_token]}")

        for chunk in pretokenize(text):
            toks = self.encode_chunk(chunk)
            chunk_ids = [self.token2id.get(t, self.token2id[self.unk_token]) for t in toks]

            if print_chunks:
                readable = []
                for t in toks:
                    try:
                        # 尝试恢复utf-8
                        r = tokens2bytes([t]).decode('utf-8', errors='ignore')
                        readable.append(r if r else t.encode('latin1').hex())
                    except:
                        readable.append(t.encode('latin1').hex())

                print(f"[Chunk] \"{chunk}\" -> {readable} -> IDs: {chunk_ids}")

            ids.extend(chunk_ids)

        if add_eos:
            ids.append(self.token2id[self.eos_token])
            if print_chunks: print(f"[Special] <eos> -> {self.token2id[self.eos_token]}")
        return ids

    def decode(self, ids: Iterable[int]):
        """
        将ID序列还原为utf-8文本：
        """
        byte_seq = bytearray()
        for i in ids:
            tok = self.id2token.get(i, self.unk_token)
            if tok in {self.pad_token, self.bos_token, self.eos_token}:
                continue
            byte_seq.extend(tokens2bytes(list(tok)))
        return byte_seq.decode('utf-8', errors='replace')

    def save(self, vocab_path: str, merges_path: str):

        # 保存vocab（token2id）
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(self.token2id, f, ensure_ascii=False, indent=2)

        # 保存merges：每个token用base64
        merges_b64 = []
        for a, b in self.merges:
            a_bytes = a.encode('latin1')
            b_bytes = b.encode('latin1')
            merges_b64.append((
                base64.b64encode(a_bytes).decode('ascii'),
                base64.b64encode(b_bytes).decode('ascii')
            ))

        with open(merges_path, 'w', encoding='utf-8') as f:
            json.dump(merges_b64, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, vocab_path: str, merges_path: str):

        # 加载vocab
        with open(vocab_path, 'r', encoding='utf-8') as f:
            token2id = json.load(f)
        vocab_tokens = [None] * (max(token2id.values()) + 1)
        for tok, idx in token2id.items():
            vocab_tokens[idx] = tok

        # 加载merges（base64 → bytes → latin1）
        with open(merges_path, 'r', encoding='utf-8') as f:
            merges_b64 = json.load(f)

        merges = []
        for a_b64, b_b64 in merges_b64:
            a = base64.b64decode(a_b64).decode('latin1')
            b = base64.b64decode(b_b64).decode('latin1')
            merges.append((a, b))
        return cls(merges, vocab_tokens)


# 提供训练函数
def train_tokenizer(texts, vocab_size=5000, num_merges=None):
    merges, vocab_tokens = train_bpe(texts, vocab_size=vocab_size, num_merges=num_merges)
    return DeepSeekV3Tokenizer(merges, vocab_tokens)

# 示例
if __name__ == "__main__":
    texts = [
        "Transformer是AI的核心技术。",
        "DeepSeek分词器支持中文、英文、emoji等多语言。",
        "Hello, 世界! 🌍🚀",
    ]

    print("训练 Tokenizer (vocab_size=1024)")
    tokenizer = train_tokenizer(texts, vocab_size=1024)
    print(f"完成训练，词表大小: {len(tokenizer.vocab_tokens)}")
    print("-"*50)

    txt = "注意力机制是AI的核心技术。 🚀 🚀"
    print(f"编码文本: {txt}")
    ids = tokenizer.encode(txt, add_bos=True, add_eos=True, print_chunks=True)

    print("-"*50)
    print("Token ID:", ids)
    decoded = tokenizer.decode(ids)
    print("解码结果:", decoded)
    print("是否可逆:", decoded == txt)
```
输入测试样例
>注意力机制是AI的核心技术。 🚀 🚀

输出
>tokens映射id，以及每个划分token对应的编码，并且对于不同位置的空格和emoji🚀对应的编码以及映射ID是相同的。

从以上代码的运行结果可以看出，分词器的token ↔ id映射仅表示token的内容，而不包含该token在句子中的相对位置。BPE或其他基于频率的合并策略是统计驱动的——它们根据token对或子串在语料中的共现频率决定合并，将常见的字节或子串压缩成更长的token。这说明分词器本身并不理解句子的抽象语义，它更像一个执行统计的模块，通过数学或概率规律重排和压缩字符序列，为上层模型*如LLM*提供可学习的离散输入单元。语义理解依赖下游模型在上下文中学习得到，并结合位置编码信息，而非由分词器直接“理解”。

## 4思考
1）有研究表明，视觉特征能够增强LLM的理解能力，但并非适用于所有语言任务。那么是否可以在视觉表征与离散 token 之间寻求一种动态“平衡点”：同时为模型提供两类表征方式，并借鉴MoE的思想设计轻量级动态路由，使模型能够在不同任务或文本片段中自动选择或融合最合适的*词——数字*映射表形式，从而显著提升跨场景的适配能力？

> 文本token的离散性限制了表达能力，视觉token可提供高密度的连续压缩表征但并不适用于所有语言场景；因此探索一种MoE风格的多表征机制，使模型能按任务动态选择文本、视觉或混合表征，以获得更丰富且具场景适配性的表示或许也值得思考。

2）能否设计一种“自适应分词器”，在训练阶段先与LLM分开训练，并通过一种特殊机制将训练好的分词器与模型结合，使其在下游任务中仍能动态学习和优化token划分策略？例如，借助微分子词模块、元学习或强化学习等方法，让分词器能够从少量对话或任务样本中自动发现最合适的token划分方式，从而降低下游任务对数据的依赖，同时提升模型的鲁棒性和泛化能力？

> 这种方式有点像半监督学习，分词器自己在“学习怎么学习”，这样即使只看到少量对话样本，它也能找到更合适的token划分方式，让模型理解语言更高效，也更不容易被新词或少量数据难住。


