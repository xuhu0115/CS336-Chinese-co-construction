# 分词器
分词器是连接人类自然语言与机器计算的“关键桥梁”，负责将非结构化的文本高效地编码为模型可理解的数字序列（Tokens）。其地位至关重要，它决定了模型的输入形式。我们可以将大语言模型（LLM）视为一位阅读者，分词器则执行了文本的“认知拆解”即将连续的文本流切分为具备最小语义或功能意义的基本单元，类似于人类阅读时对词汇或音节的本能识别。本章节将剖析分词器实现文本“数字化”的核心原理，并探讨BPE等主流分词算法的实际应用。

<p align="center">
<img width="600" height="350" alt="0e717e4cf0df64875a271123bf962631" src="https://github.com/user-attachments/assets/a53dd727-8682-4314-92dc-3b3e18dfaf58" />
</p>

### 1训练分词器
在我们把海量数据喂给大模型之前，必须先经过一道关键工序——分词。分词器常被视为LLM的一部分，但它其实拥有独立的训练生命周期。我们需要利用正则表达式对原始文本进行预处理，并统计构建出一套高效的**词元———数字**离散序列转化映射表（vocab），这个映射过程决定了模型眼中的世界是由字、词还是更碎的片段组成的，直接影响后续模型对语义的理解效率。也正因如此，[分词器](https://tiktokenizer.vercel.app/?model=deepseek-ai%2FDeepSeek-R1)虽独立训练却与LLM保持着“强耦合”的关系。

>训练一个用于现代大型语言模型的分词器可以拆成四步：准备语料 → 初始化基础单元 → 统计并迭代合并 → 输出产物并用于编码、解码。

#### 第一步 准备语料

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

5. 在多语言或混合语料场景中，应当统计每种语言的占比并考虑是否对低资源语言做过采样或专门保留，以避免词表被高频语言主导导致低频语言表现差即语料类型、语言不平衡会导致`token`碎片化、占用更多`token`、增加Transformer计算成本并降低低资源语言性能。
   
   例如准备一种可以支持四种语言的分词器，这里假设提前收集到的原始未经过第7步处理的各语言原始语料占比如下：
   
| 语言 | 语料量 |
| :--- | ---: |
| 中文 | 200 GB |
| 英文 | 150 GB |
| 法语 | 10 GB |
| 韩文 | 5 GB |

>这是一个典型的多语言语料不平衡场景。若将上述语料不经处理直接混合训练分词器，其统计过程会被中文和英文主导，导致法语与韩文的常见字串在合并阶段难以进入高频统计，从而无法占据足够的词表空间，最终在`vocab`中会出现大量被切得过碎的`token`，形成严重碎片化，下游LLM在法语与韩文任务上会因此表现显著劣化。

因此在准备语料的第4步应先按语言统计语料占比，并根据目标能力设定合理的采样策略。例如将语料比例调整为 中文:英文:法语:韩文=4:4:1:1或者采用完全均衡策略。通过对高资源语言下采样或对低资源语言过采样、增强，可以获得更符合目标分布的训练语料，再使用验证集评估各语言的token覆盖率、平均其碎片化程度，以确保最终词表在多语言任务中具备稳定且均衡的表示能力。
   
5. 在语料准备过程中还需要考虑保留哪些特殊标记例如句子起始、句子结束、填充、未知、掩码等，并将这些特殊token的设计纳入最终词表预算以避免表大小超标。
    
6. 建议保留一小部分未参与训练的验证语料比如训练集:验证集＝99:1，用来在训练过程中评估分词器对真实文本的编码效率与平均token长度等统计指标。

#### 第二步 初始化基础单元

1. 预分词的主要任务是把原始文本切分成可统计、可合并的最小基础单元例如字符、字节或预定义字片段。常见策略包括基于空格和标点的切分、Unicode类别划分或直接采用字节级分割。
   
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
                  # 不同类 → 切分
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
              print(f"{ch} 转化为 UTF-8 字节序列：{hex_bytes}")
      
              # 加入 token 列表
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

在LLM的Token划分策略中，通常分为基于规则的预分词如空格标点切分、Unicode类别划分和更底层的UTF-8字节级切分。前两种策略虽然常见，但在处理缺乏明确分隔符以及统一Unicode类别的长文本<ins>如连续的中文长句或代码字符串时</ins>存在局限，它们难以在预处理阶段进行有效断句，这可能迫使模型在后续处理中退化为更细碎的字符粒度，导致序列变长进而增加注意力机制的计算负担。相比之下，基于UTF-8字节级的token划分策略具有极强的通用性，它通过将所有文本分解为字节序列彻底打破了语言和字符类型的限制，并从根本上减少了未登录词（OOV）问题，保证了模型对任何输入文本的覆盖率。不过由于它将文本初始化为最细粒度的字节序列，因此在构建词表时往往需要通过更多次迭代的共现统计与合并，才能将零散的字节压缩为紧凑且具有语义价值的Token ID映射。

>未登录词是当LLM模型在处理新的、实际应用的文本时，如果遇到一个词汇表中没有的Token，那么这个Token就被视为一个OOV。

2. 对大多数以空格为词边界的语言，可先用正则表达式按单词边界和标点进行初步切割，而对中文、日文等不以空格为词界的语言则通常采用逐字符或基于字的初始单元来保证覆盖性。

3. 预分词时应当明确是否做小写化或保留大小写信息，二者会影响词表大小、上下文信息与下游模型的语言理解细粒度；如果保留大小写则需要将大小写相关变体视为不同基础单元。

4. 在预分词阶段同时可以标注并保留一些常见实体或标记例如表情符号、URL、数字序列和特殊符号等以便训练时能优先学习到这些高频模式的合并行为。
   
5. 预分词生成的基础单元序列将作为后续统计合并的输入，务必保存该序列与对应位置信息以便在训练过程反复高效更新。
   
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

#### 第三步 统计并迭代合并 ——> 核心步骤

1. 以BPE为代表的合并算法从基础单元开始，通过统计所有相邻基础单元对在语料中的共现频率，将频次最高的一对合并为新的token，并把该合并规则记录到merges列表中；这一过程重复进行直到达到要求的词表大小或满足其他终止条件。
2. 具体实现上需要维护一个频率表用于快速统计相邻对出现次数，并在每次合并后高效更新相关对的计数；对于大规模语料应采用哈希表、优先队列或者 Trie 等数据结构以减少重复扫描与更新开销。
3. 合并时要区分“共现频率”与“相似度”概念，BPE只关注相邻对的出现次数而非语义相似度，因此其合并结果反映的是统计上的片段组合而不是语言学上的词义单位。
4. 在合并过程中要注意添加保留的特殊tokens（如[PAD]、[UNK]、[CLS]等）到最终词表中，并确保这些tokens在编码解码时具有固定的id与优先级，不被后续合并覆盖。
5. 对于包含多语言或特殊字符的语料，建议采用BPE或在合并策略中加入“字节后备”机制，以便在遇到罕见字符时仍能用字节序列表达，保证编码的完备性与可逆性。
6. 合并迭代的终止条件可以是词表达到目标大小、合并步数达到上限、或者频率阈值降到某个程度；不同的终止条件会直接影响到平均token长度、下游模型的序列长度和训练效率。
7. 在大语料训练时还应考虑并行化或分布式统计方法，以及内存受限时的分块训练策略，必要时采用近似计数或采样方法以在可接受的资源下获得近似最优的合并序列。
8. 训练期间应实时评估若干指标，如平均每条文本的token数、未登录词率（OOV）、最大最小token长度分布以及合并后常见token示例，帮助判断词表是否符合预期需求。
9. 对合并顺序保持可复现的记录非常重要，一旦merges表确定就能确保不同时间或不同环境中训练得到的词表与编码器行为一致，便于模型重训、部署和版本管理。

#### 第四步 输出产物并用于编码与解码

1. 完成合并后需要导出至少两个核心产物：一是包含token与对应id的词表（vocab）；二是按顺序记录的合并规则表，这两者共同决定了编码器与解码器的行为与可逆性。
2. 在部署分词器时，应实现一致的编码与解码逻辑，确保编码得到的id序列能精确还原为原始文本，或在必要时还原为语义等价的文本形式。
3. 对下游模型使用时建议先在一小部分验证集上统计分词器的性能指标，包括平均token长度、token化后的最大长度分布以及对关键实体的拆分情况，以便评估是否需要调整词表大小或合并策略。
4. 若需扩表例如加入新领域专用词或品牌名，应优先采用增量合并或微调merges表的方式，避免重训整个分词器以节省计算资源，并在扩表后做回归测试以确认无副作用。
5. 将词表和merges表进行版本管理例如语义化的版本号与变更日志，并在模型训练与推理管道中显式固定使用的分词器版本，防止不同阶段使用不一致的分词策略导致性能不可复现。
6. 最后在生产部署时还需要考虑序列长度上限、实时编码延迟以及内存占用等工程问题，并在客户端或服务端实现相同的分词器逻辑以保持输入处理的一致性。

### 2常用的分词器
在NLP的发展历程中，分词策略经历了几次重要的演变。我们主要关注四种最典型的范式：字符、字节、词级、BPE分词器，以及结合课程[lecture1](https://stanford-cs336.github.io/spring2025-lectures/?trace=var/traces/lecture_01.json)各个分词器伪代码转化为python代码实践。

#### 2.1 字符分词器

#### 原理介绍

这是最直观、最简单的分词方式，它将文本拆解为最小的字符单位如英语中的字母 a, b, c、中文里的单字 中, 国。

  - **优点：**
      - **词表极小：** 英语只需包含26个字母+符号；中文只需包含常用汉字（约几千个）。
      - **无OOV问题：** 任何生僻词都是由基础字符组成的，不会出现“未知词”。
  - **缺点：**
      - **序列过长：** 一句话变成字符后，长度会增加数倍，大大消耗LLM宝贵的上下文窗口，从而加大LLM的transformer计算现存消耗。
      - **语义稀疏：** 单个字符（如t）通常不具备独立的语义，模型需要更深的网络层数来组合出意义。

   实现示例：
   ```python
   # 简易字符分词器实现
   text = "Hello，World!"
   
   # 1.构建词表(去重并排序)
   vocab = sorted(list(set(text)))
   print(f"词表: {vocab}") 
   
   # 2.创建映射 (字符->ID)
   char_to_id = {char: i for i, char in enumerate(vocab)}
   id_to_char = {i: char for i, char in enumerate(vocab)}
   
   # 3.编码(Encode)
   encoded = [char_to_id[c] for c in text]
   print(f"原文: '{text}' -> 编码后: {encoded}")
   
   # 4.解码(Decode)
   decoded = "".join([id_to_char[i] for i in encoded])
   print(f"解码后: '{decoded}'")
   ```

   
输入
> Hello，World!

输出
> 词表: ['!', 'H', 'W', 'd', 'e', 'l', 'o', 'r', '，']
> 
> 原文: 'Hello，World!' -> 编码后: [1, 4, 5, 5, 6, 8, 2, 6, 7, 5, 3, 0]
> 
> 解码后: 'Hello，World!'


#### 2.2 字节分词器

#### 原理介绍

计算机底层存储文本本质上都是**字节**，在UTF-8编码中，英文通常占1个字节，汉字通常占3个字节。字节分词器直接对二进制字节进行操作。

  - **核心逻辑：** 不再维护“字符”的词表，而是维护一个大小为256的基础词表（0x00到0xFF）。
  - **应用：** 现代LLM（如GPT-4, Llama）通常不单独使用纯字节分词，而是将字节作为BPE的基础单位*即BBPE，这样可以彻底解决跨语言和特殊符号（如emoji 🌍等）的编码问题。

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
- 因此`token数量 = UTF-8字节数`。

所以

$$
compression_{ratio}
= \frac{\text{UTF-8 字节长度}}{\text{token 数量}}
= \frac{N}{N}
= 1
$$

也就是说，字节级分词器完全不具备压缩能力：每个字节对应一个token，不会产生更长或更短的词片段。

#### 2.3 词级分词器

#### 原理介绍

在深度学习早期（如RNN时代）这是最主流的方法。它基于空格（英文）或分词算法（中文）将文本切分为具备独立语义的“词”。

  - **优点：** Token保留了完整的语义信息（"apple" 直接对应一个Token ID）。
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

#### 2.4 BPE分词器

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

>所以`</w>`的核心作用是保证单词完整性，并让编码可逆。
#### 四种分词器对比表

| 分词器类型 | 粒度 | 词表大小 | 未登录词 (OOV) | 序列长度 | 代表模型 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **字符级** | 细 | 小 (100-5k) | 无 | 非常长 | Char-RNN |
| **词级** | 粗 | 极大 (>100k) | 严重 | 短 | Word2Vec, GloVe |
| **BPE** | **中 (自适应)** | **适中 (30k-100k)** | **极少** | **适中** | **GPT-4, Llama 3** |

