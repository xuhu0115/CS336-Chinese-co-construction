from evalscope.collections import CollectionSchema, DatasetInfo

from evalscope.collections import WeightedSampler
from evalscope.utils.io_utils import dump_jsonl_data

import os
from evalscope import TaskConfig, run_task

# 嵌套结构示例：数学组 + 推理组
schema = CollectionSchema(name='reasoning_index', datasets=[
    CollectionSchema(name='math', weight=0.5, datasets=[
        DatasetInfo(name='gsm8k', weight=0.5, tags=['en']),
        DatasetInfo(name='aime25', weight=0.5, tags=['en']),
    ]),
    CollectionSchema(name='logic', weight=0.5, datasets=[
        DatasetInfo(name='arc', weight=0.5, tags=['en']),
        DatasetInfo(name='ceval', weight=0.5, tags=['zh'], args={'subset_list': ['logic']}),
    ]),
])

# 打印查看归一化后的权重分布
print(schema.flatten())


# 初始化加权采样器
sampler = WeightedSampler(schema)

# 采样 100 条数据作为最终测试集
# 根据权重，知识问答 30 条，长文本检索 30 条，指令遵循 40 条
# 实际采样数量可根据需要调整
mixed_data = sampler.sample(count=10)

# 将混合好的数据保存为 JSONL 文件，这就是你的“指数评测集”
dump_jsonl_data(mixed_data, 'data/index_testset.jsonl')

task_cfg = TaskConfig(
    # model='qwen2.5-14b-instruct', # 评测模型
    # # 使用一个提供 OpenAI 兼容接口的模型进行评测
    # # 可以是云上的API，也可以是通过vllm等框架部署的本地模型
    # api_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    # api_key=os.getenv('DASHSCOPE_API_KEY'),
    # eval_type='openai_api',
    model='openai-community/gpt2', # 评测模型
    
    # 关键配置：指定数据集为 'data_collection' 模式
    datasets=['data_collection'],
    dataset_args={
        'data_collection': {
            'local_path': 'data/index_testset.jsonl', # 指向刚才生成的文件
            'shuffle': True # 打乱顺序
        }
    },
    eval_batch_size=5, # 根据你的 API 并发限额调整
    generation_config={
        'temperature': 0.0 # 评测通常设为 0 以保证结果可复现
    },
    use_cache="outputs/20260119_232050", # 复用本地缓存路径的推理结果和评测结果
)

run_task(task_cfg=task_cfg)