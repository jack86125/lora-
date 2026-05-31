# JSON格式解析，将字符串类型的样例转换为字典
import json
# traceback.format_exc()：返回当前异常堆栈的完整字符串信息，用于精确的错误定位
import traceback
# 将Python列表转换为NumPy数组，方便后续DataLoader拼接为Tensor
import numpy as np
# tqdm进度条：在统计数据集token长度分布时提供可视化进度反馈
from tqdm import tqdm
# HuggingFace datasets库的load_dataset函数：支持直接读取文本文件(jsonl/csv/txt)并构建Dataset对象
from datasets import load_dataset
# ChatGLM的tokenizer加载器：trust_remote_code=True允许执行模型目录下的自定义分词代码
from transformers import AutoTokenizer
# partial函数：柯里化工具，预设部分参数后生成新函数，避免重复传递tokenizer和长度配置
from functools import partial

# 导入全局项目配置类ProjectConfig，获取模型路径、数据集路径、序列长度等超参数
from glm_config import *


def convert_example_chatglm(
        examples: dict,
        tokenizer,
        max_source_seq_len: int,
        max_target_seq_len: int,
    ):
    """
    将原始JSONL样本数据转换为ChatGLM-6B模型微调所需的input_ids和labels。

    核心处理逻辑：
    1. 解析每行JSON字符串，提取context和target字段
    2. 分别对context和target进行tokenize编码（不自动添加特殊标记）
    3. 对超长序列执行截断操作（预留特殊token的位置）
    4. 调用tokenizer.build_inputs_with_special_tokens()拼接为完整序列
    5. 构造labels：context部分及填充部分标记为-100（不计算损失），target部分保留真实token ID

    Args:
        examples (dict): HuggingFace datasets传递的批量数据字典，
            结构为 {"text": [json_string_1, json_string_2, ...]}
            每条json_string包含{"context": "指令+输入文本", "target": "答案文本"}两个字段
        tokenizer: ChatGLM的tokenizer实例，负责文本到token ID的双向转换
        max_source_seq_len (int): context部分允许的最大token数（需为[gMASK]预留1个位置）
        max_target_seq_len (int): target部分允许的最大token数（需为<sop>和<eop>预留2个位置）

    Returns:
        dict: 包含两个NumPy数组的字典
            {
                'input_ids': np.array,  shape=(样本数, max_seq_length)，dtype=int64
                'labels':    np.array,  shape=(样本数, max_seq_length)，dtype=int64
            }
            labels中值为-100的位置在交叉熵损失计算时会被忽略(ignore_index)
    """
    # 初始化输出字典，input_ids和labels分别收集每个样本的token序列
    tokenized_output = {
        'input_ids': [],
        'labels': []
    }
    # 计算每条样本的最大序列长度 = source最大长度 + target最大长度
    # 例如默认配置下：100 + 100 = 200
    max_seq_length = max_source_seq_len + max_target_seq_len

    # 遍历批量数据中的每一条样本（text字段是JSON字符串列表）
    for example in examples['text']:
        # 异常捕获：防止个别脏数据导致整个batch处理中断
        try:
            # JSON反序列化：将字符串 '{"context": "...", "target": "..."}' 解析为Python字典
            example = json.loads(example)
            # 提取context字段：包含指令模板(Instruction: ...\nInput: ...\nAnswer: )和待抽取的文本
            context = example["context"]
            # 提取target字段：包含标准答案，格式为Markdown代码块包裹的JSON数组
            target = example["target"]

            # ==================== 第一步：Tokenize编码 ====================
            # 对context文本进行编码，add_special_tokens=False表示不自动添加[CLS]/[SEP]等标记
            # 因为后续会通过build_inputs_with_special_tokens手动添加ChatGLM特有的特殊标记
            # 返回值prompts_ids是token ID的Python列表，例如：[1525, 10, 3342, 5, 76331, ...]
            prompts_ids = tokenizer.encode(
                text=context,
                add_special_tokens=False
            )

            # 对target答案文本进行编码，同样不自动添加特殊标记
            # 返回值target_ids是token ID的Python列表，例如：[6589, 10, 5, 130005, ...]
            target_ids = tokenizer.encode(
                text=target,
                add_special_tokens=False
            )

            # ==================== 第二步：超长截断 ====================
            # context截断逻辑：当prompt的token数量超过max_source_seq_len时
            # 截取前(max_source_seq_len - 1)个token，空出最后1个位置用于放置[gMASK]标记
            # >= 而非 > 的原因是：当长度恰好等于上限时，已经没有空间容纳[gMASK]了
            if len(prompts_ids) >= max_source_seq_len:
                prompts_ids = prompts_ids[:max_source_seq_len - 1]

            # target截断逻辑：当target的token数量超过(max_target_seq_len-1)时
            # 截取前(max_target_seq_len - 2)个token，空出2个位置：开头放<sop>(BOS)、结尾放<eop>(EOS)
            if len(target_ids) >= max_target_seq_len - 1:
                target_ids = target_ids[:max_target_seq_len - 2]

            # ==================== 第三步：拼接为完整输入序列 ====================
            # ChatGLM的build_inputs_with_special_tokens内部逻辑（位于tokenization_chatglm.py L323-L344）：
            #   prompts_ids + [gMASK] + [BOS/<sop>] + target_ids + [EOS/<eop>]
            # 最终序列结构示意：
            #   [context_tokens...] [gMASK(130001)] [BOS(130004)] [target_tokens...] [EOS(130005)]
            input_ids = tokenizer.build_inputs_with_special_tokens(prompts_ids, target_ids)

            # ==================== 第四步：构造labels ====================
            # 定位BOS token在input_ids中的索引位置
            # BOS(<sop>)标记了target答案文本的起始位置
            # 如果input_ids中不存在bos_token_id，index()会抛出ValueError并被外层try捕获
            context_length = input_ids.index(tokenizer.bos_token_id)
            # [gMASK]的索引位置 = BOS位置 - 1，它是context和target之间的分界标记
            mask_position = context_length - 1

            # labels构造规则：
            #   前context_length个位置(含[gMASK])：全部填充-100，表示不参与损失计算
            #   从BOS开始到序列末尾(含EOS)：保留原始token ID作为监督信号
            # -100是PyTorch CrossEntropyLoss的默认ignore_index，对应位置的预测不会贡献梯度
            # 这确保了模型只学习生成target答案部分，而不学习复述prompt上下文
            labels = [-100] * context_length + input_ids[mask_position + 1:]

            # ==================== 第五步：填充到固定长度 ====================
            # 计算需要在末尾填充的token数量 = 最大序列长度 - 当前实际长度
            pad_len = max_seq_length - len(input_ids)

            # 用PAD token ID(130002)填充input_ids到max_seq_length
            # 右侧填充：在序列末尾追加pad_token_id，保证同一batch内所有样本长度一致
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_len

            # 对labels也进行等量填充，填充值使用-100（同样不参与损失计算）
            labels = labels + [-100] * pad_len

            # ==================== 第六步：收集当前样本的处理结果 ====================
            # 将处理好的input_ids列表追加到输出字典中
            tokenized_output['input_ids'].append(input_ids)
            # 将处理好的labels列表追加到输出字典中
            tokenized_output['labels'].append(labels)

        except:
            # 捕获所有异常（JSON解析失败、字段缺失、tokenizer错误等）
            # traceback.format_exc()打印完整的异常类型、异常信息和调用堆栈
            # continue跳过当前错误样本，继续处理剩余数据，保证整个数据集的鲁棒性
            print(f'"{example}" -> {traceback.format_exc()}')
            continue

    # 将input_ids和labels的Python列表转换为NumPy数组
    # 这是datasets.map()函数的要求：批量处理函数必须返回NumPy数组或列表的字典
    # np.array转换后dtype默认为int64，与后续PyTorch LongTensor格式兼容
    for k, v in tokenized_output.items():
        tokenized_output[k] = np.array(v)

    # 返回处理完成的tokenized数据字典
    return tokenized_output


def get_max_length(
        tokenizer,
        dataset_file: str
    ):
    """
    统计指定数据集文件中source(context)和target的token长度分布。

    该函数用于数据探索阶段，帮助确定max_source_seq_len和max_target_seq_len的合理取值。
    输出最大值、平均值和中位数三个统计量，以此评估截断阈值对数据覆盖率的实际影响。

    Args:
        tokenizer: ChatGLM的tokenizer实例，用于计算文本的token数量
        dataset_file (str): JSONL格式数据集的绝对路径，每行为一个{"context":..., "target":...}对象

    输出示例:
        【Source Sequence】 Max: 156, Avg: 42, Middle: 35.
        【Target Sequence】 Max: 89, Avg: 28, Middle: 22.
    """
    # 初始化两个列表，分别收集每条样本的source和target token长度
    source_seq_len_list = []
    target_seq_len_list = []

    # 以UTF-8编码打开JSONL文件
    with open(dataset_file, 'r', encoding='utf-8') as f:
        # readlines()一次性读取全部行，配合tqdm显示处理进度条
        # 注意：对于超大文件(>1GB)建议改为逐行流式读取以避免内存溢出
        for line in tqdm(f.readlines()):
            # JSON反序列化：将当前行的JSON字符串解析为Python字典
            line = json.loads(line)
            # 对context字段进行完整的tokenize编码，获取token数量
            # encode()返回token ID列表，len()即为token数
            source_len = tokenizer.encode(line['context'])
            # 将当前样本的context token长度加入列表
            source_seq_len_list.append(len(source_len))

            # 对target字段进行完整的tokenize编码，获取token数量
            target_len = tokenizer.encode(line['target'])
            # 将当前样本的target token长度加入列表
            target_seq_len_list.append(len(target_len))

    # 输出source序列长度统计信息
    # Max：最大长度，用于确定max_source_seq_len的上限参考
    # Avg：平均长度，反映数据集的整体token消耗水平
    # Middle：中位数(排序后取中间值)，相比平均值更能抵抗极端长样本的干扰
    print(f"【Source Sequence】 Max: {max(source_seq_len_list)}, Avg: {int(sum(source_seq_len_list) / len(source_seq_len_list))}, Middle: {sorted(source_seq_len_list)[int(len(source_seq_len_list) / 2)]}.")

    # 输出target序列长度统计信息，统计维度同上
    print(f"【Target Sequence】 Max: {max(target_seq_len_list)}, Avg: {int(sum(target_seq_len_list) / len(target_seq_len_list))}, Middle: {sorted(target_seq_len_list)[int(len(target_seq_len_list) / 2)]}.")


# ==================== 脚本入口：直接运行此文件时执行数据长度统计 ====================
if __name__ == '__main__':
    # 实例化项目配置对象，获取所有超参数（模型路径、数据路径等）
    pc = ProjectConfig()

    # 使用HuggingFace datasets库加载训练集JSONL文件
    # 'text'类型会自动将文件每行解析为{"text": "行内容"}格式的Dataset
    # data_files参数指定数据文件路径映射，键为数据集分片名称
    train_dataset = load_dataset('text', data_files={'train': pc.train_path})

    # 加载ChatGLM-6B的tokenizer
    # trust_remote_code=True必须设置：因为ChatGLM-6B的tokenization代码不在transformers官方库中
    # from_pretrained会从本地模型目录(pc.pre_model)加载vocab文件和自定义分词脚本
    tokenizer = AutoTokenizer.from_pretrained(pc.pre_model, trust_remote_code=True)

    # 调用get_max_length统计训练集的token长度分布
    # 根据输出的Max/Avg/Middle统计值来合理设置glm_config.py中的
    # max_source_seq_len和max_target_seq_len参数
    get_max_length(tokenizer, pc.train_path)
