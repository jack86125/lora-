# Project Documentation: ptune_chatglm

---

## 1. 项目概述与背景

`ptune_chatglm` 是一个基于 **ChatGLM-6B** 大语言模型的 **P-Tuning / LoRA 参数高效微调**（Parameter-Efficient Fine-Tuning, PEFT）项目。核心目标是对 ChatGLM-6B 进行下游任务适配，使其具备从自然语言文本中抽取 **SPO（Subject-Predicate-Object）三元组结构化信息** 的能力。

项目设计了两种指令类型的训练数据混合训练：一类以"阅读理解器"身份、另一类以"SPO 抽取器"身份，覆盖人物属性、影视作品、书籍出版、生物分类、地理信息、企业机构等领域的实体关系抽取任务。

**核心亮点**：
- 使用 LoRA（Low-Rank Adaptation）技术，仅训练极小比例的模型参数（适配器），极大降低了显存和训练成本。
- 支持 P-Tuning v1 / P-Tuning v2 作为备选微调方案，可在配置中灵活切换。
- 集成 FP16 半精度、Gradient Checkpointing、混合精度训练等显存优化手段。
- 完整的训练-验证-推理 Pipeline。

---

## 2. 技术栈

### 2.1 编程语言

| 语言 | 版本  |
|------|-------|
| Python | 3.8+ |

### 2.2 深度学习框架

| 框架/库 | 用途 |
|---------|------|
| **PyTorch** | 核心深度学习框架，负责张量运算、自动求导、GPU 加速 |
| **Transformers** | HuggingFace 生态，提供 ChatGLM-6B 模型加载、Tokenizer、Scheduler 等 |
| **PEFT** | 参数高效微调库，本项目使用其中的 `LoraConfig` + `get_peft_model` |
| **torch.cuda.amp** | PyTorch 混合精度训练模块（`autocast`），LoRA 模式下在 GPU 上启用 |

### 2.3 数据处理

| 库 | 用途 |
|----|------|
| **datasets**（HuggingFace） | `load_dataset('text', ...)` 加载 JSONL 数据集，`map()` 批量预处理 |
| **numpy** | tokenized 输出转换为 numpy 数组 |
| **tqdm** | 数据统计时的进度条显示 |

### 2.4 辅助工具

| 库 | 用途 |
|----|------|
| **rich** | 推理脚本 `__main__` 中的美化打印（`from rich import print`） |
| **functools.partial** | 柯里化，对 `convert_example_chatglm` 预设 tokenizer 和长度参数 |
| **copy** | `deepcopy` 模型用于 LoRA 权重合并与保存 |

---

## 3. 项目结构与文件组织

```
ptune_chatglm/
│
├── project_doc.md                        # 本文件：项目完整文档
│
├── __init__.py                           # 包初始化标记（utf-8 编码声明）
├── glm_config.py                         # 全局配置类：所有超参数集中管理
├── train.py                              # 训练主入口：模型加载 → 训练循环 → 保存
├── inference.py                          # 推理脚本：加载训练好的模型进行 SPO 抽取
│
├── data/                                 # 数据目录
│   ├── dataset.jsonl                     # 数据子集（"SPO 抽取器" 风格指令样本）
│   ├── mixed_train_dataset.jsonl         # 训练集（"阅读理解器" 风格指令样本）
│   └── mixed_dev_dataset.jsonl           # 验证集（"阅读理解器" 风格指令样本）
│
├── data_handle/                          # 数据处理模块
│   ├── __init__.py                       # 空文件，包标记
│   ├── data_preprocess.py                # 核心数据预处理：tokenization + label构造
│   ├── data_loader.py                    # DataLoader 构建：加载 JSONL → 预处理 → 返回迭代器
│   └── test.py                           # 调试脚本（traceback 示例 / 零除测试）
│
└── utils/                                # 工具模块
    ├── __init__.py                       # 空文件，包标记
    ├── common_utils.py                   # 通用工具：CastOutputToFloat / second2time / save_model
    └── test.py                           # second2time 函数的调试测试脚本
```

### 3.1 文件职责速查表

| 文件 | 核心职责 | 入口函数 |
|------|---------|---------|
| `glm_config.py` | 定义所有训练超参数 | `ProjectConfig.__init__()` |
| `train.py` | 完整训练 Pipeline | `model2train()` |
| `inference.py` | 模型推理 | `inference(model, tokenizer, instruction, sentence)` |
| `data_handle/data_preprocess.py` | 数据 tokenize 与 label 构造 | `convert_example_chatglm()` |
| `data_handle/data_loader.py` | 构建 DataLoader | `get_data()` |
| `utils/common_utils.py` | 模型保存与工具函数 | `save_model()` / `CastOutputToFloat` / `second2time()` |

---

## 4. 核心模块详细说明

### 4.1 `glm_config.py` —— 全局配置类

**类**: `ProjectConfig`

集中管理所有超参数，采用单一配置源（Single Source of Truth）的设计模式。所有其他模块均通过 `from glm_config import *` 导入其实例 `pc`。

| 参数 | 类型 | 当前值 | 说明 |
|------|------|--------|------|
| `device` | `str` | `'cuda:0'` 或 `'cpu'` | 自动检测 GPU 可用性 |
| `pre_model` | `str` | 本地 ChatGLM-6B 路径 | 预训练模型的绝对路径 |
| `train_path` | `str` | `mixed_train_dataset.jsonl` | 训练集 JSONL 文件路径 |
| `dev_path` | `str` | `mixed_dev_dataset.jsonl` | 验证集 JSONL 文件路径 |
| `use_lora` | `bool` | `True` | 是否启用 LoRA 微调 |
| `use_ptuning` | `bool` | `False` | 是否启用 P-Tuning 微调 |
| `lora_rank` | `int` | `8` | LoRA 低秩矩阵的秩（r） |
| `batch_size` | `int` | `4` | 训练/验证的批次大小 |
| `epochs` | `int` | `2` | 训练总轮数 |
| `learning_rate` | `float` | `3e-5` | 优化器学习率 |
| `weight_decay` | `float` | `0` | 权重衰减系数（L2 正则化） |
| `warmup_ratio` | `float` | `0.06` | 学习率预热步数占比 |
| `max_source_seq_len` | `int` | `100` | context（source）文本的 token 最大长度 |
| `max_target_seq_len` | `int` | `100` | target（答案）文本的 token 最大长度 |
| `logging_steps` | `int` | `10` | 每隔多少步打印训练日志 |
| `save_freq` | `int` | `200` | 每隔多少步进行验证评估与模型保存 |
| `pre_seq_len` | `int` | `200` | P-Tuning 伪 token 序列长度（仅 P-Tuning 模式生效） |
| `prefix_projection` | `bool` | `False` | `False`=P-Tuning v1，`True`=P-Tuning v2 |
| `save_dir` | `str` | 检查点目录 | 模型保存的根目录 |

---

### 4.2 `data_handle/data_preprocess.py` —— 数据预处理

#### 4.2.1 `convert_example_chatglm(examples, tokenizer, max_source_seq_len, max_target_seq_len)`

**职责**: 将原始 JSONL 数据转换为 ChatGLM 模型可接收的 `input_ids` + `labels` 张量。

**输入**:
```python
examples = {
    "text": [
        '{"context": "...", "target": "..."}',
        '{"context": "...", "target": "..."}',
        ...
    ]
}
```

**处理流程**:

```
┌──────────────────────────────────────────────────────────────┐
│ 1. 遍历 examples['text']，逐条 json.loads() 解析           │
├──────────────────────────────────────────────────────────────┤
│ 2. 分别对 context 和 target 进行 tokenizer.encode()         │
│    ── add_special_tokens=False（不自动添加特殊标记）        │
├──────────────────────────────────────────────────────────────┤
│ 3. 截断处理                                                  │
│    ── context: 若 len >= max_source_seq_len                 │
│       则截取 [:max_source_seq_len - 1]（预留 [gMASK] 位）   │
│    ── target:  若 len >= max_target_seq_len                 │
│       则截取 [:max_target_seq_len - 2]（预留 <sop>+<eop>）  │
├──────────────────────────────────────────────────────────────┤
│ 4. 调用 tokenizer.build_inputs_with_special_tokens()        │
│    拼接为: prompts_ids + [gMASK] + [BOS] + target_ids +[EOS]│
├──────────────────────────────────────────────────────────────┤
│ 5. 构造 labels                                              │
│    ── 找到 input_ids 中 BOS token 的位置 (context_length)   │
│    ── BOS 之前的所有 token（含 [gMASK]）标记为 -100         │
│    ── BOS 及之后的所有 target token 作为真实 label           │
│    ── 末尾 PAD token 标记为 -100                            │
├──────────────────────────────────────────────────────────────┤
│ 6. 右侧 PAD 填充到 max_seq_length                           │
│    ── input_ids 用 pad_token_id 填充                        │
│    ── labels 用 -100 填充                                   │
├──────────────────────────────────────────────────────────────┤
│ 7. 汇总为 numpy array 后返回                                │
└──────────────────────────────────────────────────────────────┘
```

**输出格式**:
```python
{
    'input_ids': np.array([[1525, 10, ..., 130005, ...], ...]),  # shape: (N, max_seq_length)
    'labels':    np.array([[-100, -100, ..., 822, ...], ...])    # shape: (N, max_seq_length)
}
```

**labels 设计原理**:

`labels` 中值为 `-100` 的位置在计算交叉熵损失时会被 PyTorch 忽略（对应 `ignore_index`）。因此：
- **context 部分**（包括 `[gMASK]`）→ `-100`：不计算损失，模型不学习预测 prompt
- **target 部分**（从 `<sop>` 到 `<eop>`）→ 保留原始 token ID：模型学习生成正确答案
- **PAD 填充** → `-100`：不参与损失计算

#### 4.2.2 `get_max_length(tokenizer, dataset_file)`

**职责**: 统计指定数据集文件中 source 和 target 的 token 长度分布（最大值、平均值、中位数），用于辅助确定 `max_source_seq_len` 和 `max_target_seq_len` 的合理取值。

> **注意**: 当前实现使用 `f.readlines()` 全量读取，对大文件存在内存压力。

---

### 4.3 `data_handle/data_loader.py` —— 数据加载器

#### 4.3.1 `get_data()`

**职责**: 加载训练集和验证集 JSONL 文件，应用预处理，构建 PyTorch DataLoader。

**处理流程**:

```
load_dataset('text', data_files={train, dev})
        │
        ▼
dataset.map(convert_example_chatglm, batched=True)
        │  ── 通过 partial 预设 tokenizer, max_source_seq_len, max_target_seq_len
        │
        ▼
train_dataloader = DataLoader(train_dataset, shuffle=True, ...)
dev_dataloader   = DataLoader(dev_dataset, shuffle=False, ...)
        │  ── collate_fn=default_data_collator
        │  ── batch_size=pc.batch_size
```

**依赖关系**: `get_data()` → `convert_example_chatglm()` (通过 `partial` 绑定)

---

### 4.4 `utils/common_utils.py` —— 通用工具

#### 4.4.1 `CastOutputToFloat(nn.Sequential)`

```python
class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)
```

**职责**: 装饰器类，将模型的输出头（`lm_head`）的输出转为 `float32`。用于 LoRA 训练模式下，确保 `lm_head` 精度与 LoRA 适配器兼容（LoRA 适配器在 `float16` 下训练，但输出头需要 `float32` 以保证数值稳定）。

#### 4.4.2 `second2time(seconds: int) -> str`

**职责**: 将秒数转换为 `HH:MM:SS` 格式字符串。训练日志中用于显示预估剩余时间（ETA）。

#### 4.4.3 `save_model(model, cur_save_dir: str)`

**职责**: 保存当前模型到指定目录。

| 模式 | 保存策略 |
|------|---------|
| **LoRA 模式** | `copy.deepcopy(model)` → `merge_and_unload()` 合并 LoRA 权重到基础模型 → `save_pretrained()` |
| **P-Tuning 模式** | 直接 `model.save_pretrained()` |

> **设计原因**: LoRA 模式下直接 `save_pretrained()` 仅保存 adapter 权重。为了推理时直接加载完整模型，需要先将 LoRA 权重合并回基础模型再保存。

---

### 4.5 `train.py` —— 训练主流程

#### 4.5.1 `evaluate_model(model, dev_dataloader) -> float`

**职责**: 在验证集上计算当前模型的平均损失值。

**流程**:
1. `model.eval()` + `torch.no_grad()`
2. 遍历验证集 DataLoader，逐 batch 前向计算 loss
3. 如果 `use_lora=True`，使用 `autocast()` 混合精度上下文
4. 收集所有 loss 值，返回平均值
5. 恢复 `model.train()` 状态

#### 4.5.2 `model2train()` —— 完整训练 Pipeline

这是项目的核心函数，包含模型初始化到训练完成的全部流程：

```
┌─────────────────────────────────────────────────────────┐
│ Step 1: 加载 Tokenizer 和 Config                       │
│   AutoTokenizer.from_pretrained()                       │
│   AutoConfig.from_pretrained()                          │
├─────────────────────────────────────────────────────────┤
│ Step 2: P-Tuning 分支（use_ptuning=True 时）           │
│   config.pre_seq_len = pc.pre_seq_len                   │
│   config.prefix_projection = pc.prefix_projection       │
├─────────────────────────────────────────────────────────┤
│ Step 3: 加载 ChatGLM-6B 模型                            │
│   AutoModel.from_pretrained()                           │
│   → model.half()（FP16 半精度）                         │
│   → gradient_checkpointing_enable()                     │
│   → enable_input_require_grads()                        │
│   → model.config.use_cache = False                      │
├─────────────────────────────────────────────────────────┤
│ Step 4: LoRA 分支（use_lora=True 时）                  │
│   model.lm_head = CastOutputToFloat(model.lm_head)      │
│   peft.LoraConfig(r=lora_rank, lora_alpha=32, ...)     │
│   model = peft.get_peft_model(model, peft_config)       │
├─────────────────────────────────────────────────────────┤
│ Step 5: 构建优化器与学习率调度器                        │
│   optimizer = AdamW (分组 weight_decay)                  │
│   lr_scheduler = Linear Warmup                          │
├─────────────────────────────────────────────────────────┤
│ Step 6: 获取数据                                        │
│   train_dataloader, dev_dataloader = get_data()         │
├─────────────────────────────────────────────────────────┤
│ Step 7: 训练循环 (epochs × steps)                       │
│   ┌─────────────────────────────────────────────────┐   │
│   │ for each batch:                                  │   │
│   │   forward → loss                                 │   │
│   │   loss.backward()                                │   │
│   │   optimizer.step()                               │   │
│   │   lr_scheduler.step()                            │   │
│   │   if step % logging_steps == 0: print log        │   │
│   │   if step % save_freq == 0: eval + save best     │   │
│   └─────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────┤
│ Step 8: 保存最佳模型                                    │
│   evaluate_model() → 比较 best_eval_loss               │
│   save_model() + tokenizer.save_pretrained()            │
└─────────────────────────────────────────────────────────┘
```

**优化器参数组策略**:

| 参数组 | 包含参数 | weight_decay |
|--------|---------|-------------|
| Group 1 | 不含 `bias` 和 `LayerNorm.weight` 的参数 | `pc.weight_decay` (当前=0) |
| Group 2 | `bias` 和 `LayerNorm.weight` 参数 | `0.0`（不衰减） |

**日志输出示例**:
```
global step 200 ( 25.00% ) , epoch: 1, loss: 0.85230, speed: 2.50 step/s, ETA: 01:20:00
```

---

### 4.6 `inference.py` —— 推理脚本

#### 4.6.1 `inference(model, tokenizer, instuction: str, sentence: str) -> str`

**职责**: 使用训练好的模型进行 SPO 三元组抽取推理。

**输入模板构造**:
```
Instruction: {instuction}\n
Input: {sentence}\n
Answer: 
```

**推理流程**:
1. 拼接指令模板文本 → tokenize → 得到 `input_ids` batch
2. 调用 `model.generate(input_ids, max_new_tokens=max_new_tokens)` 自回归生成
3. 解码生成结果 `tokenizer.decode(out[0])`
4. 按 `'Answer: '` 分割，取最后一段作为答案

**使用方式**（`__main__` 示例）:
```python
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

res = inference(model, tokenizer, 
    instuction="现在你是一个非常厉害的SPO抽取器。",
    sentence="下面这句中包含了哪些三元组，用json列表的形式回答..."
)
```

> **注意**: `device` 和 `max_new_tokens` 是模块级变量（在 `__main__` 中定义），`inference()` 函数内部通过闭包访问。在生产环境中需确保调用前正确定义这些变量。

---

## 5. 函数调用关系与依赖图谱

```
train.py: model2train()
│
├─► glm_config.py: ProjectConfig.__init__()
│
├─► transformers.AutoTokenizer.from_pretrained()
├─► transformers.AutoConfig.from_pretrained()
├─► transformers.AutoModel.from_pretrained()
│
├─► utils/common_utils.py
│   ├── CastOutputToFloat(model.lm_head)
│   ├── save_model(model, cur_save_dir)
│   │   └── peft: merged_model.merge_and_unload() / save_pretrained()
│   └── second2time(seconds)
│
├─► peft.LoraConfig(...)
├─► peft.get_peft_model(model, peft_config)
│
├─► data_handle/data_loader.py: get_data()
│   ├── datasets.load_dataset('text', ...)
│   ├── datasets.Dataset.map()
│   │   └─► data_handle/data_preprocess.py: convert_example_chatglm()
│   │       ├── tokenizer.encode() ─── 两次调用（context + target）
│   │       ├── tokenizer.build_inputs_with_special_tokens()
│   │       └── numpy.array()
│   └── torch.utils.data.DataLoader(...)
│
├─► torch.optim.AdamW(...)
├─► transformers.get_scheduler(name='linear', ...)
│
├─► evaluate_model(model, dev_dataloader)
│   └── model.forward(input_ids, labels) → loss
│
└─► model.forward() / loss.backward() / optimizer.step()

─────────────────────────────────────────────

inference.py: inference()
│
├── tokenizer(input_text, return_tensors="pt")
├── model.generate(input_ids, max_new_tokens)
├── tokenizer.decode(out[0])
└── (split by 'Answer: ')

─────────────────────────────────────────────

data_handle/data_preprocess.py: get_max_length()
│
├── json.loads(line)
├── tokenizer.encode(line['context'])
├── tokenizer.encode(line['target'])
└── tqdm.tqdm(f.readlines())
```

### 5.1 依赖关系矩阵

```
                     config   preprocess   loader   common_utils   train   inference
glm_config.py          ●                                            ○        ○ (路径)
data_preprocess.py     ●          ●
data_loader.py         ●          ●          ●
common_utils.py        ●                                            ○
train.py               ●          ○          ●          ●          ●
inference.py           ○                                            ○       ●
```

> `●` = 强依赖（import/调用），`○` = 弱依赖/间接关联

---

## 6. 数据格式说明

### 6.1 JSONL 文件结构

每行为一个完整的 JSON 对象，包含两个字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `context` | `str` | 指令模板 + 输入文本，以 `Answer: ` 结尾 |
| `target` | `str` | 标准答案（JSON 格式的 SPO 三元组列表，含 Markdown 代码块包裹） |

### 6.2 数据样例

```json
{
  "context": "Instruction: 你现在是一个很厉害的阅读理解器，严格按照人类指令进行回答。\nInput: 句子中包含了哪些信息，输出json：\n\n《腾空的日子》是由毕鑫业执导，李佳航、胡冰卿、郑家彬领衔主演。\nAnswer: ",
  "target": "```json\n[{\"predicate\": \"主演\", \"object_type\": \"人物\", \"subject_type\": \"影视作品\", \"object\": \"李佳航\", \"subject\": \"腾空的日子\"}, {\"predicate\": \"导演\", \"object_type\": \"人物\", \"subject_type\": \"影视作品\", \"object\": \"毕鑫业\", \"subject\": \"腾空的日子\"}, {\"predicate\": \"主演\", \"object_type\": \"人物\", \"subject_type\": \"影视作品\", \"object\": \"郑家彬\", \"subject\": \"腾空的日子\"}, {\"predicate\": \"主演\", \"object_type\": \"人物\", \"subject_type\": \"影视作品\", \"object\": \"胡冰卿\", \"subject\": \"腾空的日子\"}]\n```"
}
```

### 6.3 SPO 三元组结构

```json
{
  "subject": "腾空的日子",
  "subject_type": "影视作品",
  "predicate": "导演",
  "object": "毕鑫业",
  "object_type": "人物"
}
```

以 JSON 列表形式组织，每个元素为一个 `{subject, subject_type, predicate, object, object_type}` 五元组。

### 6.4 数据集文件对比

| 文件 | 指令风格 | 样本量（大约） | 用途 |
|------|---------|--------------|------|
| `dataset.jsonl` | "SPO 抽取器" | 少量（约5条） | 示例/测试 |
| `mixed_train_dataset.jsonl` | "阅读理解器" | 较多 | 训练 |
| `mixed_dev_dataset.jsonl` | "阅读理解器" | 较多 | 验证 |

---

## 7. ChatGLM 特殊 Token 机制

### 7.1 关键特殊 Token

| Token | 符号 | ID（示例） | 含义 |
|-------|------|-----------|------|
| `[gMASK]` | 生成掩码 | `130001` | 标记模型生成起始位置 |
| `<sop>` / `[BOS]` | 句子开头 | `130004` | target 起始标记 |
| `<eop>` / `[EOS]` | 句子结尾 | `130005` | target 结束标记 |
| `[PAD]` | 填充 | `130002` | 序列长度对齐填充 |

### 7.2 `build_inputs_with_special_tokens` 拼接逻辑

源码位置：`tokenization_chatglm.py` L323-L344

```python
def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
    gmask_id = self.sp_tokenizer[self.gmask_token]
    eos_id = self.sp_tokenizer[self.eos_token]
    token_ids_0 = token_ids_0 + [gmask_id, self.sp_tokenizer[self.bos_token]]
    if token_ids_1 is not None:
        token_ids_0 = token_ids_0 + token_ids_1 + [eos_id]
    return token_ids_0
```

**最终序列结构**:

```
[source tokens] + [gMASK] + [BOS/<sop>] + [target tokens] + [EOS/<eop>]
    ▲ context部分（labels=-100）         ▲ target部分（labels=真实token ID）
```

### 7.3 labels 构造图解

```
input_ids:  [ T1  T2  ...  Tn-1  Tn  [gMASK]  [BOS]  A1  A2  ...  Am  [EOS]  [PAD]  [PAD] ]
               ▲ context tokens                  ▲ target tokens               ▲ padding
labels:     [-100 -100 ... -100 -100  -100     A1     A1  A2  ...  Am  [EOS]  -100   -100 ]
               ▲ 不计算loss                      ▲ 计算loss                    ▲ 不计算loss
```

---

## 8. 配置运行说明

### 8.1 当前运行模式

| 项目 | 配置 |
|------|------|
| 微调方法 | **LoRA**（r=8, alpha=32, dropout=0.1） |
| P-Tuning | **关闭** (`use_ptuning=False`) |
| 精度 | **FP16**（半精度） + autocast 混合精度 |
| 批大小 | **4** |
| 训练轮数 | **2** |
| 学习率 | **3e-5**，Linear Warmup (6%) |
| 最大序列长度 | context=100, target=100 |
| 评估/保存频率 | 每 200 步 |
| 日志频率 | 每 10 步 |

### 8.2 切换微调模式

**只用 LoRA**（当前）:
```python
self.use_lora = True
self.use_ptuning = False
```

**只用 P-Tuning**:
```python
self.use_lora = False
self.use_ptuning = True
self.pre_seq_len = 200
self.prefix_projection = False   # P-Tuning v1
# self.prefix_projection = True  # P-Tuning v2
```

> **注意**: 两种模式互斥，不建议同时启用。

### 8.3 运行方式

**训练**:
```bash
python train.py
```

**推理**:
```bash
python inference.py
```

### 8.4 路径配置注意事项

- `glm_config.py` 中的 `pre_model`、`train_path`、`dev_path`、`save_dir` 均为硬编码绝对路径。
- 在不同环境下运行时需要修改 `glm_config.py` 中的路径为本地实际路径。
- 当前 `save_dir` 为 `/gemini/checkpoints/ptune`（云端路径），本地运行前必须修改。

---

## 9. 潜在问题与注意事项

| 序号 | 问题 | 影响 | 建议 |
|------|------|------|------|
| 1 | 路径硬编码 | 跨环境无法直接运行 | 使用相对路径或环境变量 |
| 2 | `inference.py` 中 `device` 变量通过闭包传递 | 模块级调用依赖 `__main__` 上下文 | 将其作为函数参数传入 |
| 3 | `train.py` L7 导入 `from transformers import AdamW` 冗余 | 实际使用 `torch.optim.AdamW` | 删除或注释冗余导入 |
| 4 | `get_max_length()` 使用 `readlines()` 全量读取 | 大文件内存溢出 | 改为逐行流式读取 |
| 5 | 最佳模型保存覆盖 `model_best` 目录 | 无历史版本轮转 | 按 global_step 带版本号命名 |
| 6 | LoRA alpha 硬编码为 32 | 修改需改源码 | 提取为配置项 |
| 7 | 项目无 `requirements.txt` | 依赖不可追溯 | 建议创建依赖声明文件 |

---

## 10. 依赖包清单（建议写入 requirements.txt）

```
torch>=1.13.0
transformers>=4.27.0
peft>=0.3.0
datasets>=2.0.0
numpy>=1.21.0
tqdm>=4.64.0
rich>=13.0.0
```
