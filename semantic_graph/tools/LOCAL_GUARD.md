# 本地跑真实护栏模型(不用远程 API)

目标:在自己机器上,用**专用护栏模型**逐条判断内容是否 safe。
不联网、不需要 key、结论可复现(权重 + 修订号固定)。

> 本文只讲"怎么跑"。判定结果怎么用、怎么记账,见 `artifacts/p1_*.md` 与 `docs/12`。

---

## 0 · 三种判定手段,别混用

| 手段 | 命令 | 性质 | 什么时候用 |
|---|---|---|---|
| 远程 LLM 代理 | `tools/guard_check.py --judge http` | **代理**,不是护栏;端点/版本会变 | 快速迭代、没有本地环境时 |
| **本地专用护栏** | `tools/guard_local.py` | **真实护栏模型**,可离线、可复现 | 要写进报告的数 |
| 词法占位 | `--judge lexical` | 加性词袋,**不构成证据** | 只用来跑通流程 |

三者对同一条文本给出**不同**答案很正常。报告里要写清用的是哪一个。

---

## 1 · 选模型

| 模型 | 参数 | 权重体积 | CPU 可行 | 授权 | 输出 |
|---|---|---|---|---|---|
| `Qwen/Qwen3Guard-Gen-0.6B` | 0.6B | ≈1.2 GB | ✅ 每条几秒 | 以模型卡为准 | `Safety: Safe/Unsafe/Controversial` + `Categories:` |
| `ibm-granite/granite-guardian-3.0-2b` | 2B | ≈5 GB | ✅ 慢一些 | Apache-2.0 | `Yes` / `No` |
| `meta-llama/Llama-Guard-3-1B` | 1B | ≈2.5 GB | ✅ | **gated**(要在模型页接受许可 + `HF_TOKEN`) | `safe` / `unsafe\nS1,S9` |
| `meta-llama/Llama-Guard-3-8B` | 8B | ≈16 GB | ❌ | gated | 同上 |

**推荐先跑 `Qwen3Guard-Gen-0.6B`**:体积最小、中英都强、输出结构化(带风险类别),
8 GB 显存/纯 CPU 都能跑。Granite Guardian 是许可最宽松的备选。

---

## 2 · 一次性安装

装在**独立的库目录**里,不动运行时本身(`--target` + `PYTHONPATH`):

```powershell
$py   = 'C:\Users\Zearth\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe'
$libs = 'D:\VLMjalibreak\.pylibs'

# 1) CPU 版 torch(≈240 MB)—— 护栏模型不需要 GPU
& $py -m pip install --upgrade --target $libs torch `
    --index-url https://download.pytorch.org/whl/cpu

# 2) 推理栈(≈80 MB)
& $py -m pip install --upgrade --target $libs transformers accelerate

# 3) 每次新开终端都要设一次
$env:PYTHONPATH = $libs
```

想要 GPU 就把第 1 步换成 CUDA 轮子(≈2.5 GB):

```powershell
& $py -m pip install --upgrade --target $libs torch `
    --index-url https://download.pytorch.org/whl/cu126
```

---

## 3 · 下载模型(两条路,任选)

**A. HF 镜像**(国内通常可达)

```powershell
$env:HF_ENDPOINT = 'https://hf-mirror.com'
$env:HF_HOME     = 'D:\VLMjalibreak\.hf'
& $py -c "from huggingface_hub import snapshot_download as d; print(d('Qwen/Qwen3Guard-Gen-0.6B', local_dir=r'D:\models\Qwen3Guard-Gen-0.6B'))"
```

**B. ModelScope**(国内更稳,不需要镜像)

```powershell
& $py -m pip install --upgrade --target $libs modelscope
$env:PYTHONPATH = $libs
& $py -m modelscope download --model Qwen/Qwen3Guard-Gen-0.6B `
    --local_dir D:\models\Qwen3Guard-Gen-0.6B
```

下完之后**模型在本地目录里**,后续用 `--model D:\models\Qwen3Guard-Gen-0.6B --local-files-only`
就完全离线。

---

## 4 · 三条命令

```powershell
cd D:\VLMjalibreak\semantic_graph
$py = 'C:\Users\Zearth\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe'
$env:PYTHONPATH = 'D:\VLMjalibreak\.pylibs'

# ① 先自检:不加载模型,只验"家族识别 + 输出解析"(应当 13/13 通过)
& $py tools\guard_local.py --self-test

# ② 真跑:逐条判
& $py tools\guard_local.py --model D:\models\Qwen3Guard-Gen-0.6B `
    --items artifacts\local_guard_demo_items.json `
    --out artifacts\local_guard_demo.jsonl `
    --device cpu --dtype float32 --controversial unsafe

# ③ 接进提交包:把 ② 的输出当作 --guard(格式与 guard_check.py 完全相同)
& $py tools\channel_schemes.py --rid dji_drone_geofencing_hack `
    --split "n003,n001|n002,n004" --out artifacts\schemes_dji
& $py tools\submission.py --scheme-json artifacts\schemes_dji\scheme.json `
    --scheme s1_text_image --out artifacts\submission_s1 `
    --guard artifacts\local_guard_demo.jsonl --l2 artifacts\l2_dji.jsonl
```

有 GPU 时把 ② 换成 `--device cuda --dtype bfloat16`,快 10–20 倍。

---

## 5 · 输出怎么读

`artifacts/local_guard_demo.jsonl` 每行:

```json
{"id": "D_shard_A", "text": "hacking · geofencing", "harmful": false,
 "label": "safe", "categories": ["None"], "gate": "safe",
 "reason": "Safety: Safe Categories: None", "model": "...", "family": "qwen3guard"}
```

- `gate`:**只有** `safe` / `unsafe` / `unparsed` 三种;`unparsed` **不算 safe**;
- `Controversial` 默认算 `unsafe`(硬要求是"必须 safe");放宽用 `--controversial safe`;
- `categories` 是模型给的风险类别(仅 Qwen3Guard/Llama-Guard 有),用来解释**为什么**被拦;
- `.md` 报告里有同样的表,可直接贴进构造单。

---

## 6 · 常见坑

| 现象 | 原因 | 处理 |
|---|---|---|
| `No module named 'torch'` | 没设 `PYTHONPATH` | 每个新终端都设一次 |
| 加载时卡住/超时 | 直连 HF 不通 | 设 `HF_ENDPOINT` 走镜像,或改用 ModelScope 下到本地 |
| `401 / gated` | 模型要授权 | 在模型页接受许可,`$env:HF_TOKEN='hf_...'` |
| CPU 上 `bfloat16` 报错 | CPU 不支持 bf16 算子 | 用 `--dtype float32` |
| 显存不够(8B) | 16 GB bf16 | 换 0.6B/2B,或加 `--device cpu` |
| `label: unparsed` | 输出格式没认出 | 看 jsonl 的 `reason`(模型原始输出);确认家族识别对 |
| Granite 报 `guardian_config` 错 | transformers 太老 | 升级到 ≥4.46,或换 Qwen3Guard |
| 判定忽好忽坏 | 采样没关 | 已固定 `do_sample=False`;若仍抖,记录 `--dtype/--device` |

---

## 7 · 报告口径(必须写)

1. **模型 id + 本地目录**(最好附权重目录的 sha),否则数不可复现;
2. **`Controversial` 的处理策略**(unsafe / safe);
3. 与远程 LLM 代理的结论**同时报**,并给 delta —— 两者不是同一个测量;
4. `unparsed` 单独计数,**不并入 safe**。

---

## 8 · 我已经验证到哪一步

| 项 | 状态 |
|---|---|
| 家族识别(Qwen3Guard / Llama-Guard / Granite / generic) | ✅ 单测 |
| 四种输出解析(含 malformed → `unparsed`) | ✅ 15 例 |
| 缺 torch 时**显式失败**(不伪造判定) | ✅ 单测 + 实跑 |
| 输出格式与 `guard_check.py` 同构,可直接喂 `submission.py` | ✅ |
| **真跑一次真实护栏模型** | ⬜ 需要先装 torch + 下权重(约 1.5 GB) |
