# Muse Spark 在 910B 上的 NPU 修复记录

记录 `muse_spark_debugmodel` 及相关清单配方在容器里的失败原因、速查对照与修复细节。本仓代码 / 补丁问题进对照表；ascend 覆盖进 A 表；出网与启动入口等非本仓问题进 E 表。本分支只关心**跑通**，不收录精度 / 线性度 / 内存 / 确定性验收改动。

分支：`npu-muse-spark-r04-code-only`。

**编号约定**：对照表里的 `#N` 与下文「详情」里的 `## N.` 一一对应；A / E 表各自用 A1、E1 编号，不占用 1–13。

## 环境（A2）

| 项 | 值 |
| --- | --- |
| 环境代号 | **A2** |
| 机器 / 容器 | `179.173.2.3` / `npu-y00839695-torchtitan-20261028`（多模态也曾在 `179.173.2.2`） |
| 卡 | 8 × Ascend **910B4** |
| 镜像 | CANN 9.2.0 910B Ubuntu22.04 py3.12 |
| torch / torch_npu | **须配套**：`2.15.0+cpu` / `2.15.0.dev20261005+git14dc54e`（旧 `…git95c7e46` 会踩 flex 导入 / SympifyError，换包即可，不单列问题） |
| 训练入口 | `ENTRYPOINT=torchtitan_ascend.train`（勿用裸 `torchtitan.train`） |
| 必打补丁（本地 910B） | 本机 `docs/npu_910b_patches/`（**不进 PR**）；或 ascend `fixes/torch_npu/` + Inductor/remat |
| 编译线程 | `TORCHINDUCTOR_COMPILE_THREADS=1`，`TORCHNPU_PRECOMPILE_THREADS=1` |

换机（910B 本地）：检出本分支 → 同版本 wheel → 本机 apply `docs/npu_910b_patches/`（gitignore，不进 #5772）或 ascend `fixes/torch_npu/` → ascend 入口。A5 验收勿打 910B flex 补丁。

---

## 由 torchtitan-ascend 覆盖（本仓不维护）

下列项**不要**再在 torchtitan 里打手工补丁；装好 ascend 并用 `ENTRYPOINT=torchtitan_ascend.train` 即可。

| # | 现象 / 失败 | 根因（一句话） | 落点 |
| --- | --- | --- | --- |
| A1 | remat 反向段错误（空指针 / null storage 占位） | NPU 上不能用空指针 storage 做占位张量 | 运行时 `torchtitan_ascend/patches/storage.py`；安装期 `fixes/torch_npu/storage.patch`。简述见 [A1](#a1-反向重算里的空指针占位张量) |
| A2 | 编译撞 `make_reduction(..., strict_reduction=...)` | Inductor API 与已装 `torch_npu` 不一致 | 运行时 `torchtitan_ascend/patches/inductor.py` |
| A3 | `NPUCombinedScheduling` 相关 Inductor 调度/接口缺口 | 同上，调度侧与上游 API 不对齐 | 运行时 `torchtitan_ascend/patches/inductor.py` |

本仓曾有对应手工改法，已在 `b3135c09` 删掉，改走 ascend。

---

## 环境 / 启动方式（非本仓代码问题）

机房出网、训练入口选错等，**不是** torchtitan 仓库里要合入的修复。E1 的绕过代码只在 `npu-muse-spark-debug-r04`；E2 无本仓改动，装好 `torchtitan-ascend` 后改入口即可。

| # | 现象 / 失败 | 根因（一句话） | 落点 | 详情 |
| --- | --- | --- | --- | --- |
| E1 | UltraData 拉 huggingface 被重置 | 机房出网受限，`huggingface.co:443` 连接重置 | 本地 JSONL：`f99ea981`（仅 `npu-muse-spark-debug-r04`） | [E1](#e1-第-5-行改读本地-ultradata-jsonl) |
| E2 | 裸 `torchtitan.train` 段错误 / 再撞 make_reduction | 未走 ascend 入口，A1–A3 补丁未加载 | `ENTRYPOINT=torchtitan_ascend.train` | 见上文 A 表 |

---

## 问题 → 改动对照表（本仓 / 补丁目录仍保留）

表中「flex/非950」列：`Y` 表示属于 **flex attention 在非 Ascend 950（含 910B4）上的适配**；汇总见 [flex 非 950 适配点](#flex-attention-非-ascend-950-适配点)。

| # | flex/非950 | 现象 / 失败 | 根因（一句话） | 改动落点 | 详情 |
| --- | --- | --- | --- | --- | --- |
| 1 | | RoPE / 视觉 RoPE `aclnnIndex` complex64 | 910B index 无 complex64 | `rope.py` / `vision_encoder.py` NPU `view_as_real` | [§1](#1-rope--视觉-rope-的-complex64-下标) |
| 2 | Y | packed-document mask：`Buffers cannot be created` / mask-out 编不过 | 间接 index 走 fallback 开临时 buffer；910B4 mask-out 核编挂 | `flex_attention.py.patch`：逐点 index + 非 950 关 mask-out | [§2](#2-packed-document-mask-的间接-index-与-mask-out) |
| 3 | | SwiGLU / Helion CUDA 路径在 NPU 挂 | 融合 override 绑 CUDA | 非 CUDA `silu(gate)*up`；ascend `with_cp_npu` 清空 override | [§3](#3-融合-swiglu-没有-npu-实现) |
| 4 | Y | flex 反向选块段错误（大 tile / 多 rank 并行 autotune） | 910B4 大分块 SIGSEGV；多进程同时选块竞态，一人崩则整组被 SIGTERM | 反向默认 tile=16；flex autotune 文件锁 | [§4](#4-flex-attention-反向选块与多-rank-锁) |
| 5 | Y | 短视觉序列 → HCCL watchdog | flex decoding 另编核超时 | 非 950 关 decoding；配方 `train_timeout=1800` | [§5](#5-短视觉序列的-flex-decoding-和-训练超时) |
| 6 | Y | 编译进程池 + flex 段错误 | 多线程编译不安全 | `COMPILE_THREADS=1` / `PRECOMPILE_THREADS=1` | [§6](#6-编译线程限制为-1) |
| 7 | | loss `_assert_async` 编译卡住 | 0 维 device assert 代码生成挂 | 编译期跳过 assert | [§7](#7-词表断言在编译期跳过) |
| 8 | | FSDP 丢掉视觉输出 all-gather | 单帧 `cat` 是 view（FSDP 通用坑，非仅 910B） | `contiguous()` | [§8](#8-视觉-encoder-输出-contiguous) |
| 9 | | 全局 `init_timeout=1800` 过猛 | 启动与训练超时混用 | init 保持 300；1800 只写配方 | [§9](#9-init-timeout-保持-300) |
| 10 | | Dist-MoE 要 SM100 | CuTe 包 / 设备检查 | ascend：`patches/dist_moe/`（Titan 公共零 NPU 改动） | [§10](#10-dist-moe-bf16-在-910b-上的专家计算) |
| 11 | | （已撤销）旧 torch 缺 PP liveness / `param_dtype_override_fn` | 仅 2.14 stable 现场需要 | **A2 的 2.15 已具备，兼容代码已删** | [§11](#11-旧-torch-上-dist-moe--fsdp-的兼容导入) |
| 12 | Y | CP block mask → `simt_only` 失败 | 非 950 不支持 | heuristics 补丁 | [§12](#12-上下文并行的-block-mask-求和被编成-simt_only) |
| 13 | | 第 4 行默认 100s 超时 | 视觉变长重编译 | `train_timeout_seconds=1800` | [§13](#13-第-4-行多模态配方的训练超时) |
| 14 | | remat 重算与 local compile 图不一致 | compile region 落在可重算区，NPU 特化图对不上 checkpoint | 配方收窄 `local_compile_regions` | [§14](#14-收窄-local_compile_regionscompile-与-remat) |

### 未做 / 不可做

| 项 | 说明 |
| --- | --- |
| Dist-MoE **mxfp8** | 第 3 行仅 bf16 / mxfp8 两个配方；mxfp8 配置期要官方 `dist_moe`/`torchao`，未改未跑。工作项里的「压缩 MoE 兼容」挂在 mxfp8 这条上，一并未验 |
| 精度 / 线性度 / 内存 / 确定性验收 | 本分支不做；不为四项指标加 perf/mem 配方或对照改动 |
| flex 与未改分块数值对拍 | 分块 16 只为稳定 |
| FSDP view 警告全量追根 | 仅改了视觉 encoder 返回 |

### 相关文档

| 文档 | 用途 |
| --- | --- |
| `docs/npu_910b_patches/`（本地） | 910B flex/heuristics；**不进 PR #5772**；合入侧用 ascend `fixes/torch_npu/` |
| `docs/npu_debugmodel_verification.md` | 按清单行跑通记录（非四项验收） |

### 本分支（`npu-muse-spark-r04-code-only`）代码落点

上表 **#1 / #3 / #7–#9** 等本仓 Python 改动可进本 PR；**#2 / #4 / #5(decoding) / #6 / #12** 的 flex/`env` 补丁目录 **只留本地**（gitignore），合入请走 `torchtitan-ascend/fixes/torch_npu/`。**#10 / #13 / #14** 在 ascend。

| # | 在本分支？ | 落点形态（检出后还要做什么） |
| --- | --- | --- |
| 1 | 是 | 本仓：`rope.py` / `vision_encoder.py` |
| 2 | 本地 910B | flex 补丁：本机 `docs/npu_910b_patches/` 或 ascend `fixes/torch_npu/`（**不进本 PR**） |
| 3 | 是 | 本仓：`torchtitan_recipes/overrides/fused_swiglu.py` |
| 4 | 本地 910B | 同上 flex + heuristics（**不进本 PR**） |
| 5 | 本地 910B | 补丁关 decoding（本地）；配方 timeout 仅 910b_local helper |
| 6 | 本地 910B | `env_npu_910b.sh`（本地 / ascend `fixes/torch_npu/`；**不进本 PR**） |
| 7 | 是 | 本仓：`torchtitan/components/loss.py` |
| 8 | 是 | 本仓：`vision_encoder.py` 的 `contiguous()` |
| 9 | 是 | 本仓：`CommConfig.init_timeout_seconds` 默认 300 |
| 10 | 否（迁至 ascend） | `torchtitan-ascend`：`torchtitan_ascend/patches/dist_moe/`（仅 `ENTRYPOINT=torchtitan_ascend.train`） |
| 11 | 否（已删） | A2 `torch 2.15.0+cpu` 已有相关 API，不再保留兼容探测 |
| 12 | 本地 910B | heuristics 补丁（**不进本 PR**） |
| 13 | 本地 910B 测试 | 不进提交默认路径；本机用 gitignored `recipes/muse_spark_910b_local.py` 设 `train_timeout=1800` |
| 14 | 本地 910B 测试 | 同上：本地 helper 收窄 `local_compile_regions`；A5 验收用干净 `recipes/muse_spark.py` |

**#2 / #4 / #5（decoding）/ #6 / #12**：不进 PR #5772。910B 本机用 gitignored `docs/npu_910b_patches/`，或 ascend `fixes/torch_npu/`。  
**不在本 PR**：E1 本地 UltraData JSONL；A1–A3；`docs/npu_910b_patches/` 目录本身。

---

## 可复用补丁（本地 910B / ascend，不进本 PR）

910B4 上仍要单独打的 **flex / heuristics** site-packages 改动：

- **本机（gitignore）**：`docs/npu_910b_patches/` → `apply_npu_910b_patches.sh` / `env_npu_910b.sh`
- **合入侧**：`torchtitan-ascend/fixes/torch_npu/`

不要再手工改 `lowering.py` / `npu_combined_scheduling.py` / `torch_remat/_placeholder.py`。

### flex attention 非 Ascend 950 适配点

下列改动只针对 **非 950**（本机为 910B4）；Ascend 950 仍走上游默认的 mask-out / decoding / 自动选块等路径。补丁落点见上；A5 勿打 910B flex 补丁。

| 适配点 | 非 950 行为 | 代码/补丁落点 | 详情 |
| --- | --- | --- | --- |
| mask 子图间接 `aten.index` | Triton 且间接访存 codegen 关时，改用上游逐点 index（与 DVM 侧已有修法对齐） | `flex_attention.py.patch` | [§2](#2-packed-document-mask-的间接-index-与-mask-out) |
| 关闭 mask-out | `_supports_flex_attention_mask_out()` → `false`，强制 mask-in | 同上 | [§2](#2-packed-document-mask-的间接-index-与-mask-out) |
| 反向默认 tile=16 | `BLOCK_M1/N1/M2/N2` 默认 16，避免大块 SIGSEGV | 同上 | [§4](#4-flex-attention-反向选块与多-rank-锁) |
| flex autotune 串行化 | 多 rank 对 flex 反向选块加文件锁，避免并行 autotune 竞态 | `triton_heuristics.py.patch` | [§4](#4-flex-attention-反向选块与多-rank-锁) |
| 关闭 flex decoding | `_use_flex_decoding()` → `false`，短 query 仍走 mask-in 主核 | `flex_attention.py.patch` | [§5](#5-短视觉序列的-flex-decoding-和-训练超时) |
| 编译线程=1 | `TORCHINDUCTOR_COMPILE_THREADS=1`，`TORCHNPU_PRECOMPILE_THREADS=1` | `env_npu_910b.sh` | [§6](#6-编译线程限制为-1) |
| 禁止 `simt_only` | `simd_simt_mix` 不再生成仅 950 支持的 `simt_only`（CP block mask 求和会踩） | `triton_heuristics.py.patch` | [§12](#12-上下文并行的-block-mask-求和被编成-simt_only) |
| 训练超时放宽（配方） | 视觉变长重编译时避免 HCCL 100s watchdog；与 decoding 关闭配合 | 仅 910B 本地 helper；A5 提交配方不设 | [§5](#5-短视觉序列的-flex-decoding-和-训练超时)、[§13](#13-第-4-行多模态配方的训练超时) |

---

## 1. RoPE / 视觉 RoPE 的 complex64 下标

文本路径 `09:59:14` 已走过 attention mask 编译，停在第一层 RoPE：

```text
rope.py: rope_cache = rope_cache[positions]
RuntimeError: aclnnIndex failed, error code 161002
Tensor self not implemented for DT_COMPLEX64
```

`ComplexRoPE` 的频率表是 `complex64`。NPU 的 index 只支持 float、bfloat16、整型和 bool，不支持 complex64。

`torchtitan/models/common/rope.py` 的 `_reshape_for_broadcast` 在 NPU 上改为先 `view_as_real`，用 float 下标取出，再 `view_as_complex` 变回去。CUDA 仍走原来的复数下标。

8 卡多模态还会在视觉编码器里再踩一次（不经过上面的 `_reshape_for_broadcast`）：

```text
RuntimeError: index_high_dims_op_api aclnnIndex 161002
DT_COMPLEX64
vision_encoder.py: freqs_cis_GD[sp_perm]
```

`torchtitan/models/muse_spark/vision_encoder.py` 在 NPU 上同样先 `view_as_real`，按下标取实部虚部，再 `view_as_complex`。其它设备仍走原来的下标。

## 2. Packed-document mask 的间接 index 与 mask-out

`10:02:16` 训练停在 `flex_attention` 编译 mask 子图：

```text
SubgraphLoweringException: Buffers cannot be created while lowering a pointwise subgraph
While executing aten.index.Tensor
attention.py: return kv_idx >= offsets[document_id[q_idx]]
```

### 在算什么

多篇文档拼成一条长序列时，mask 要回答：这个 query 能不能看那个 key。做法是查表，而不是写死公式，例如：

1. 用 query 位置查出它属于哪篇文档：`document_id[q_idx]`
2. 再用文档号查出这篇文档从哪开始：`offsets[document_id[q_idx]]`
3. 比较 key 位置是否在允许范围内

第 2 步是「用第一次查表的结果，再去查第二次」——间接下标（gather）。

flex 会把整段 mask 编成一块**逐点小图**：每个 `(query, key)` 当场算出 true/false。规则很严：**小图里不能再临时开一块中间结果缓冲区**。

### 问题 1：间接下标被编成「先写出中间结果」

910B 默认 Triton 路径通常关着「间接访存专用代码生成」（`inductor_indirect_memory_mode`）。于是 `aten.index`（下标取值）不会融进小图，而被登记成 AclNN 兜底：

- 兜底做法：先申请一块临时内存，把下标取值的结果整块写进去，后面再读
- 小图规则：不允许在编译这段 mask 时再开临时缓冲区

两边打架，就报上面的 `Buffers cannot be created...`。

DVM（`torch_npu` 的另一种 Inductor 后端，与默认 Triton 并列）已经改过：在 flex 的 mask 小图里，把 `aten.index` 改回上游那种「在小图里直接读」，不走会开临时内存的兜底。默认 Triton + 910B 原先没有同样处理。

**改法**：backend 是 DVM，或间接访存代码生成没开时，给 flex mask 小图同样挂上上游的逐点 `aten.index`。补丁在 `docs/npu_910b_patches/flex_attention.py.patch`。

### 问题 2：下标过了，mask-out 核在 910B4 仍编不过

修好间接下标后，缩小的 flex 能走过 mask 小图。但 910B4 上还有第二种写法 **mask-out**（单独一套 mask 核外挂）编不过：片上 UB 不够，或 `bishengir-compile` 直接崩。同一套 mask 含义改用 **mask-in**（mask 写进 attention 主路径）可以编过、跑通。

**改法**：非 Ascend 950 上 `_supports_flex_attention_mask_out()` 返回 `false`，训练只用 mask-in；950 仍可用 mask-out。同属上面的 flex 补丁。

### 边界

不改模型里 mask 的数学定义；不改 CUDA，也不改 Ascend 950 的 mask-out。顺序是：先让间接下标能合法编进小图，再在 910B4 上避开编挂的 mask-out。

## 3. 融合 SwiGLU 没有 NPU 实现

`10:14:12` 的训练停在 MoE 专家的激活：

```text
moe.py: hidden_RF = self.activation_fn(gate_up_R2F, offsets=offsets_E)
fused_swiglu.py: return silu_and_mul_op(gate_up, offsets)
NotImplementedError: Could not run 'torchtitan::silu_and_mul' with arguments from the 'CPU' backend
```

`torchtitan::silu_and_mul` 只登记了 CUDA。张量在 NPU 上时，PrivateUse1 的 fallback 会改去调 CPU 实现，而 CPU 也没有这个算子。

`FusedSwiGLU.__call__` 在非 CUDA 设备上改为 `silu(gate) * up`。有效行的结果和 Triton 核一致，反向走普通 autograd。CUDA 仍走原来的融合核。

## 4. flex attention 反向选块与多 rank 锁

### 在干什么

flex attention **反向**核要把计算切成小块（tile），块大小由 `BLOCK_M1`、`BLOCK_N1`、`BLOCK_M2`、`BLOCK_N2` 等参数决定。Inductor/Triton 常会 **autotune**：试很多组块大小，按耗时挑一组。910B4 上这套自动选块不稳定。

### 问题 1：块太大，本 rank 直接段错误

四卡训练走到 flex 反向后，某个 rank（例如 rank 3）以 **SIGSEGV（exit -11）** 退出，没有 Python 异常。对照发现：自动选块挑中 **大于 16** 的 `BLOCK_M1/BLOCK_N1` 时会崩；同一段 mask 把反向分块 **固定成 16** 时，前向和反向都能跑完，梯度有限、非 0。

**改法**：非 Ascend 950 上，mask-in 反向给上述 `BLOCK_*` **默认设为 16**（调用方已经指定则不覆盖）。见 `docs/npu_910b_patches/flex_attention.py.patch`。

### 问题 2：多卡同时选块，一个崩、整组被收掉

8 卡时还有第二种挂法：多个 rank **各自**在跑 `triton_per_fused_flex_attention_backward_0` 的几十组候选。某个 rank 先段错误退出；其余 rank 往往并不是自己算错，而是同伴死后被 `torchrun`/进程组发 **SIGTERM 连带杀掉**（分布式一人挂、整组收工）。同一形状在 **单卡** 上前向反向常能算出有限梯度；多卡时崩在哪个 rank 不固定，更像多进程并行 autotune 的竞态。

曾试过选块时直接拿最小 runtime tile；launcher 与 runtime block 对不齐，已撤掉。

**改法**：在 `triton_heuristics.py` 里，名字带 `flex_attention` 的 autotune 用 `fcntl.flock` 锁 `/tmp/npu_flex_attention_autotune.lock`，**同一时刻只让一个进程试配置**。同进程重入用深度计数，避免自己锁死自己。候选仍按原来的计时挑选，不改成「永远用最小块」。见 `docs/npu_910b_patches/triton_heuristics.py.patch`。

### 边界

不改 flex 反向的数学；CUDA / Ascend 950 不强制默认 16。§6 的「编译线程=1」管的是编译进程池，和本节的「选块加锁」是两件事。

## 5. 短视觉序列的 flex decoding 和训练超时

第 1 到第 4 步能打完之后，第 5 步视觉 token 数变成 4。inductor 改走 flex decoding（query 长度小于 128）。rank 0 还在编译，另一个 rank 已经发出 all-gather。第 1 步之后进程组超时若仍是默认 `train_timeout_seconds=100`，大约 100 秒后：

```text
Watchdog caught collective operation timeout:
WorkHCCL(OpType=ALLGATHER, NumelIn=15856, NumelOut=31712, Timeout(ms)=100000)
```

HCCL watchdog 把还在集合通信里的 rank 打成 SIGABRT（exit -6）。`NumelOut` 是 `NumelIn` 的 2 倍，对应 TP=2 的 all-gather。

两处一起改：

- `flex_attention.py` 的 `_use_flex_decoding` 在非 Ascend 950 上直接返回 false。短序列继续用已经能跑的 mask-in flex attention，不再编一套 decoding 核。
- 相关 8 卡配方把 `comm.train_timeout_seconds` 设为 1800。新的视觉长度仍会重新编译，第 5 步和第 8 步各花了大约一分钟，100 秒盖不住，1800 秒盖得住。

## 6. 编译线程限制为 1

### 在干什么

`torch.compile` / Inductor 编核时，可以开一个**进程池或线程池**，并行编译、并行试很多组 kernel 配置。默认会开很多工人（几十上百），为的是编得快。

这里有两路旋钮，名字很像，但是两套东西：

| 环境变量 | 谁在用 | 干什么 |
| --- | --- | --- |
| `TORCHINDUCTOR_COMPILE_THREADS` | PyTorch Inductor | 编译子进程池大小 |
| `TORCHNPU_PRECOMPILE_THREADS` | `torch_npu` | NPU 侧预编译 / 选配置时的并行度（默认往往 ≥32） |

### 问题：为什么说「段错误」

这里的崩**不是** Python 抛了个 `Exception`，而是底层原生代码（编译器 / 驱动 / 子进程里的 C/C++）访问非法内存，进程被操作系统干掉：

- 常见表现：`SIGSEGV`、exit `-11`，或日志 `SubprocPool unclean exit`（编译**子进程**先死了，主进程只看到池子不干净退出）
- Python 栈往往没有有用的业务报错，所以不像「哪里写错公式」那么好读

§4 把反向 tile 钉成 16 之后，前几步能跑了，但后面某个 step 在**并行编译/并行试配置**时仍可能这样死。更重的一例：`muse_spark_debugmodel_fsdp2_cp2_ep4` 在 flex 反向选块时，`TORCHNPU_PRECOMPILE_THREADS` 开到约 128 个工人去测几十组配置，rank 0 段错误。

### 和「公式错了」的差别

我们**没有**追到「某一行 NPU 编译器源码必现 bug」的精确定位。现有结论是对照实验：

| 做法 | 结果 |
| --- | --- |
| 编译/预编译工人很多（默认或上百） | 编 flex 相关核时易段错误 / SubprocPool 异常退出 |
| 两套线程都设为 1，串行编 | 同样配方能跑完 |

也就是说：崩发生在**并行编译这条路径**上；改成串行就避开。可能原因包括（推断，非已证明）：多线程同时调同一套编译器/设备库线程不安全、多个子进程同时占编译资源互踩等。本节不宣称已定位到其中哪一种。

和 §4「多 rank 文件锁」不同：§4 管的是 **多个训练进程（rank）** 不要同时选块；本节管的是 **单个 rank 内部** 不要用一大坨编译工人并行干活。

### 改法

`docs/npu_910b_patches/env_npu_910b.sh`（以及容器 `env.sh`）里两套都设成 **1**：

```bash
export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHNPU_PRECOMPILE_THREADS=1
```

效果：编译和预编译改在**当前训练进程里串行做**，不再拉一堆子进程/线程并行试。编得慢一些，但能避开上述并行编译路径上的崩溃，`muse_spark_debugmodel` 可跑完 10 步。

### 边界

这是环境变量级的**规避**，不是改 flex 数学，也不是已向上游提交的编译器根治补丁。训练前需要 `source docs/npu_910b_patches/env_npu_910b.sh`（或等价写入容器 env）。

## 7. 词表断言在编译期跳过

### 在干什么

算 loss 前，代码会检查 labels 是否合法（是 `IGNORE_INDEX`，或落在词表范围内）。检查用的是 `torch._assert_async(...)`：在设备上异步断言，失败时再报错。

### 问题

这段 loss 会进 `torch.compile`。910B 的 Inductor **编不出**这种 0 维 device 上的 `_assert_async`，编译阶段会卡住/挂住，训练走不下去。不是 labels 真的错了，是**断言这个算子本身在 910B 上编不过**。

### 改法

`torchtitan/components/loss.py` 里包一层：

- **正在 compile**（`torch.compiler.is_compiling()` 为真）→ 跳过 `_assert_async`
- **普通 eager 跑** → 照旧做断言

这样编译能过；平时不 compile 时检查还在。

### 边界

只跳过「编译期」的断言，不关掉 eager 检查；不改 loss 计算公式。

## 8. 视觉 encoder 输出 contiguous

### 在干什么

**FSDP**（Fully Sharded Data Parallel）把参数切到各卡，用时 **all-gather** 拼起来算，算完再切回。反向前靠挂在模块输出上的 **pre-backward hook** 触发参数 all-gather。

`NaViTEncoder` 前向最后把各帧结果 `torch.cat` 拼成一块再往下传。若这个输出只是别的张量的 **view**（共享存储），后面再对它做就地（in-place）改写时，autograd 可能把 FSDP 的 hook 「改写丢了」，于是 **pre-backward all-gather 被跳过**，数值/通信会错。

### 问题

只有 **一帧** 时，`torch.cat([那一帧])` 往往直接返回原张量本身，在 FSDP 眼里就是 view。

### 是不是只有 910B / 上游不知道？

**不是 910B 专有 bug。** 这是 FSDP（含 FSDP2）与「forward 输出为 view + 后续 in-place」组合下的**已知风险**：上游有相关 issue，也加过「grad 所需的 forward 输出是 view」一类 **UserWarning**；框架侧是否一律自动 clone 仍可能在跟进。CUDA 上同样结构可以踩。

本仓库是在 910B 多模态路径上撞到 **这个调用点**（单帧 `cat`），用 `contiguous()` 做模型侧规避。verification 里还出现过其它模块（如 Transformer block）返回 view 的 FSDP 警告，当时未全量追完（见文首「未做」表）。

### 改法

返回改成：

```python
return torch.cat(outputs).contiguous()
```

`contiguous()` 保证得到一块连续、独立的存储，避免把该输出当成危险的 view。

### 边界

这次只改了视觉 encoder 这一处。音频编码器、文本 block 上若有同类警告，本轮未改。不声称已从上游 FSDP 根治所有 view 问题。

## 9. init timeout 保持 300

### 在干什么

分布式有两类通信超时：

| 字段 | 管什么 | 默认（现） |
| --- | --- | --- |
| `init_timeout_seconds` | **建进程组、初始化、第一步同步** 等启动阶段 | **300** 秒 |
| `train_timeout_seconds` | **第 1 步之后** 训练中的集合通信 | 默认 100；部分 NPU 配方写成 **1800** |

### 问题

为了盖住 flex 首步/变长重编译，有人曾把 **全局默认** `init_timeout_seconds` 也改成 1800。启动阶段其实用不了那么久；默认抬太高会让「进程组没起来 / 卡在 init」时干等很久才报错，排查变慢。真正需要长超时的是 **训练中** 的 all-gather 等（见 §5 / §13），不该和 init 绑死成同一个全局默认。

### 改法

`d0f5101f`：`CommConfig.init_timeout_seconds` 默认改回 **300**。需要 1800 的场景写在具体配方上：

- 多数：只加长 `train_timeout_seconds=1800`
- 个别配方若首步前也要更长，再单独设自己的 `init_timeout_seconds`

### 边界

这是配置分层，不是改通信算法。§5 / §13 的 1800 仍然必要，只是不要污染全局 init 默认值。

## 10. Dist-MoE bf16 在 910B 上的专家计算

清单第 3 行：`dist_moe_bf16` **已接 NPU 路径可跑通**；`dist_moe_mxfp8` **未改未跑**。

### 上游假定（`release/0.4`）

官方路径同时要求三件事（910B 都不满足或未接）：

1. **可选包** `dist_moe`：`dist_moe/__init__.py` 的 `_LazyModule`，未安装则 `ModuleNotFoundError`（提示从 meta-pytorch/dist_moe 安装）。
2. **设备门** `runtime.py`：`device.type == "cuda"` 且 `get_device_capability(...)[0] >= 10`（SM100），否则 `ValueError: Dist-MoE requires an SM100-or-newer CUDA device`。
3. **CUDA annex 前向**：`create_context(...)` + `routed_experts.forward` 调 `dist_moe.routed_experts`（fused dispatch / GEMM / combine）。

NPU 专用路径 **只在** `torchtitan-ascend` 的 `patches/dist_moe/`：`ENTRYPOINT=torchtitan_ascend.train` 时 `setup()` → `apply_all()` wrap 公共 `DistMoeRuntime.__init__` / `DistMoeRoutedExperts.forward`。**本仓 Titan `models/common/dist_moe/` 保持上游零 NPU 改动**（不改 site-packages，也不在公共目录加 `npu.py`）。未经 ascend 入口的其它 Dist-MoE 模型不受影响。不做 mxfp8 假适配。

### `npu_dist_moe_forward` 实现要点

ascend patch 在 `x_TD.device.type == "npu"` 时进入，外层 `remat.region(..., recompute=False)`：

1. **上下文**：`_NpuDistMoeContext` 占位，**不** `dist_moe.create_context`。
2. **数据流**：`_exchange`（all-to-all）→ `cumsum(counts)` → `_grouped_mm(w13)` → `_activate` → `_grouped_mm(w2)` → 可选 postprocess → `_combine` 回 `(T,D)`。
3. **算子**：GEMM 用 `npu_grouped_matmul`（权重转布局，FSDP 先 `to_local()`）；SwiGLU 用 `npu_swiglu`。
4. **remat**：公开 `dispatch()`/`combine()` 会再开 `recompute=True` 区，不能嵌在外层 SAVE 里；故有 EP 时直接调 **`_dispatch` / `_combine`**。

配方侧：`disable_cuda_graphs=True`，收窄 `local_compile_regions`（见 [§14](#14-收窄-local_compile_regionscompile-与-remat)），`train_timeout_seconds=1800`（4 卡 `dp2×tp2×ep2`）。

### 为啥只有 bf16

不是硬件只能 bf16，而是 **这条路径按 dense bf16 写死**：

- `_grouped_mm` 把 activation/weight **cast 成 bf16** 再调 `npu_grouped_matmul`；无 MXFP8 scale / prepared weight。
- 只挂在 bf16 的 `DistMoeRoutedExperts` + 配方 `expert_precision="bf16"`。
- mxfp8 走 `MXFP8DistMoeRoutedExperts` + `BlockScaledConfig` / `PreparedWeight` 等官方 annex API；ascend **不做** mxfp8 假适配，配置期仍会卡在官方包依赖上。

要支持 mxfp8 需另开 NPU 的 block-scaled 权重准备与对应 GEMM，不能只扩现有 `npu_dist_moe_forward`。

### 验证

2026-10-30，`NGPU=4`、确定性开关，bf16 连跑两次 10 步 loss / grad_norm 对齐。见 `docs/npu_debugmodel_verification.md` 第 3 行。

## 11. 旧 torch 上 Dist-MoE / FSDP 的兼容导入（已撤销）

这和 §10「NPU 上怎么算 Dist-MoE」不是一回事：当初只解决 **一 import / 一构造就报错**。

### 历史问题（2.14 stable 现场）

早期验收钉在 **torch 2.14.0 stable / `2.14.0+cpu`** 时会出现：

- Dist-MoE：`from torch.distributed.pipelining import analyze_pipeline_activation_liveness, ...` → **`ImportError`**
- FSDP：无条件传 `param_dtype_override_fn=...` → **构造时报错**

曾用 `f1c0e844` / TT-20、TT-21 做 try/except 与签名探测。

### A2（2.15）复查结论

对当前环境 `torch 2.15.0+cpu`（`site-packages`）静态确认：

- `MixedPrecisionPolicy` **已有** `param_dtype_override_fn`
- `torch.distributed.pipelining` **已导出** `analyze_pipeline_activation_liveness` / `PipelineStageInfo`

因此兼容代码已从本分支与 torchtitan-ascend patches 删除；缺 API 时不再兜底，直接依赖 2.15。真正跑 Dist-MoE bf16 仍靠 §10。

## 12. 上下文并行的 block mask 求和被编成 simt_only

`muse_spark_debugmodel_fsdp2_cp2_ep4` 在 `_cp_shard` 里编译 `create_block_mask`。query 被 CP 切开后，mask 先垫到 128 的倍数，再按 `128 x 128` 的块求和。这个 reduction 走 `NPUTritonKernel`，元数据里的 `npu_kernel_type` 被写成 `simd_simt_mix`。选块时会先生成 `compile_mode="simt_only"`。910B4 不是 Ascend 950，Triton 直接拒绝这个模式，于是整组配置都算失败：

```text
NoTritonConfigsError: No valid triton configs for kernel
triton_red_fused__to_copy_bitwise_and_eq_gt_lt_permute_sum_view_3
ValueError: compile_mode="simt_only" is supported only on Ascend 950 targets.
```

`torch_npu/_inductor/runtime/triton_heuristics.py` 里，非 Ascend 950 不再为 `simd_simt_mix` 生成 `simt_only` 配置，只保留 `simd_simt_template`。910B 上这个模式走普通 SIMD。补丁在 `docs/npu_910b_patches/triton_heuristics.py.patch`。

## 13. 第 4 行多模态配方的训练超时

`muse_spark_debugmodel_mm` 是清单第 4 行：8 卡 `dp_shard=2, tp=2, cp=2, ep=2`，Ulysses。视觉 token 长度一变，flex attention 就要重新编译。第 2 行③那条 EP4、AllGatherKV 的变体已经把 `train_timeout_seconds` 设成 1800；清单上的这份配方原来没有。

A5 验收配方（提交）不设 1800。910B 本机测试用 gitignored `torchtitan_ascend/recipes/muse_spark_910b_local.py`。flex / `env_npu_910b` 同样只本地 apply，不进 A5 默认路径。

2026-10-30 在 `179.173.2.3` 上，`NGPU=8` 连跑两次 10 步可对齐。逐步数字在 `docs/npu_debugmodel_verification.md` 第 4 行。

（同波次 8 卡多模态在补丁齐全后可跑完 10 步，见 verification 文档；此处不另占编号。）

---

## 14. 收窄 local_compile_regions（compile 与 remat）

### 在干什么

`model.local_compile_regions` 是一份 **函数名白名单**，不是整模 `torch.compile`。

机制（`torchtitan/distributed/local_compile.py`）：

1. 源码用 `@local_compile("名字")` 标出一小段函数（RoPE、RMSNorm、attention 前后处理、loss 等）。
2. 启动时 `apply_local_compile(model.local_compile_regions)`：名字 **在名单里** → 对该函数 `torch.compile(..., fullgraph=True)`；**不在** → 保持 eager。
3. 大块算子（GEMM、flex attention、MoE dispatch 等）一般不走这套，或另有路径。

上游 Muse Spark（`release/0.4`，`muse_spark/model.py`）默认开得很宽，目的是把「胶水」融进 Inductor、少打 HBM：

```text
loss, fp32_to_bf16_split, rope,
muse_spark_attention_qk, muse_spark_attention_output,
muse_spark_input_scaled_rmsnorm, muse_spark_logit_postprocess,
muse_spark_residual, muse_spark_rmsnorm
```

### 问题

在 910B 上，上述 region（尤其 `rope` / `muse_spark_attention_qk` / norm / residual）常落在 **activation remat 可重算区** 里。反传重算会再走一遍已 compile 的图；这次特化出来的图和前向保存的 checkpoint **对不上**——保存的 NPU tensor 在重算路径里变成 CPU scalar 一类东西，checkpoint metadata 崩，训练走不下去。

这是 **compile × remat 在 NPU 上的组合问题**，不是 Muse 数学要改名单。GPU 上默认长名单仍是性能向的。

### 改法

910B 本机用 gitignored `muse_spark_910b_local.py` 收窄到已验证能跑完一步的集合（**不进提交**；A5 用干净上游 region 名单）：

```python
config.model.local_compile_regions = ["loss", "fp32_to_bf16_split"]
```

落点：

| 配方 | 为何再设一遍 |
| --- | --- |
| `muse_spark_debugmodel_npu`（910b_local） | 覆盖上游默认长名单 |
| `muse_spark_debugmodel_dist_moe_bf16_npu`（910b_local） | Dist-MoE 路径同样避开 remat+compile |
| `muse_spark_debugmodel_mm_npu`（910b_local） | `mm` 会 **换掉整个 `model_config`**，又把默认长名单带回来，必须再收窄 |

不改 `muse_spark/model.py` 的 `default_factory`，也不改上游 `torchtitan_recipes` 社区用例；只在 ascend 的 910B 验收配方出口覆盖。

### 边界

- 用正确性换掉那部分 local-compile 加速；未做「NPU 上重新选可 compile 的 region」的性能回补。
- 与 §10 配方侧的「收窄 local_compile_regions」是同一改动；细节以本节为准。
- 与 A1（remat 空指针占位）不同：A1 是 placeholder storage；本节是 **哪些函数被 `torch.compile`**。

## A1. 反向重算里的空指针占位张量

属上表 **A1**。不再在本仓库记录改法，也不再打 `torch_remat` 占位补丁。用 `torchtitan-ascend` 的 `torchtitan_ascend/patches/storage.py` 和安装期 `fixes/torch_npu/storage.patch`（跟踪 Ascend/pytorch !47944）。

## E1. 第 5 行改读本地 UltraData JSONL

属上表 **E1**（环境问题，非仓库缺陷）。

对应改动只在 **`npu-muse-spark-debug-r04`**：`f99ea981`（本地 JSONL）。**本分支不含该 recipe 改动**；仍走默认 `HuggingFaceStreamingSource`。

`muse_spark_debugmodel_ultradata_lora_sft` 默认数据集 `openbmb/UltraData-SFT-Agent-2609`。本机 `huggingface.co:443` 会被连接重置，流式数据起不来。r04 上改读 `MUSE_SPARK_ULTRADATA_JSONL_PATTERN` / 本地 JSONL（`IndexedJsonlSource`）。JSONL 不进 git（`assets/datasets/`）。

不为线性度 / 性能对照另开 4 卡 EP 配方；本分支只保留清单默认 8 卡 10 步跑通配置。
