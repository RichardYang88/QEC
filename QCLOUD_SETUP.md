# 本源量子云(OriginQ)真机实验接入指南

VSCR [[5,1,3]] 实验已实现硬件迁移管线:`qcloud_vscr.py`。
本地仿真部分(电路正确性验证)不需要任何账号;真机提交需要你完成下面的注册。

## 一、为什么选本源量子云

| 平台 | SDK | 真机 | 中间测量 | 结论 |
|---|---|---|---|---|
| **本源量子云** qcloud.originqc.com.cn | pyqpanda(已装 3.8.5) | 悟空/悟源系列超导芯片 | 支持 | **首选** |
| 百度量子平台 | Paddle Quantum | 10 比特超导 | 支持有限 | 备选 |
| 量子创新研究院云平台 | 网页为主 | 有 | — | 不便程序化提交 |

pyqpanda 的 `QCloud` 类原生支持 `real_chip_measure` / `batch_real_chip_measure`,
并自带 `is_mapping`(自动比特映射,不需要手工对齐芯片拓扑)、
`is_amend`(电路修正)、读出误差缓解/ZNE 等平台级功能。

## 二、你需要完成的注册步骤

1. 打开 http://qcloud.originqc.com.cn/ ,用手机号注册并实名认证;
2. 登录后进入 `用户中心 → API 密钥`,复制 API Key(即 token);
3. 在 `量子计算` 页面确认你的账号可用的**真机芯片型号**与剩余 shot 配额
   (新用户一般有免费模拟器配额;真机配额可能需要申请/购买,若有导师课题组的
   机时账号更好);
4. 把 API Key 发给我(或自行设置环境变量):
   ```bash
   export ORIGINQ_TOKEN='你的APIKey'
   ```
   并告诉我可用的芯片编号(chip_id)。

## 三、实验协议(与论文仿真严格对应)

- 9 比特:数据 q0..q4 + 症候辅助 a0..a3;
- Clifford 编码电路由 Qiskit 稳定子合成 + 自适应相位对齐(逻辑帧与仿真约定
  完全一致,已数值验证 overlap=1);
- 去极化噪声 = 每个数据比特独立采样 Pauli 帧(I 概率 1−p,X/Y/Z 各 p/3),
  帧平均即精确重构仿真中的信道期望;
- 症候提取:数据比特基旋转(X→H,Y→S†H)→ CNOT(j→a) 奇偶累积 → 逆旋转,
  中间测量 4 个辅助比特(等价于对 16 个 P_s 的投影测量);
- 恢复:每个症候分支 s 一个电路,加载**已训练好的** R_s(60 个角度,
  RZZ 分解为 CNOT-Rz-CNOT);真机上不做变分训练(变分训练留在仿真端,
  真机只执行学好的恢复电路——这是硬件 QEC 论文的标准做法);
- 读出:解码 U_enc† + 逻辑基旋转,测 5 个数据比特;
  后选辅助比特 = s 的 shots,F_s = P(00000 | s),F = Σ_s P(s)·F_s;
- 检验:同帧 Pauli 的"理想协议仿真值"作为逐点对照(硬件 vs 理想的差 =
  真机门误差/读出误差的影响),另有密集仿真参考曲线。

## 四、运行方式

```bash
# 1) 本地精确验证(无需账号):电路级 vs 密度矩阵仿真器,应 <1e-9
./qenv/bin/python qcloud_vscr.py --mode verify

# 2) 本地 pyqpanda CPUQVM 采样冒烟测试(无需账号)
./qenv/bin/python qcloud_vscr.py --mode sample

# 3) 真机提交(需要 token)
export ORIGINQ_TOKEN='...'
./qenv/bin/python qcloud_vscr.py --mode cloud \
    --chip-id <你的芯片编号> --shots 4000 \
    --p-values 0.02,0.08 --frames 2 --states 0,+ \
    --dump-raw raw_first.json        # 首次建议保留原始返回以便核对格式
```

默认预算:2 个逻辑态 × 2 个 p × 2 个噪声帧 × 16 分支 = 128 条电路 × 4000 shots。
若配额紧张,可减为 `--frames 1 --states 0 --p-values 0.05`(16 条电路)。
首次上真机建议先 `--no-batch --p-values 0.05 --frames 1 --states 0` 单条试跑,
确认返回格式与比特位序(代码里有自动位序检测 detect_layout)。

输出:`qcloud_results_<tag>.json` + `figures/fig_qcloud_hardware_<tag>.png`
(真机点 + 误差棒 vs 同帧理想仿真 vs 密集参考曲线)。

## 五、实测记录(2026-09-08 深夜)与 pyqpanda 3.8.5 注意事项

> **⚠️ 已废弃**:本节及 `qcloud_auto.py` 守护进程走的是旧 `pyqpanda` QCloud 接口,
> 该接口对此账号已不可用(芯片枚举/维护状态均失效)。现行可用路径见 **第六节**
> (`pyqpanda3.qcloud` + 字符串芯片 ID `WK_C180`)。

token 已写入 `.originq_token`(已加入 .gitignore,勿提交),认证已通过。
实测发现的坑与已实施的修复(全部已在 `qcloud_vscr.py` 中落实):

| # | 现象 | 修复 |
|---|---|---|
| 1 | `QCloud.init(token)` 只调 C++ 基类,`real_chip_measure` 报 `AttributeError: enable_pqc_encryption` | 改用 `cloud.init_qvm(user_token=token)` |
| 2 | 把 `QProg` 对象直接传给 `real_chip_measure` → **C++ 层段错误**(exit 139) | 先 `convert_qprog_to_originir(prog, qvm)`(注意**必须传两个参数**)转成 OriginIR 字符串再提交(签名本就接受 `Union[QProg, str]`) |
| 3 | `batch_real_chip_measure` 返回的是**概率分布** `List[Dict[str,float]]` 而非 counts | `_normalize_counts` 自动判别:值和 ≈1 → 乘以 shots 还原 counts |
| 4 | 云端返回位序未知(c0 在前/在后/整体反转) | `detect_layout` 用分支一致性对 4 种假设(ancilla 前/后 × 正序/反转)自动打分选择 |
| 5 | 真机编译器可能拒绝中间测量 | `--late-measure`:辅助比特测量挪到电路末尾(无经典反馈,联合分布严格不变,已在 CPUQVM 上逐位验证等价) |

芯片枚举(`real_chip_type`):`2=悟源D5`,`5=悟源D4`,`7=悟源D3`(本账号 resource is null),`72=悟空 72 比特`。

**当前状态**:深夜时段 chip 2/5/72 均返回 `Quantum computer under maintenance`。
已部署自动重试守护进程:

```bash
cd /home/yqc/github/QEC
(nohup ./qenv/bin/python -u qcloud_auto.py > qcloud_auto.log 2>&1 &)
tail -f qcloud_auto.log     # 观察进度
```

守护进程逻辑:每 15 分钟用真实 9 比特分支电路(100 shots)探测 chip 2→72→5;
若遇非维护类错误(如编译器拒绝中间测量)自动切换 `late_measure` 变体重试;
任一芯片接受任务后,自动执行完整实验(128 电路 × 4000 shots,每 32 条一批
提交,batch 失败自动降级为逐条 `--no-batch`),并生成
`qcloud_results_full_chip<id>.json` + `figures/fig_qcloud_hardware_full_chip<id>.png`。

## 六、pyqpanda3 真机协议最终结论(2026-09-10 深夜实测,现行方案)

现行管线:`qcloud_vscr_new.py`(pyqpanda3)+ `QCloudService.backend('WK_C180')`。
芯片 ID 为字符串:`WK_C180`(队列较短,首选)、`WK_C180_2`(排队任务多)、
`PQPUMESH8`、`HanYuan_01`(离线)。在线状态可用 `service.backends()` 免费查询。

### 6.1 三个决定成败的实测发现

| # | 发现 | 证据 | 对策 |
|---|---|---|---|
| 1 | **无 QCloudOptions 的默认提交返回近均匀垃圾**(trial 16×2000,最高 key 仅 3.4%,任何位序假设 data-hits=0) | `raw_trial_batch.json` | `QCloudOptions`:`set_amend(True)` + `set_mapping(True)` + `set_optimization(True)` + `set_specified_block(best_qubit_blocks(9))` |
| 2 | **WK_C180 上中间测量(mid-measure)会毁掉数据比特的后续计算**:即使带正确 options,mid 变体仍全无信号(sel=97 而 hit=0);late 变体出现真实信号 | `qcloud_verify_options.py` → `raw_verify_options.json`:late+IIXII+s=12 电路 top key `'001100000'`(44/371),anc 位反转=`'1100'`=s12,data=`'00000'`,**F_s=0.80**;伴随 `'000100000'`(50)=a1 辅助比特 T1 弛豫/读出头 flip 峰 | 全量实验必须 `--late-measure`。副作用:辅助比特在 recovery+decode 期间空闲,弛豫使症候读数偏向低权重 → 该系统性效应计入硬件基准(与理想仿真的 gap 的一部分) |
| 3 | **返回位序 = 整串反转的 cb0..cb8**,即 string=`[c3c2c1c0|c8c7c6c5c4]`;且旧 `detect_layout` 用 Σselected 打分会被全零 key 骗过(错选 `[5:9]rev=False`) | 4 假设 hits 对比:`(slice(5,9),rev=True)` hits=53,其余 ≈9 | `detect_layout` 已改为 **(data-hits, selected) 字典序**打分,离线验证选择正确 |

最优物理比特块(`backend.best_qubit_blocks(9)`,免费查询):
`[147, 155, 157, 164, 165, 166, 174, 175, 176]`(submit_cloud 会自动查询并固定)。

### 6.2 配额(QPU time)现状 — 当前阻塞点

- 报错 `RuntimeError: QPU time is insufficient. Please purchase more.` = **真机机时**
  不足(注意:网站"账户余额"充值 ≠ 真机机时,机时需在控制台单独购买/申请)。
- 时间线:22:23 全量 4000 shots 跑到 batch2 时机时耗尽(batch1 的 128K shots
  数据未落盘,教训 → 已改为**每 chunk 增量写盘**);充值后 1600-shots 验证任务
  成功;随后 32×1000 全量与 16×100 探针**均被拒** → 机时又见底。
- 全量预算:128 电路 × 1000 shots = **128K shots**(4 个任务 × 32 电路,
  每 chunk 恰为 2 个完整数据点,中途断掉已得数据可分析)。
  更省: `--shots 500`(64K,单点误差 ±2%)。

### 6.3 机时到账后的运行手册

```bash
cd /home/yqc/github/QEC
# 1) (可选)最小探针:1 个数据点 × 100 shots = 1600 shots,端到端验证管线
./qenv/bin/python -u qcloud_probe_point.py

# 2) 全量实验(late-measure + options + 最优块,增量落盘 qcloud_raw_full.json)
bash launch_full.sh          # 等价命令见脚本内;日志 qcloud_full.log
tail -f qcloud_full.log

# 3) 输出: qcloud_results_full_late_1000.json
#          figures/fig_qcloud_hardware_full_late_1000.png
#    (真机点+误差棒 vs 同帧理想仿真 x vs 密集参考曲线)
```

手动等价命令:

```bash
./qenv/bin/python -u qcloud_vscr_new.py --mode cloud \
    --chip-id WK_C180 --token "$(cat .originq_token)" \
    --shots 1000 --p-values 0.02,0.08 --frames 2 --states 0,+ \
    --late-measure --chunk 32 --job-timeout 21600 \
    --dump-raw qcloud_raw_full.json --tag full_late_1000
```

新增 CLI:`--no-options`(退回无 options 提交,不推荐)、
`--block "147,155,..."`(手动指定物理块,默认自动查询)。

### 6.4 论文模型角度与离线分析(2026-09 论文定稿配套)

- 论文仿真模型为 **warm-start VSCR**(见 `vscr_paper.py`),角度文件
  `vscr_angles_paper_{dep,ad,mixed}.npz`;真机全量运行前请设置
  `export VSCR_ANGLES_FILE=$PWD/vscr_angles_paper_dep.npz`
  (`load_angles` 支持该环境变量覆盖默认的 `vscr_angles_dep.npz`)。
  `vscr_angles_dep.npz`(v1)保留不动——4 电路可行性数据由它产生。
- 新增 `--analyze <raw.json>` 离线分析模式:读取 `--dump-raw` 落盘的
  原始 counts 列表,按当前 CLI 的 plan(需与提交时一致的
  `--p-values/--frames/--states/--late-measure/--shots`)重建 meta,
  执行 detect_layout → aggregate → 绘图,无需 token。例:
  ```bash
  ./qenv/bin/python -u qcloud_vscr_new.py --mode cloud \
      --analyze qcloud_raw_full.json --p-values 0.02,0.08 \
      --frames 2 --states 0,+ --late-measure --shots 1000 --tag full_paper
  ```
- 全量数据接入论文的完整流程见 `paper/HW_FULL_INTEGRATION.md`。

## 七、实测设备混入度实验(2026-10,`hw_mixedness.py`)—— 机时紧张时的方案

第六节的全量基准要 128 电路 ×1000 shots,机时见底时跑不动。本节是一个**为小机时
重新设计**的实验:它不再去补全 p 扫描,而是去测论文最新那一层(ED Table 10d/10e
的混入边界)唯一还缺的东西——**设备真实交付的逻辑态有多混**。

ED Table 10d 现在的 `r = 0.9375`、码空间权重 `0.6068`、泄漏 `39%` 全是**推出来的**:
把 `hw_error_budget.json` 反推的平均每指令错误率送进 36 门编码器当作各向同性去极化
信道。本节把它换成**测出来的**。

### 7.1 后端选择:180-2(`WK_C180_2`),依据是本电路的功效研究

不是看厂商指标排序,而是把三块芯片的指标送进**本实验真实要提交的电路**
(`--mode simulate`,零机时)算出可分辨能力。8000 shots/电路时:

| 量 | 180-2 | 180 | ~~180-3~~ |
|---|---|---|---|
| 真值码空间权重 $w$ | **0.3396** | 0.2675 | ~~0.1780~~ |
| 真值半径 $r$ | **0.855–0.881** | 0.797–0.830 | ~~0.669–0.713~~ |
| $r$ 的 95% CI 半宽 | **0.0278** | 0.0489 | — |
| 达到 ±0.02 所需 shots | **15 486** | 47 918 | — |
| M2 的 $\lvert L_Z-L_X\rvert$ 95% 上界 | **±0.0225(项跨度的 4.7%)** | ±0.0251(6.5%) | — |
| 零对照(null control)最大偏离 | **1.0σ** | 1.6σ | — |
| M3 替代模型的预测结局 | **失败:−3.4σ(d=1)/ −3.9σ(d=2)** | 通过:−0.3σ / 1.9σ | — |
| 6 个种子的 $r$ 散布 / $w$ 散布 | **0.0149 / 0.0026** | 0.0237 / 0.0048 | — |
| 6 个种子的 M2 z 均值 / rms / max | +0.50 / 1.21 / 1.99 | −0.51 / 1.29 / 2.79 | — |

**`180-3` 一列作废,但作废理由要改**:2026-09-08 的 `devs.txt` 快照里确实没有
`WK_C180_3`,而 2026-10-11 的活 `service.backends()` 返回 `WK_C180_3: true` ——
该芯片在那之后上线了。所以这一列作废是因为**不在本次范围**(用户只考虑
`WK_C180_2`),而不是因为设备不存在;详见 §8.0 与 §8.8.1。

> **⚠️ 本节所有功效数字都由引用表值推出,已被实测标定推翻。** §8.8.3 显示
> `WK_C180_2` 厂商推荐块的 f2q 实测为 **0.99275**,优于引用的 0.9831,于是预测
> radius 从 0.8550 升到 0.9478,`r<1` 的显著性从 1.8σ 降到 **0.60σ**,5σ 需要
> **18 651 s ≈ 5.2 小时** QPU(预算的 62 倍)。下表的相对比较(180-2 误差棒更窄)
> 仍然成立,但**绝对功效要按此下调**;M3 的 −3.4σ 尤其不再可信。
上表的两列有效数据是 2026-10-10 用 `--mode simulate --seeds 6` 重跑的,
走的是**与真机完全相同的分析路径**(含 tomo 自带标定电路)。

180-2 在**同样 shots 下误差棒窄 1.76 倍**,达到同样精度所需 shots 少 3.1 倍。
$T_2^{echo}$ 最长(19.77 µs vs 13.78/19.01)对 bench 阶段的 late-measure 尤其重要——
4 个辅助比特要空转过 recovery+decode,已记录的弛豫卫星峰($s_8/s_{12}=1.75$)就是这个效应。

**最后一行 M3 才是选 180-2 的真正理由,而不只是"误差棒更窄"**:各向同性去极化
替代模型在 180-2 上被预测**决定性地失败**(−3.4σ/−3.9σ),而在 180 上会**通过**
(−0.3σ/1.9σ)。也就是说,如果在 180 上做,M3 这个检验得不出任何张力、变成
无信息量的一项;只有在 180-2 上,它才能定量地证明"用单一各向同性速率推 $r$
是不成立的",而这正是本实验要去**测** $r$ 的全部动机。

三个必须知道的前提:
1. **芯片 ID 已确认**(2026-10-11 免费 preflight,§8.8.1):`WK_C180`、
   `WK_C180_2`、`WK_C180_3`、`PQPUMESH8` 在线,`HanYuan_01` 离线。`WK_C180_3` 在
   2026-09-08 的 `devs.txt` 快照里还不存在,是之后上线的。本方案只用 `WK_C180_2`,
   其 `best_qubit_blocks(5)` = `[39,48,49,58,67]`(块连通,4 条 CZ 边成一棵树)。
2. **换芯片 = 新数据必须自带 chip id**,且不能把新测的 $r$ 与旧的 `0.9375` 直接比
   (不同芯片、不同深度)。模型-实测比较改成同芯片同深度自洽:用本次实测的 $w$
   反推单一 $p_{instr}$,再让它**预测** $r$。旧的 WK_C180 run 保留为独立标注的平台表征。
3. **mid-measure 失效是 WK_C180 的设备特性**,必须在 180-2 上重测(bench 阶段头两条
   电路就是这个 A/B)。如果 180-2 的中间测量能用,后续全量扫描的后选效率会从 ~15% 大幅上升。

### 7.2 电路与预算

| Job | 比特块 | 电路 | shots | 小计 | 内容 |
|---|---|---|---|---|---|
| probe | 5 | 8 | 1500 | 12 000 | 2+1 读出标定(重复的那条夹出漂移)+ 5 条单 X 指纹电路 |
| tomo | 5 | 16 | 8000 | 128 000 | 4 输入 × 3 基(depth 1,78 指令/36 CNOT)+ 4 条 depth 2(150 指令/72 CNOT,含 2 个零对照) |
| bench | 9 | 34 | 1000 | 34 000 | mid/late A/B ×2 + 全 16 分支求和(p=0,ψ∈{0,+},late-measure,论文 warm-start 角度) |
| 合计 | | **58** | | **174 000** | 纯执行时间约 4 s(15–40 µs/shot) |

核心电路:`prep(ψ) on q4 → U_enc(36) → U_enc†(36) → R_B on q4 → MEAS q0..q4`。
解码后"q0..q3 全 0"**就是** `device_prep` 解析地施加的投影 $VV^\dagger$,q4 携带逻辑态,
所以同一批 counts 同时给出 $w$、$\langle\sigma_B\rangle_{cs}$、$r$、purity。
**故意不加 `val` 旋转**——加了会把 $|1\rangle$ 的符号抹掉,M2 就测不到了。

### 7.3 五个可测断言,和一个明确不测的东西

| | 断言 | 判据 | 8000 shots 时的预期能力 |
|---|---|---|---|
| M1 | 设备交付的逻辑态是混的 | 实测 $r$ 的 95% CI 上界 < 1 | $r\approx0.86\pm0.028$,$w\approx0.340\pm0.005$ |
| M2 | **线性分数在真机上精确可分解** | $\lvert L_Z-L_X\rvert$ 的 CI 覆盖 0 | ±0.023,而单项 $\langle Z_L\rangle$ 跨度 ~0.96(即 4.7%) |
| M3 | 各向同性去极化替代模型受检 | 1 个拟合参数($p_{instr}$ 由实测 $w$ 定)预测 $r$,看是否落在 2σ 内 | 功效研究**预测它会失败:−3.4σ(d=1)/ −3.9σ(d=2)**;同一检验在 180 上反而会通过(−0.3σ/1.9σ),见 §7.1 |
| M4 | 自测 5 比特读出指派矩阵 | 两条标定电路 + 重复条夹出漂移 | 与 `ChipInfo` 的厂商标定**互为对照**(该端点自 2026-10-11 起不再 401,§8.8.1);自测值还含平台少给 shots 与漂移的真实影响,厂商值不含 |
| M5 | depth 1 vs depth 2 分离"每门"与"每时间"损耗 | 同一 $p_{instr}$ 是否解释两个深度;2 个零对照须 < 3σ | 零对照 1.0σ |

M2 是 `mixed_input_boundary.selftests` 第 (d) 项(线性分数精确分解,数值上验到
5.55e-17)的**真机版**:$\rho_{in}=I/2$ 有两个不同的纯态系综,无条件 $\langle Z_L\rangle$
是 $\rho_{in}$ 的线性泛函,所以 $L_Z=\mathrm{Tr}[Z_L M(I/2)]=L_X$ 对**任意**信道精确成立。
单项跨度 ~1 而两个平均必须相等,这才是有内容的检验。**必须用无条件值**:后选不是
线性泛函,条件版只作为泄漏不对称的诊断量报告,永远不作为 M2 的判据。

**明确不测:over-read $\delta^T C\delta$ 本身。** 在预测半径下它是
$(1-r^2)(F_{Haar}-c_0)\approx0.007$(depolarizing),而分支求和保真度的散粒噪声
$\sigma(F)\approx\sqrt{F(1-F)/N_{eff}}$,本设备症候读出下 $N_{eff}\approx0.15\times shots$,
1000 shots 时 $\sigma(F)\approx0.033$、$\sigma(F^2)\approx0.05$——是效应的 7 倍。
压到 0.001 需要每分支 $N_{eff}\approx4\times10^5$,合计 ~$5\times10^7$ shots。
所以真机测的是**给 over-read 定价的半径**和**线性精确性**,over-read 数值本身仍由
已审计到 1e-14 的解析结果给出。这与 ED Table 3 已有的"受限于 shot 预算而非门数"一致。

**功效研究还预测了一个真结果**:180-2 的单比特错误率(0.0007)与双比特错误率(0.0169)
相差 24 倍,用一个各向同性速率去同时匹配"泄漏"和"逻辑相干"是不可能的——
拟合到 $w$ 的 $p_{instr}=0.01169$ 会把 $r$ **低估** 0.048(depth 1,−3.4σ)和 0.100
(depth 2,−3.9σ;$p_{instr}=0.01245$)。这正是"必须去测 $r$ 而不是推 $r$"的定量理由,
也支持论文本来就把 $r$ 称作量级估计。M3 若如实失败,就按失败报告。

同一检验在 180 上**不会**失败(−0.3σ / 1.9σ,拟合的 $p_{instr}=0.01464$/$0.01298$),
因为那台设备的读出误差大得多(0.0740 vs 0.0388),各向同性模型恰好能吸收它。
所以后端选择与"M3 是否有信息量"是耦合的,这一条已写进 §7.1 的表。

### 7.4 运行手册

> **本节的第 1–3 步已被 §8.7 取代**(§8 是按 300 s 机时重排的 job 结构、shots 取值与
> 放弃规则)。这里保留的是零机时验证清单,它仍然有效且必须先全绿。

```bash
cd /home/yqc/github/QEC

# ---- 第 0 步:零机时验证,必须全绿再花钱 ----
./qenv/bin/python hw_mixedness.py --mode plan           # 打印预算表 + shots/成本/精度权衡
./qenv/bin/python hw_mixedness.py --mode verify         # 9 个锚点 + 每条电路的精确理想参考
./qenv/bin/python run_selftests.py hwmixedness          # 同一套锚点,走全量自检框架
./qenv/bin/python hw_mixedness.py --mode sample --shots 3000
#   本地 CPUQVM 无噪声跑完整条分析链:w=1.000, r=1.000, 零对照 <1.1σ,
#   位序从指纹电路唯一确定(σ=[4,3,2,1,0],margin 1.2e5)
./qenv/bin/python hw_mixedness.py --mode simulate --seeds 6 \
    --vendor 0.9993,0.9831,0.9612 --json /tmp/power_180-2.json
#   功效研究:两块真实存在的芯片各跑一次即得 §7.1 的表(WK_C180_3 不存在,见 §8.0)

# ---- 第 1 步:免费云查询(不提交任何电路) ----
./qenv/bin/python hw_mixedness.py --mode preflight \
    --chip-ids WK_C180_2,WK_C180,WK_C180_3 --vendor 0.9993,0.9831,0.9612
#   -> hw_mixedness_preflight.json:在线状态/队列、best_qubit_blocks(5) 与 (9)、
#      以及再一次尝试 ChipInfo —— 【2026-10-11 已成功】写入厂商逐比特数据,
#      不再 401;详见 §8.8.1–8.8.2

# ---- 第 2 步:三个 job,按优先级,逐个增量落盘 ----
bash launch_mixedness.sh probe      # 12 000 shots;先看它花掉多少机时/墙钟
tail -f qcloud_mixedness_probe.log
bash launch_mixedness.sh tomo       # 128 000 shots
bash launch_mixedness.sh bench      #  34 000 shots

# ---- 第 3 步:离线重分析(无需 token) ----
./qenv/bin/python hw_mixedness.py --mode analyze --stage all
#   -> hw_mixedness.json + figures/fig_hw_mixedness_<tag>.png
```

机时按 shot 计还是按 job 墙钟计,**只有 probe 能告诉我们**:纯执行时间合计约 4 s,
所以若 300 s 指执行机时,余量约 75 倍,应把 `--shots` 提到 15 900($r$ 达 ±0.02);
若指 job 墙钟,则**三个 job 的结构本身就是预算**,不能增加 job 数。
被拒时的行为已内建:凡报错含 `QPU time`/`insufficient`/`quota`,`--shots` 自动减半重试,
直到 `SHOTS_FLOOR=1000`,每次尝试都写进 artifact 的 `refused_attempts`。

### 7.5 建这套代码时踩到并已固化为自检的坑

| # | 坑 | 现在怎么防 |
|---|---|---|
| 1 | **无噪声下 tomo 电路的理想分布只支撑在 `{00000,00001}`**,5!=120 个位序假设里有 **24 个完全简并**(q0..q3 的排列无法区分) | 自检 (f) 断言恰好 24 个并列、且**所有估计量在 24 个之间完全相同**(码空间条件是"另外 4 位全 0",排列不变);另加 5 条单 X 指纹电路把位序唯一钉死(3 个植入顺序全部唯一恢复,margin > 1e5) |
| 2 | **`prep` 被施加两次**:`build_tomo` 的门表已含 prep,再传 `psi_vec=prep_vec(ψ)` 就变成 $H^2=I$,$|+\rangle$ 静默变成 $|0\rangle$,所有系综平均变成垃圾 | 自检 (e) 与 `simulate_device` 都固定一种约定;功效研究第一次跑就把 M2 打到 33σ 暴露了它 |
| 3 | **clip-and-renormalize 的读出缓解是有偏的**:32×32 逆矩阵把噪声放大 $(1-2e)^{-5}$,裁剪负值引入偏置,表现是 M2 出现假的 2.8σ 违背 | 改用标准的"缓解观测量"$\tilde O = A^{-T}O$:对任意逐比特读出信道**精确无偏**、不需要求逆分布、不需要裁剪,且自带精确标准误(含比值分子分母的协方差)。自检 (d) 断言无偏到 1e-15,并在对称特例退化到解析对比度除法 |
| 4 | 我最初写的"解析对照"公式 $(P_0-P_1)/(1-e_{01}-e_{10})$ 在 $e_{01}\ne e_{10}$ 时**是错的** | 自检 (d) 改成只断言对称特例 $e_{01}=e_{10}=e$ 时退化为 $(P_0-P_1)/(1-2e)$;非对称情形由 $A^{-T}O$ 精确处理 |
| 5 | 把多个比特各自的 Kraus 集合摊平成一个列表求和,会把"退极化 a"和"退极化 b"当成**互斥事件**,每条双比特门把 $\mathrm{Tr}\rho$ 乘 2 | `_propagate` 逐比特施加信道,并**每步断言 $\mathrm{Tr}\rho=1$**;自检 (c) 用审计过的 `device_prep` 数值($r$、$w$、purity、leakage)锚定,残差 **0.00e+00** |
| 6 | M2 的原假设是"差为 0",单次抽样有 ~5% 概率落到 2σ 外 | `--seeds N` 报告 z 分数的**分布**(6 个种子:均值 +0.50、rms 1.21、max 1.99),而不是单次结果;$r$ 的种子间散度 0.0120 与 bootstrap CI 半宽 0.0282/1.96=0.0144 一致,这才是 CI 宽度可信的证据 |

### 7.6 数据到手后要补的论文工作(现在还没做)

1. `paper/audit_numbers.py` 新增 §18:从 `--dump-raw` 的原始 counts **第二次独立重算**
   `hw_mixedness.json` 里每一个数(沿用 §15/§16 对派生产物的约定),并断言位序假设唯一、
   $0<w\le1$、$r\le1+10^{-12}$、purity$=(1+r^2)/2$、M2 的 CI、以及 M3 残差的符号与量级被记录。
2. `paper/make_ed.py` 新增 **ED Table 10f/10g/10h**(实测设备混入度;读出指派矩阵 +
   线性分数系综等式;同深度模型 vs 实测 + 深度扫描),挂在 10a–10e 的混入边界块后面
   ——现有编号到 14,10f 起最自然。
3. `paper/main.tex` 硬件段:把"model-implied $r=0.9375$"升级为"measured $r_{hw}=\dots$
   (chip, depth)",Data Availability 补上自测标定。**`preflight` 已于 2026-10-11
   拿到厂商逐比特数据**,所以 §16 与 ED Table 9 里"401 被拒"的表述**必须**升级:
   ChipInfo 自该日起可访问,`WK_C180_2` 的 q39–86 有标定(17 比特 / 20 边)。
   同时 §8.8.3 的实测 f2q=0.99275 优于引用表值 0.9831,任何由引用表值推出的功效
   陈述都要按实测重算(σ 从 1.66 降到 0.60)。
4. 不动 `mixed_input_boundary.json`(audit §17 已锁字段);同深度的模型预测由
   `hw_mixedness.py` 自己算,只写新 artifact。

## 八、300 秒机时的完整执行方案(2026-10-10 定稿,待审)

§7 是**科学设计**,本节是**在 300 s 机时下如何把它执行完**。两者的电路相同,
但 job 的切分、shots 的取值和放弃规则全部由本节的成本分析决定。

### 8.0 先纠正两处我此前说过、但档案不支持的话

**(1) `WK_C180_3` 当时不在 API 列表里 —— 但它确实存在,已于 9/8 之后上线。**
`devs.txt` 是 **2026-09-08 23:25** 的 `service.backends()` 真实返回,那一刻在线
设备只有三台:

```
WK_C180    Origin Wukong 180                 onlineFlag: True
WK_C180_2  Origin Wukong 180-2               onlineFlag: True
PQPUMESH8  SILICON EXTREME-Qiming Optical    onlineFlag: True   (光量子,与本实验无关)
```

`QCLOUD_SETUP.md` §6 另外记过一台离线的 `HanYuan_01`。**那一刻没有任何一处出现过
`WK_C180_3`**,它只存在于我在 §7.1 的功效研究表和 `launch_mixedness.sh` 的
`case` 分支里,来源是待核实的口述指标;所以当时的结论是 §7.1 表格第三行**作废**、
候选后端只有 `WK_C180_2` 与 `WK_C180` 两个。

`--mode preflight` 在 **2026-10-11** 给出了最终裁决,结果是**推翻**:活的
`service.backends()` 返回 `WK_C180_3: true`,而离线的反而是 `HanYuan_01: false`。
正确表述是**该芯片在 9/8 之后上线**,而不是"从未存在"。本方案范围不变(用户指定
只考虑 `WK_C180_2`),但 §7.1 第三行的作废理由要改成"不在本次范围"。详见 §8.8.1。

**(2) 此前引用的"92 电路 ×1000 shots ≈ 100 s"没有档案依据。** 我把**计划过但
被拒绝/丢失**的全量运行当成了**已完成**的运行。真实存活的真机数据只有:

| 档案 | 内容 | 结局 |
|---|---|---|
| `hw_feasibility_numbers.json` | **4 条电路 × 400 shots**(请求 400,返回 366/369/370/371) | 论文引用的全部真机数据 |
| `qcloud_results_trial_batch.json` | trial batch,16×2000,无 QCloudOptions | 近均匀垃圾,只留 1 个点 |
| §6.2 记录的 batch1 | 32 电路 × 4000 shots = 128 000 shots | **执行了但数据未落盘,已丢失** |

论文正文写的也是这个量级:`Feasibility data: 4 circuits × 400 shots (366--371
usable)`,小节标题是 *Feasibility demonstration*。**这正是 PRX 会打回来的地方**,
也是本次实验要解决的核心问题。

两个由档案直接得出、必须写进设计的平台事实:
- **平台少给 shots**:请求 400 → 返回 366–371,即 **91.5%–92.8%**。所有估计量
  必须用返回的 counts 归一(现有代码已如此),且 $N_{eff}$ 要按 ~8% 折减。
- **mid-measure 在 WK_C180 上是死的**:`mid_measure_hits_total = 0` / 735 shots;
  late 变体 $F_s = 0.80$ [0.676, 0.884],弛豫卫星峰 $s_8/s_{12} = 1.745$。

### 8.1 成本模型:已确认,~50 倍的对冲就此关闭

**账户持有人于 2026-10 确认了三件事:**

1. **实时扣费** —— 机时在 job 运行过程中被扣除;
2. **超时则立即无法继续使用** —— 额度耗尽即刻停用账户;
3. **不可超额执行** —— job 不可能跑超自己的额度。

因此原先"两种解释相差 ~50 倍"的对冲**不再存在**:按 shot / 按执行时间计费的那一支
就是真实模型,不再需要 canary job 去区分。取 §6.2 的时间线
1800 s ÷ 128 000 shots ≈ **14 ms/shot**,则

> **300 s ⇒ ~21 400 shots —— 这是整个实验的总量,不是单个 job 的。**

**这个费率与电路长度无关**,理由可核查:14 ms/shot 摊到一条 204 指令的电路上是
~68 µs/指令,约为超导门时间(~30 ns)的 **2000 倍**。也就是说账单由**每 shot 的固定
开销**(读出、复位、控制死时间)主导,而不是由门数主导 —— 所以**缩短电路换不来
shots**,唯一的旋钮是 shots 本身。

**"不可超额执行"把失败模式整个换掉了。** 旧 §8.3 依赖的假设是"超额 job 在提交时被
拒绝、不花钱";确认后的事实是**超额 job 会被中途切断**,而**被切断的 job 什么都不
返回**。在 `chunk = len(gates)`(旧默认:整个 stage 一次提交)下,这意味着

> 一个按 2128 s 设计的 job 投进 300 s 的额度 → 跑到额度耗尽被切断 → **数据全丢,
> 而已扣掉的机时也拿不回来**。零数据 + 零预算,是最坏结局。

所以本次改动的重心不是"选哪个 shots",而是**让"被切断"变成可抢救的部分结果**。
两条机制,都已实现并由自检验证:

1. **分块提交** —— `submit_cloud` 每完成一个 chunk 就把原始 counts 写盘,所以被切断
   只损失**正在飞的那一个 chunk**;
2. **按价值排序** —— 标定电路放在最前,depth-1 按输入态分组,depth-2 放最后,于是
   **每一个 chunk 边界都恰好落在一个"可独立分析的前缀"上**(§8.4 的抢救阶梯)。

`--shots` 的减半阶梯(`SHOTS_FLOOR` 兜底)仍然保留,但在硬额度下它已不是主要机制:
被**拒绝**才走减半,被**切断**走的是抢救阶梯。

### 8.2 Job 结构:四个 stage 都还在,但只有一个跑得动

| Job | 比特 | 电路 | shots/电路 | 总 shots | 成本 | 判定 |
|---|---|---|---|---|---|---|
| **0 preflight** | — | 0 | — | 0 | **0(免费)** | ✅ **已于 2026-10-11 完成**(§8.8):`backends()` 在线状态、`best_qubit_blocks(5)` = `[39,48,49,58,67]`、`ChipInfo` **不再 401**(§16 缺口已补)。**但它推翻了本表的功效前提**:实测 f2q=0.99275 优于引用的 0.9831,r<1 从 1.66σ 降到 **0.60σ** |
| **1 tomo** | 5 | **19** | **1000** | **19 000** | **~266 s** | ✅ **这就是全部计划**,剩 ~34 s 余量 |
| 2 probe | 5 | 8 | 300 | 2 400 | ~34 s | ⏸ **推迟:已冗余**(见下) |
| 3 bench | 9 | 34 | 2000 | 68 000 | ~952 s | ❌ 放不下(3.2 倍超支) |
| 4 ladder | 5 | 4 | 4000 | 16 000 | ~224 s | ❌ 与 tomo 互斥 |

**上表每一行都是"该 stage 单独跑"对比"全部 300 s 额度"—— 它们互斥,不可累加。**
tomo 一项就吃掉 266 s,所以无论跑哪个都只能跑一个。`--mode plan` 会把这张判定连同
算术一起打印,数字是从 builder 现算的,不会与代码漂移。

**为什么砍 probe,而不是砍 tomo 的 depth-2?** 因为 probe 现在**已经冗余**,不只是昂贵。
它原本提供三样东西,而 tomo 自己全都有:

| probe 原本提供 | tomo 是否已自带 |
|---|---|
| 5×5 读出指派矩阵 | ✅ 自带 3 条标定电路(zero/one/zero);`analyze_stage` **优先用本 job 的标定** |
| 实测位序 | ✅ 没有 probe 时,`analyze_stage` 用 `detect_layout_ml` 从**层析电路自身**恢复位序 |
| 平台少给 shots 的比例 | ✅ 每条电路都记录 `shots_used` / `shots_return_fraction` |

唯一的差别是:指纹电路给出**唯一**位序,而层析电路留下一个 **4 重 code-space 并列**。
**自检 (f) 已证明这个并列无害**(`layout_tie_invariance = 0.0`,即所有并列假设给出完全
相同的可观测量)。用 34 s —— 也就是全部剩余余量 —— 去换一个"已被证明无害"的性质,
不划算;何况 depth-2 是 M3 在第二个深度上的检验,也是 M5 的全部信号。

**两种比特数仍然用两个不同的物理块,不混用。** 一次提交只有一个 block,而"5 比特电路
放进 9 比特块后返回的 bitstring 是 5 位还是 9 位"**无法离线验证**;若是 9 位,
5 位置的位序似然和 5×5 指派矩阵会**静默失效**。所以 tomo 走 `best_qubit_blocks(5)`,
bench 走 `best_qubit_blocks(9)`,两个块都记录进 artifact,**不假设它们可比**。代价是
指派矩阵不能跨块复用 —— 这正是 tomo 自带标定电路的原因,而它现在也成了砍掉 probe
的理由。`analyze_stage` 会把本 job 标定与 canary 标定的 `max|Δe01|` 写进
`calibration_source`,所以两次标定是否一致是**被记录的**,不是被假设的(本次没有
canary,该字段只记来源)。

**tomo 的 19 条电路按此顺序提交,顺序本身就是预算设计**(详见 §8.4):

```
 1- 3   cal zero / one / zero            -> 指派矩阵(M4)
 4- 6   |0>_L 的 X,Y,Z                   -> M1(|0>_L) + M3@depth1
 7- 9   |1>_L 的 X,Y,Z                   -> M1 两个输入
10-12   |+>_L 的 X,Y,Z                   -> M1 三个输入,L_Z 系综齐了
13-15   |->_L 的 X,Y,Z                   -> M1 + M2 四个输入全齐
16-18   depth-2 (0,Z),(+,X),(0,X)       -> M3@depth2 + M5 信号
19      depth-2 (+,Z)                    -> M5 第二个零对照
```

按**输入态**分组而不是按**基**分组,是因为 `analyze_tomo` 需要三个基才能算出一个态的
半径(半径是三分量 Bloch 向量的模),需要四个态才能做 M2。按基分组会让每个 chunk 边界
都留下一个"测了一半的态",那些 shots 就白花了。

mid/late A/B 仍然留在 bench 的头两条电路(它需要 9 比特,而 tomo 必须留在 5 比特上,
否则上面那个位宽假设就会污染整个标定),本次不跑。

电路规模是**实测的**(不是估算):

```
probe     8 电路       (5,0,5)x2  (6,0,5)x5  (10,0,5)x1        0 CNOT
tomo     19 电路  d=1: 77-81 指令/36 CNOT ×12   d=2: 149-151/72 ×4   cal: 5,5,10
bench    34 电路      203-205 指令 / 82 CNOT / 9 MEAS              2788 CNOT
ladder    4 电路      293-295 指令 / 144 CNOT / 5 MEAS   (rounds=4)  576 CNOT
(rounds=8 为 581 指令 / 288 CNOT,已验证可构建,默认关闭)
```

### 8.3 shots ↔ 成本 ↔ 精度:唯一还剩的旋钮

tomo 是 19 条电路。CI 按功效研究实测的 $0.028\times\sqrt{8000/S}$ 缩放,
$\sigma(r<1)$ 取厂商预测的 $r\approx0.86$,成本按已确认的 14 ms/shot:

| shots/电路 | 总 shots | 成本 | CI 半宽 | $\sigma(r<1)$ | 判定 |
|---|---|---|---|---|---|
| 8000 | 152 000 | 2128 s | 0.028 | 5.0 | ❌ 超额 7.1 倍,会被切断 |
| 4000 | 76 000 | 1064 s | 0.040 | 3.5 | ❌ 超额 3.5 倍 |
| 2000 | 38 000 | 532 s | 0.056 | 2.5 | ❌ 超额 1.8 倍 |
| 1500 | 28 500 | 399 s | 0.065 | 2.2 | ❌ 超额 1.3 倍 |
| **1000** | **19 000** | **266 s** | **0.079** | **1.8** | ✅ **默认值,余 34 s** |
| 750 | 14 250 | 200 s | 0.091 | 1.5 | ✅ 更保守,余 100 s |
| 500 | 9 500 | 133 s | 0.112 | 1.2 | ✅ 很保守,精度已接近无用 |

**这张表必须诚实读**:1000 shots 下 $r<1$ 只是 **~1.8σ**,不是 8000 shots 时的 5σ。
同理 M3 对"各向同性替代模型"的检验,功效研究预测在 8000 shots 下是 $-3.4\sigma$(d=1)
/$-3.9\sigma$(d=2),在 1000 shots 下衰减到约 $-1.2\sigma$/$-1.4\sigma$ —— 那是
**一个趋势,不是一次反驳**。M2 的精度从 ±0.023 放宽到 ±0.060(项跨度 ~1.43 的 4.2%)。

**那这 266 s 还值不值得花?值得,但理由不是显著性,而是替换掉什么。** 论文现有的
全部真机数据是 *4 电路 × 400 shots*、小节标题 *Feasibility demonstration*。本次运行
把它换成 **19 电路 × 1000 shots 的完整层析**:四个逻辑输入的 $w$、Bloch 半径、purity,
带 bootstrap 置信区间,自带读出标定与位序判定。这是一个**真实测量**,只是误差棒诚实地
反映了 300 s 的额度。分析代码不会因为这批数据显著性不够而把子集说成完整结论(§8.4)。

> **要不要更保守?** 若担心 14 ms/shot 低估(它是从 1800 s ÷ 128 000 shots 反推的,
> 分母是"计划过"的 shots 数,不是实测账单),可用 `SHOTS=750`(200 s,余 100 s)。
> 代价是 CI 半宽从 0.079 涨到 0.091。**但更稳妥的做法不是压低 shots,而是依赖 §8.4
> 的分块抢救**:即使 1000 shots 的估算偏乐观、job 在第 6 个 chunk 被切断,前 5 个
> chunk(15 条电路,M1+M2 完整)也已经落盘。

### 8.4 抢救阶梯(取代原先"由 Job 1 墙钟决定的放弃规则")

原先的放弃规则以"用 Job 1 的实测墙钟反推成本模型"为前提。**成本模型已确认,那个前提
消失了**;而"不可超额执行"意味着平台自己就会执行上限,不需要人去猜墙钟。真正要解决的
问题变成了:**被切断时,已经花掉的机时能换回多少科学?**

答案由两条已实现的机制给出。`STAGE_CHUNK['tomo'] = 3`,`submit_cloud` 每完成一个 chunk
就把原始 counts 写盘,而电路顺序(§8.2)让**每个 chunk 边界都落在一个可独立分析的前缀上**:

| 完成到 | 已扣机时 | 完整输入态 | 能写进论文的 | 不能写的 |
|---|---|---|---|---|
| **3/19** | ~42 s | — | 5×5 读出指派矩阵(与 `ChipInfo` 厂商标定互为对照,§8.8.1)+ 平台少给 shots 的复现 | 任何混入度的话;`analyze_stage` 会明确报 `NOT ANALYSABLE` 而不是静默出错 |
| **6/19** | ~84 s | `\|0>` | **M1 对 \|0>_L**(实测 $w$、$r$、purity + bootstrap CI)+ **M3 @ depth 1** | M2、深度趋势 |
| **9/19** | ~126 s | `\|0>,\|1>` | M1 两个输入 | M2(需四个态的 Z 电路)、深度趋势 |
| **12/19** | ~168 s | `\|0>,\|1>,\|+>` | M1 三个输入,$L_Z$ 系综齐了 | M2 的 $L_Z=L_X$ 对比、深度趋势 |
| **15/19** | ~210 s | **四个全齐** | **M1 完整 + M2 完整**(线性分数系综等式) | 深度趋势(M5) |
| **18/19** | ~252 s | 四个全齐 | 上述全部 + **M3 @ depth 2** + M5 两个信号方向 | 第二个零对照 |
| **19/19** | ~266 s | 四个全齐 | **全部:M1 M2 M3 M4 M5** | — |

**读这张表的方式**:它不是"失败模式的清单",而是**这个 job 在任意时刻被切断都能交出
的东西**。被切断最多损失**正在飞的那一个 chunk**(3 条电路 ≈ 42 s),已落盘的部分照常分析。

配套的分析端保证(全部已实现并实测):

- `--mode analyze` **接受截断的 dump**:按提交顺序取前缀,打印缺了哪些电路,并把
  `truncated_stages` 写进 artifact;dump 比 stage 定义**更长**则直接报错(说明它不是
  这一版 builder 产出的,不能拿错的 meta 去分析)。
- `analyze_tomo` **只对三个基都齐的输入态报半径**。半径是三分量 Bloch 向量的模,少一个基
  就没有半径 —— 报一个出来等于用同一个名字指代另一个量。
- **M2 缺任一输入就报 `UNAVAILABLE` 并说明原因**,绝不在子集上算一个看起来像通过的
  `diff`(系综等式是"两个系综相等"的陈述,单边版本会伪装成通过)。
- **M5 只有 depth 1 时明确报 `DEPTH CURVE UNAVAILABLE`**,并把实测到的深度列表打出来。
- `make_figure` 画**实际测到的**输入态,标题标注"n of 4 inputs missing",不崩溃也不假装。

`--mode cloud --stage all` 仍然**硬拒绝**(exit 2),但理由换了:不是"放弃规则需要人
看着墙钟决定",而是**一份额度四个 stage,无人值守连跑会把额度记在第一个 job 上、然后
在账户已停用后损失其余三个**。

`--shots` 的减半阶梯保留,只在 job 被**拒绝**(而非被切断)时触发,兜底 `SHOTS_FLOOR`。

### 8.5 明确不追求的东西

- **over-read $\delta^T C\delta$ 的真机数值**:§7.3 的算术给出需要 ~$5\times10^7$ shots。
- **"4.4% 优于 identity"的显著性宣称**:在可达 shots 下最多 ~2σ。论文应继续按量级一致性
  陈述,而不是按显著性。
- **协议基准的跨芯片复现**(bench)与**深度标度曲线**(ladder):本次额度放不下,
  代码保留,充值后再跑。
- **任何"用 1000 shots 的数据宣称 5σ 结论"的写法**。§8.3 的表就是这次运行误差棒的
  上限,论文必须按实测的 CI 陈述。
### 8.6 为执行本方案改的代码(全部离线、零机时,已实施并验证)

**2026-10-10 的改动(仍然有效):**

| # | 改动 | 说明 |
|---|---|---|
| 1 | `STAGE_SHOTS['probe']=1500→300`,`['bench']=1000→2000`,新增 `['ladder']=4000` | canary 当时要小到只花 ~11% 预算 |
| 2 | `tomo` 阶段追加 3 条标定电路(16→19 条) | 换芯片/换块后仍自洽 —— **现在这也是砍掉 probe 的依据** |
| 3 | `analyze_tomo` 过滤 `kind!='tomo'`;`analyze_stage` 优先用本 job 标定并把与 canary 的 `max\|Δe01\|` 写进 `calibration_source` | 两次标定是否一致是**被记录的**,不是被假设的 |
| 4 | 新增 `ladder` 阶段:`DEPTH_SCAN` @ `LADDER_ROUNDS=(4,)`,`--rounds 4,8` 可加深 | `analyze_tomo` 的深度循环已泛化为"meta 里出现的所有深度" |
| 5 | `--mode analyze --stage all` 把 ladder 并入 tomo;缺 dump 的阶段跳过而非报错 | ladder 单独分析会被断言拒绝(没有 d=1 的信号电路),这是有意的 |
| 6 | `--mode cloud --stage all` **硬拒绝**(exit 2) | 理由已更新,见 §8.4 末尾 |
| 7 | 新增 `--mode plan`,并把表写进 artifact 的 `meta.plan` | 电路数/指令数/CNOT 数从 builder 现算,不会与代码漂移 |
| 8 | `launch_mixedness.sh`:stage 必填(无 `all`)、支持 `BLOCK=`/`SHOTS=`/`ROUNDS=` | 固定 block 才能跨 chunk 复用标定 |
| 9 | 自检 (h) 扩到 4 阶段,新增 (i) 断言 tomo 自带标定、ladder 深度 >2、合并深度 = [1,2,4]、`\|0>,Z)` 在每个深度都存在 | 缺最后一条时 M3 的拟合会**静默跳过**那个深度 |

**2026-10-11 为"已确认的硬额度"新增的改动:**

| # | 改动 | 为什么 |
|---|---|---|
| 10 | `STAGE_SHOTS['tomo'] = 8000 → 1000` | 266 s ≤ 300 s。8000 shots 的 job 会被切断并**全部丢失** |
| 11 | 新增 `STAGE_CHUNK = {'probe':8,'tomo':3,'bench':4,'ladder':4}`;`run_stage` 用它作默认 `chunk`(旧默认是整个 stage 一次提交) | 旧默认下被切断 = **整个 stage 全丢 + 机时已扣**。分块后只丢正在飞的那一个 chunk |
| 12 | `stage_circuits('tomo')` **重排**:3 条标定移到最前,depth-1 按 `TOMO_STATE_PRIORITY=('0','1','+','-')` 分组,depth-2 放最后 | 标定在最后 = 被切断时拿到"有数据、无标定",最坏组合;按**基**分组会在每个边界留下一个测了一半的态 |
| 13 | 新增自检 **(j)**:断言标定占满第一个 chunk、任何 chunk 边界都不留半测态、depth-1 全部早于任何 depth-2、19 条时四态齐 | 顺序现在是**承重结构**,不能让未来的重排静默毁掉抢救路径 |
| 14 | `--mode analyze` **接受截断 dump**(按提交顺序取前缀),打印缺哪些电路并写 `truncated_stages`;dump 比定义**更长**则报错 | 截断现在是**预期产物**,不是异常。更长说明它不是这版 builder 产出的,不能拿错的 meta 分析 |
| 15 | `analyze_tomo` 按 `states_ok` 计算;`mixedness` 记 `inputs`/`inputs_missing`/`n_inputs`;M2 缺态则 `available=False` 并说明;M3 缺 \|0> 的三基则记录原因;M5 只有 depth 1 则报 `DEPTH CURVE UNAVAILABLE` | 不把子集伪装成完整结论。半径是三分量向量的模,少一个基就**没有**半径 |
| 16 | `_tomo_report` 打印 "over n of 4 inputs (MISSING: ...)";M2/M5 不可用时打印原因 | 报告和 artifact 必须说同一件事 |
| 17 | `make_figure` 画实际测到的态、标题标注缺失数、M2 不可用时改画文字;`per_input` 缺失时跳过该面板 | 图与数必须来自同一份(可能不完整的)artifact |
| 18 | `make_figure` 的 `yerr` **钳到 ≥0** | 截断测试暴露的**既有缺陷**:CI 退化到零宽时浮点噪声会让 yerr 变负,matplotlib 直接报错。真机数据 CI 非零所以从未触发 |
| 19 | `analyze_stage`:只拿到标定电路时记 `NOT ANALYSABLE` 并**仍然输出指派矩阵**,而不是抛出 `detect_layout_ml` 的"120 个并列假设" | 那条错误读起来像 builder 的 bug,实情是"这批数据里根本没有层析电路";而指派矩阵本身仍是有价值的测量 |
| 20 | 新增 `BUDGET_S = 300.0` 单一常量,`print_plan` 默认取它 | 所有成本陈述从同一个数推导,不会与预算漂移 |
| 21 | `print_plan` 重写:已确认的成本模型、可负担性判定(并说明各 stage **互斥、不可累加**)、抢救阶梯;`launch_mixedness.sh` 与模块 docstring 同步 | 计划文本不能再引用已作废的两模型对冲和 canary 逻辑 |
| 22 | 删除 `WK_C180_3`(devs.txt 证实不存在);`WK_C180` 仅作为后端选择依据保留,不提交 | 用户指定只考虑 `WK_C180_2`。**⚠️ 理由已被推翻**:`devs.txt` 是 2026-09-08 的快照,2026-10-11 的活 `backends()` 返回 `WK_C180_3: true`,该芯片是之后上线的。删除本身仍对(不在范围),但依据要改成"不在本次范围",见 §8.8.1 |
| 23 | `try_vendor_calibration(chip_id, block=None, block_source=None)`,新增 `qubits_calibrated`/`qubit_id_range`/`block_matches_chip` | **既有缺陷**:它用模块级 `BLOCK`(2026-09 WK_C180 run 的 q147–176)去问 WK_C180_2(只公布 q39–86),返回的逐比特字段**全 null**,与"厂商不公布数据"无法区分。见 §8.8.2 |
| 24 | `preflight()` 把当前芯片的**活块**传进标定查询,并新增 `measured_block_calibration()`:免费取逐比特标定、检查连通性、聚合 `f1q/f2q/fread`,用实测值与引用表值各跑一次 `vendor_prediction`,把两侧 radius / n_eff / CI / σ 写进 artifact | 块必须来自被查询的那台芯片;且**每次 preflight 都从活设备重算**,重新标定会体现为数字变化而不是留下一条过时预测。见 §8.8.2–8.8.3 |
| 25 | 新增实测标定常量(`VENDOR_BLOCK_WK_C180_2`/`MEASURED_F1Q`/`MEASURED_F2Q`/`MEASURED_FREAD`/`MEASURED_CALIB_UTC`/`CI_HALF_WIDTH_AT_1000`)与 `BLOCK_SELECTION_RULE`;**selftest (k)** 锁死方向与禁令 | 实测 f2q=0.99275 优于引用的 0.9831,radius 0.8550→0.9478,r<1 从 1.66σ 降到 **0.60σ**;而按显著性排序 96 个块会把厂商推荐块排到 **96/96**、把最差的块排到 6.29σ。两个方向都必须锁住,见 §8.8.3–8.8.4 |
| 26 | `MS_PER_SHOT` 从 `print_plan` 局部变量提升为模块常量;`--mode plan` 的 M1/M2 段改为**实时重算**而非硬编码 1.8σ;删除 `selftests()` 里重复的 `return res` 死代码 | 费率若只存在于一个函数内,另一个函数就无法用它核预算;硬编码的 σ 会与实测标定漂移。重复 `return` 是此前编辑遗留 |

**实施过程中被否决的设计**(仍然成立):原本想让 canary 在 9 比特块上同时跑 5 比特标定和
9 比特 A/B,以省一次排队。否决理由是"5 比特电路放进 9 比特块后返回几位"**无法离线验证**,
而一旦是 9 位,位序似然和 5×5 指派矩阵会静默失效 —— §7.4 记录的第 1 号陷阱的翻版。

**本次新否决的一个诱人方案**:把 shots 压到 750 以求"绝对安全"。否决理由是它把风险
错置了 —— 真正的不确定性是 14 ms/shot 这个费率(它从"计划过的 shots 数"反推,不是实测
账单),而 #11 的分块抢救已经把"费率偏乐观"的后果从**全丢**降为**少几个 chunk**。压低
shots 只是把确定的精度损失换掉不确定的风险,不划算。

零机时验证(2026-10-11 全部实跑通过):

```
--mode verify                     exit 0,自检 (a)-(j) 全过
run_selftests.py                  exit 0,15 passed / 0 failed
run_selftests.py hwmixedness      exit 0,1 passed(含新的 (j))
--mode plan                       exit 0,可负担性判定与抢救阶梯同代码一致
--mode sample                     exit 0,位序 margin 124339.6
--mode analyze --stage all        exit 0,优雅跳过缺失的 dump
--mode cloud --stage all          exit 2(按设计拒绝)
截断测试 k=3,6,9,12,15,18,19      全部 exit 0(见下表)
paper/audit_numbers.py            exit 0,ALL CHECKS PASSED
paper/make_ed.py                  extended_data.tex 逐字节一致(论文未被动过)
```

截断测试:用 CPUQVM 无噪声 counts 造出**每个 chunk 边界**的前缀,走真实的
`--mode analyze` 路径(含无 probe 时的位序自恢复),验证降级是渐进且响亮的:

| k | exit | 完整输入 | M2 | depths | 拿到的结论 |
|---|---|---|---|---|---|
| 3 | 0 | — | — | — | 记 `NOT ANALYSABLE`,仍输出指派矩阵 |
| 6 | 0 | 1 | False | [1] | **M1(\|0>_L)+ M3@d1** |
| 9 | 0 | 2 | False | [1] | M1 两输入 |
| 12 | 0 | 3 | False | [1] | M1 三输入 |
| 15 | 0 | **4** | **True** | [1] | **M1 + M2 完整**,M5 明确报不可用 |
| 18 | 0 | 4 | True | [1,2] | + M3@d2、M5 两个信号方向 |
| 19 | 0 | 4 | True | [1,2] | 全 stage |

### 8.7 拿到 API Key 之后的执行顺序

```bash
cd /home/yqc/github/QEC
printf '%s' "$KEY" > .originq_token      # 现存 96 字节那个已失效,全部 Unauthorized

# ---- 第 0 步:全部免费,不花一秒机时 ----
./qenv/bin/python hw_mixedness.py --mode plan     # 预算表 + 权衡表 + 抢救阶梯
./qenv/bin/python hw_mixedness.py --mode verify
./qenv/bin/python run_selftests.py hwmixedness
./qenv/bin/python hw_mixedness.py --mode sample
./qenv/bin/python hw_mixedness.py --mode simulate --seeds 6 --shots 1000 \
    --vendor 0.9993,0.9831,0.9612 --json /tmp/power1000.json
./qenv/bin/python hw_mixedness.py --mode preflight \
    --chip-ids WK_C180_2 --vendor 0.9993,0.9831,0.9612
#  从 preflight 输出抄下 B5 = best_qubit_blocks(5);同时确认 WK_C180_2 在线。
#  【2026-10-11 已执行,见 §8.8】结果:B5 = 39,48,49,58,67;WK_C180_2 在线;
#  ChipInfo 不再 401(§16 的缺口已补上),但它公布的 f2q=0.99275 优于引用的
#  0.9831,使 r<1 的显著性从 1.66σ 降到 0.60σ —— 这一步是免费的,却推翻了
#  下面第 1 步的预期收益,所以必须在花钱之前做。

# ---- 第 1 步:唯一要花钱的 job,~266 s / ~300 s ----
CHIP=WK_C180_2 BLOCK="$B5" bash launch_mixedness.sh tomo
tail -f qcloud_mixedness_tomo.log
#  日志会逐 chunk 打印 "batch k: received",每收到一个 chunk 数据就已落盘。
#  被切断是预期结果之一,不是失败:已落盘的部分照常分析。

# ---- 第 2 步:离线分析,不需要 token,任何时候都能重跑 ----
./qenv/bin/python hw_mixedness.py --mode analyze --stage all
#  它会自己报告完整还是截断、缺哪些输入和深度,并把 truncated_stages 写进
#  artifact。不要靠人去数日志。
```

**决策点只剩一个,在第 1 步之前**:`SHOTS` 用默认 **1000**(266 s,余 34 s,CI 半宽
0.079),还是 **750**(200 s,余 100 s,CI 半宽 0.091)。我倾向 1000 —— 因为 §8.4 的分块
抢救已经把"14 ms/shot 估低了"的后果从**全丢**降为**少几个 chunk**,而压低 shots 是拿
确定的精度损失去换那个已被兜住的风险。

**充值之后**(如果有的话)再按 `probe → bench → ladder` 跑,每个都是独立 job;
`--mode analyze --stage all` 会自动把 ladder 的 counts 并入 tomo,得到 d=1,2,4 的
同一条深度曲线。

**在你给出 Key 之前,我不会执行第 1 步及之后的任何命令。**

---

### 8.8 2026-10-11 免费 preflight 的实测结果:额度保留,未提交任何 job

Key 于 2026-10-11 08:52 写入 `.originq_token`(96 字节),`SHOTS` 确认为默认 1000。
随后执行的**只有免费查询**:`--mode cloud` 一次都没有运行,**没有提交任何付费 job,
300 s 额度完整保留**。

#### 8.8.1 preflight 通过,并推翻了两条已归档的陈述

```
./qenv/bin/python hw_mixedness.py --mode preflight \
    --chip-ids WK_C180_2 --vendor 0.9993,0.9831,0.9612
# rc=0,errors: []     (上一次 2026-10-10T13:29:57Z 是 7 条 Unauthorized)
```

| 查询 | 结果 |
|---|---|
| `service.backends()` | `HanYuan_01: false`,`PQPUMESH8/WK_C180/WK_C180_2/WK_C180_3: true`,`full/partial/single_amplitude: true` |
| `best_qubit_blocks(5)` | **`[39, 48, 49, 58, 67]`** ← 唯一候选,即 `B5` |
| `best_qubit_blocks(9)` | `[39, 48, 49, 57, 58, 66, 67, 75, 85]`(含 `B5` 全部 5 个) |
| `chip_info()` | **不再 401**:`qubits_num=180`,1Q 30 ns / 2Q 40 ns,基础门 `['RPhi','CZ']`,17 个比特有标定(q39–86),20 条边 |

**纠正 (a):`WK_C180_3` 存在。** §8.0(1) 与 §8.6 第 22 条写的"devs.txt 证实不存在"
只对 `devs.txt` 那一刻成立 —— 那是 **2026-09-08 23:25** 的 `service.backends()` 快照,
里面确实只有 `WK_C180`/`WK_C180_2`/`PQPUMESH8`。活的 API 在 2026-10-11 返回
`WK_C180_3: true`,所以正确表述是**该芯片在 9/8 之后上线**,而非"从未存在";离线的
是 `HanYuan_01`。本方案范围不变(用户指定只考虑 `WK_C180_2`),但 §7.1 功效研究表
第三行的作废理由应改为"不在本次范围",而不是"设备不存在"。

**纠正 (b):`ChipInfo` 已不再 401。** §7 的 M4 行、§8.3 表格、§8.6 第 16 条、§8.7
第 2 步中"填补 401 缺口""仍 401 就照实记录"等措辞全部过时:逐比特标定**是可取的**。

#### 8.8.2 标定先前返回全 null,原因是跨芯片串用了块

第一次 preflight 的 `calibration` 里 9 个比特的 `block_qubits` **全是 null**、
`block_edges` 为空,看起来像"厂商不公布数据"。真实原因是
`hw_error_budget.try_vendor_calibration()` 用了模块级常量
`BLOCK = [147,155,157,164,165,166,174,175,176]` —— 那是 **2026-09 WK_C180 run**
钉住的块,而 `WK_C180_2` 只公布 q39–86。已修:

- `try_vendor_calibration(chip_id, block=None, block_source=None)`,并新增
  `qubits_calibrated` / `qubit_id_range` / `block_matches_chip`;整块全 null 时显式
  写明"ChipInfo 有响应但没有这个块的标定,它覆盖 q39–86",不再留一个与"厂商无数据"
  无法区分的 null。
- `preflight()` 把**当前芯片的活块**传进去(`block_source` 记为
  `live best_qubit_blocks(9)`)。修复后 `block_matches_chip: True`,返回真实逐比特
  数据与 9 条边。
- 新增 `measured_block_calibration(chip, block, token)`:免费取该块逐比特标定,
  检查**连通性**,聚合出 `f1q/f2q/fread`,并同时用实测值与引用表值各跑一次
  `vendor_prediction`,把 radius、`fread⁴` 存活率、n_eff、CI 半宽和两侧的 σ 一并
  写入 artifact 的 `measured_block` 字段。**每次 preflight 都从活设备重算**,
  所以重新标定会体现为数字变化,而不是留下一条过时的预测。

#### 8.8.3 决定性发现:设备比标称更好,所以 headline 更弱

`B5 = [39,48,49,58,67]` 的实测标定(取自 `hw_mixedness_preflight.json` 的
`measured_block.WK_C180_2`):

| q | readout | 1Q gate | T1 | T2 |
|---|---|---|---|---|
| 39 | 0.96870 | 0.99860 | 32.2 µs | 2.1 µs |
| 48 | 0.96610 | 0.99940 | 36.5 | 6.2 |
| 49 | 0.94730 | 0.99930 | 33.8 | 3.0 |
| 58 | **0.92280** | 0.99950 | 41.6 | 8.9 |
| 67 | 0.94990 | 0.99930 | 48.2 | 5.3 |

CZ 边:39–48 (0.9922)、39–49 (0.9936)、49–58 (0.9875)、58–67 (0.9977) —— 5 个点
4 条边是一棵树,**块连通**(`connected: True`),可直接提交。聚合得
`f1q=0.99922, f2q=0.99275, fread=0.95096`。

**`f2q` 实测 0.99275 优于计划全程引用的表值 0.9831。** 门更好 ⇒ 态更纯 ⇒ 信号更小:

| | 引用表值 | **实测(该块)** |
|---|---|---|
| f2q | 0.9831 | **0.99275** |
| 36 个 CNOT 的存活率 | 0.5414 | **0.7695** |
| 预测 radius | 0.8550 | **0.9478** |
| 信号 1−r | 0.1450 | **0.0522**(小 2.8 倍) |
| post-selection 后 n_eff @1000 shots | — | **818**(fread⁴=0.818) |
| CI 半宽 | 0.079 | **0.0874** |
| **r<1 的显著性** | 1.66σ | **0.60σ** |

> 注:此前文档写的 1.8σ 是 `(1−0.855)/0.079`,**未**按 `fread⁴` 折减 n_eff;
> 折减后同一组引用表值给 1.66σ。两个数都来自引用表值,都与实测的 0.60σ 不同。

**加 shots 救不回来**,因为分子 1−r 本身就小:

| shots/电路 | 总 shots | 墙钟 | CI | σ |
|---|---|---|---|---|
| 1000 | 19 000 | 267 s ✅ | 0.0874 | **0.60** |
| 2000 | 38 000 | 534 s | 0.0618 | 0.84 |
| 8000 | 152 000 | 2138 s | 0.0309 | 1.69 |
| 70 118 | 1 332 242 | **18 651 s** | 0.0104 | **5.0** |

**5σ 需要 18 651 s ≈ 5.2 小时 = 预算的 62 倍。** 对这个主张而言,**约束不是 300 s,
而是"这台机器最好的块太好了"**。因此 §7–§8 中所有由引用表值推出的功效数字都要下调:
1.8σ → 0.60σ;**M3 尤其危险** —— 设备越接近各向同性去极化,替代模型越可能
**拟合得好**,那一项可能完全失去区分力(原预测 −3.4σ 不再成立)。
**M2 不受影响**:它是零检验,精度只取决于 shots。

#### 8.8.4 一个明确拒绝的做法:按显著性挑块

把 96 个连通 5 比特块按 `(1−r)/σ` 排序,结果是**反的**:

| 块 | f2q | radius | σ |
|---|---|---|---|
| `[40,49,50,58,68]`(**最差**) | 0.9579 | 0.4479 | **6.29** |
| `[40,49,50,59,68]` | 0.9590 | 0.4679 | 6.20 |
| … | | | |
| `[39,48,49,58,67]`(**厂商推荐**) | **0.99275** | **0.9478** | **0.60(排 96/96)** |

**挑校准最差的比特就能把 r<1 做到 6.3σ。** 这是选择仪器去让假设检验通过,不是测量,
也经不起审稿人一句"为什么是这个块"。唯一站得住的判据是**在看任何结果之前就固定
下来**的,厂商自己的 `best_qubit_blocks()` 正是这样的判据。已写入代码:

- `BLOCK_SELECTION_RULE` 常量,以及实测标定常量 `VENDOR_BLOCK_WK_C180_2` /
  `MEASURED_F1Q` / `MEASURED_F2Q` / `MEASURED_FREAD` / `MEASURED_CALIB_UTC` /
  `CI_HALF_WIDTH_AT_1000`(每个都带来源与推导注释)。
- **selftest (k)** 断言:实测 radius > 引用 radius、实测 σ < 引用 σ、σ<1、
  陷阱块 radius 更低而 σ>5、5σ 墙钟 > 50× 预算、电路数 == 19。**方向与禁令一起
  锁死**,任何一侧被悄悄改回去都会失败。(k) 第一次运行就抓到了它自己的一个 bug:
  `len(stage_circuits('tomo'))` 是 **2**(该函数返回 `(gates, meta)` 二元组)而不是
  19,使 5σ 墙钟被算成 1963 s;已改用 (j) 中已绑定的 `tm` 并加断言。
- `--mode plan` 不再硬编码 1.8σ,改为从这些常量**实时重算**并打印上表。
- `MS_PER_SHOT` 从 `print_plan` 的局部变量提升为模块常量,否则 (k) 无法用同一个
  费率去核预算。

#### 8.8.5 决定:保留额度

用户裁定**先不花**。300 s 额度完整保留,未提交任何 job。理由:这次花费是
267 s / 300 s(额度的 **89%**)且不可逆,而它要支撑的 headline 已从 1.66σ 掉到
**0.60σ**。

**恢复运行时**(额度充值,或论文主张重新界定之后)顺序不变,`B5` 已知:

```bash
CHIP=WK_C180_2 BLOCK=39,48,49,58,67 bash launch_mixedness.sh tomo
```

**即使主张重新界定,这次运行仍买得到**(都不依赖 r<1):完整 19 电路层析(替换论文
现有的 4 电路 ×400 shots feasibility demo)、M2 零检验(±0.060 对项跨度 1.43)、
5×5 实测读出指派矩阵、平台少给 shots 的复现,以及"该芯片最好块的 f2q 比引用表值高
约 1%"这个可直接修正误差预算的结果。
