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

**`180-3` 一列作废**:`devs.txt`(即 `service.backends()` 的实测返回)里只有
`WK_C180`、`WK_C180_2` 和光量子的 `PQPUMESH8`,没有 `WK_C180_3`,详见 §8.0。
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
1. **芯片 ID 待确认**。仓库里只出现过 `WK_C180`、`WK_C180_2`、`PQPUMESH8`,
   `WK_C180_3` 从未出现;提交前先用免费的 `--mode preflight` 核实。
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
| M4 | 自测 5 比特读出指派矩阵 | 两条标定电路 + 重复条夹出漂移 | 填补 `ChipInfo` 返回 401 的记录缺口 |
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
#      以及再一次尝试 ChipInfo(成功就写厂商逐比特数据,仍 401 就照实记录)

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
   (chip, depth)",Data Availability 补上自测标定;若 `preflight` 这次拿到了厂商数据,
   §16 与 ED Table 9 里"401 被拒"的表述相应升级(API 被拒、控制台聚合值已记录)。
4. 不动 `mixed_input_boundary.json`(audit §17 已锁字段);同深度的模型预测由
   `hw_mixedness.py` 自己算,只写新 artifact。

## 八、300 秒机时的完整执行方案(2026-10-10 定稿,待审)

§7 是**科学设计**,本节是**在 300 s 机时下如何把它执行完**。两者的电路相同,
但 job 的切分、shots 的取值和放弃规则全部由本节的成本分析决定。

### 8.0 先纠正两处我此前说过、但档案不支持的话

**(1) `WK_C180_3` 不存在。** `devs.txt` 是 `service.backends()` 的真实返回,
在线设备只有三台:

```
WK_C180    Origin Wukong 180                 onlineFlag: True
WK_C180_2  Origin Wukong 180-2               onlineFlag: True
PQPUMESH8  SILICON EXTREME-Qiming Optical    onlineFlag: True   (光量子,与本实验无关)
```

`QCLOUD_SETUP.md` §6 另外记过一台离线的 `HanYuan_01`。**没有任何一处出现过
`WK_C180_3`**,它只存在于我在 §7.1 的功效研究表和 `launch_mixedness.sh` 的
`case` 分支里,来源是待核实的口述指标。所以 §7.1 表格的第三行**作废**,
候选后端只有 `WK_C180_2` 与 `WK_C180` 两个。`--mode preflight` 会在一秒内、
零机时地给出最终裁决;若它确实存在(例如控制台可见但 API 未列),该行才恢复有效。

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

### 8.1 成本模型:两种解释相差 ~50 倍,所以第一个 job 的唯一任务是区分它们

档案里**没有任何计时记录**(`hw_error_budget.json` 的 `wall_s: 1.02` /
`qpu_time_spent: 0` 都是离线分析)。唯一的约束来自 §6.2 的时间线:约
**162 000 shots 执行完之后机时耗尽**,而当时的预算秒数没有被记录下来。于是:

| 假设 | 300 s 能买多少 shots | 依据 |
|---|---|---|
| **悲观:按 shot/执行时间计** | **~21 000 shots** | 1800 s ÷ 128 000 shots ≈ 14 ms/shot |
| **乐观:按 job 墙钟(含排队)计** | **shots 近乎免费,job 数才是预算** | 128 000 shots × 204 指令 × ~30 ns ≈ **0.8 s** 执行时间,不可能占满 1800 s;且 launcher 在 1800 s 后 kill 掉的两个 job"仍然消耗了机时"——那是排队墙钟 |

两者相差约 **50 倍**,无法从档案判定。§6 还记着 `WK_C180_2`「排队任务多」、
`WK_C180`「队列较短」,而 `backends()` **不暴露队列深度**,所以排队时间也无法免费预知。

**由此得到三条不可让步的设计原则:**
1. **第一个 job 必须小到在悲观模型下也只花 ~10% 预算**,但又要大到能独立发表。
2. **一个 job = 一次排队**。乐观模型下 job 数就是预算,所以 job 总数硬上限 **4**。
3. **被拒绝不花钱**(`QPU time is insufficient` 在提交时就被挡下),所以 shots
   应**从乐观值起跳,靠内置的减半阶梯往下走**,而不是从保守值往上试。

### 8.2 Job 结构

**两种比特数用两个不同的物理块,不混用。** 一次提交只有一个 block,而"5 比特电路
放进 9 比特块后返回的 bitstring 是 5 位还是 9 位"**无法离线验证**;若是 9 位,
5 位置的位序似然和 5×5 指派矩阵会**静默失效**。所以 Job 1/2/4 走
`best_qubit_blocks(5)`,Job 3 走 `best_qubit_blocks(9)`,两个块都记录进 artifact,
**不假设它们可比**。代价是指派矩阵不能跨块复用——因此 Job 2 自带标定电路。

| Job | 比特 | 电路 | shots/电路 | 总 shots | 悲观成本 | 内容 |
|---|---|---|---|---|---|---|
| **0 preflight** | — | 0 | — | 0 | **0(免费)** | `backends()` 在线状态;`best_qubit_blocks(5)` 与 `(9)`;`ChipInfo` 再试一次(成功就补上 §16 的 401 缺口);裁决 `WK_C180_3` 是否存在 |
| **1 probe** | 5 | **8** | **300** | **2 400** | ~34 s | 3 条读出标定(zero/one/zero 重复)+ 5 条单 X 指纹 |
| **2 tomo** | 5 | **19** | **8000**(阶梯 ↓1000) | 152 000 | 2128 s | 12 条 d=1(4 输入 × 3 基)+ 4 条 d=2(含 2 个零对照)+ **3 条自带标定** |
| **3 bench** | 9 | **34** | **2000** | 68 000 | 952 s | **2 条 mid/late A/B(电路 1–2)** + 32 条全分支(p=0,ψ∈{0,+},late) |
| **4 ladder** | 5 | **4** | **4000** | 16 000 | 224 s | DEPTH_SCAN 网格 @ rounds=4;`ROUNDS=4,8` 则 8 条 |
| | | **65** | | **238 400** | **3338 s** | |

**注意最后两列:悲观模型下整个方案要 3338 s,是 300 s 预算的 11 倍。** 这不是笔误,
而是本节的核心结论——方案**不可能在悲观模型下跑完**,它被设计成:要么乐观模型成立
(于是全部跑得完),要么靠拒绝阶梯和 §8.3 的放弃规则**有序地降级到只跑 Job 1+2**。

mid/late A/B 从 Job 1 移到了 Job 3(它就是 bench 的头两条电路):A/B 需要 9 比特,
而 Job 1 必须留在 5 比特上,否则上面那个位宽假设就会污染整个标定。

Job 2 自带 3 条标定电路,是为了**让它在换芯片后仍然自洽**:如果 canary 显示
`WK_C180_2` 排队病态,可以改投 `WK_C180`,Job 2 仍能独立给出指派矩阵,不依赖 Job 1。
`analyze_stage` 优先用本 job 的标定,并把与 canary 的 `max|e01 差|` 写进
`calibration_source`,所以两次标定是否一致是**被记录的**,不是被假设的。

电路规模是**实测的**(不是估算):

```
probe     8 电路       (5,0,5)x2  (6,0,5)x5  (10,0,5)x1        0 CNOT
tomo     19 电路  d=1: 77-81 指令/36 CNOT ×12   d=2: 149-151/72 ×4   cal: 5,5,10
bench    34 电路      203-205 指令 / 82 CNOT / 9 MEAS              2788 CNOT
ladder    4 电路      293-295 指令 / 144 CNOT / 5 MEAS   (rounds=4)  576 CNOT
(rounds=8 为 581 指令 / 288 CNOT,已验证可构建,默认关闭)
```

### 8.3 shots ↔ 成本 ↔ 精度:放弃规则真正依赖的那张表

Job 2 是 19 条电路。CI 按功效研究实测的 $0.028\times\sqrt{8000/S}$ 缩放,
$\sigma(r<1)$ 取厂商预测的 $r\approx0.86$:

| shots/电路 | 总 shots | 悲观成本 | CI 半宽 | $\sigma(r<1)$ |
|---|---|---|---|---|
| 8000 | 152 000 | 2128 s | 0.028 | 5.0 |
| 4000 | 76 000 | 1064 s | 0.040 | 3.5 |
| 2000 | 38 000 | 532 s | 0.056 | 2.5 |
| **1000** | **19 000** | **266 s** | 0.079 | 1.8 | ← **悲观模型下这就是全部剩余预算** |
| 500 | 9 500 | 133 s | 0.112 | 1.25 |

两条结论:
1. **悲观模型下 Job 2 与 Job 3 只能二选一**,且应选 Job 2(新科学)而不是 Job 3
   (2026-09 可行性运行的更好统计版本)。
2. **内置的减半阶梯 8000→4000→2000→1000 本身就是成本模型的测量手段**:拒绝在
   提交时返回、不花钱(2026-09 的记录是 batch1 跑完、batch2 被拒),所以每次拒绝
   都免费地把可负担的 shots 收窄一档。

> **这里依赖一个必须说清的假设**:厂商在 job 成本超过剩余机时时会**拒绝**,而不是
> 接受后超额执行。若该假设不成立,从 8000 起跳可能把 ~2100 s 记在 300 s 的额度上。
> **除非你确认额度是在提交时硬限制的,否则 Job 2 应从 1000 shots 起跳。**

### 8.4 放弃规则(由 Job 1 的实测墙钟决定)

设 `w1` = Job 1 从提交到拿到结果的墙钟秒数(`run_stage` 已记录为 `wall_s`)。

| Job 1 结果 | 判定 | Job 2 | Job 3 | Job 4 |
|---|---|---|---|---|
| **被拒绝**(`QPU time insufficient`) | 预算已空 | 停 | 停 | 停 |
| `w1 ≤ 60 s` | 队列短 | **8000 shots**(拒绝则自动减半至 1000) | 2000 shots | 4000 shots |
| `60 < w1 ≤ 180 s` | 排队显著,job 数就是预算 | **3000 shots** | 仅当 `w1+w2 ≤ 240 s`,1000 shots | 放弃 |
| `w1 > 180 s` | 单 job 就吃掉大半预算 | **1500 shots** | 放弃 | 放弃 |
| Job 2 走完减半阶梯仍全部被拒 | 悲观模型成立且预算极小 | 记录拒绝次数与 shots 值 | 放弃 | 放弃 |

任何情况下**已落盘的数据都不丢**(`submit_cloud` 每 chunk 增量写盘),
`--mode analyze` 完全离线、不需要 token,且 `--stage all` 会自动把 ladder 的
counts 并入 tomo 得到 d=1,2,4 的同一条深度曲线(ladder 单独分析会被断言拒绝,
因为它没有 d=1 的信号电路)。`--mode cloud --stage all` 被**硬拒绝**(exit 2):
放弃规则是人看着上一个 job 的实测墙钟做的决定,不能无人值守连跑四个 job。

### 8.5 各 job 单独能支撑的论文内容(渐进式,不押注在跑完)

| 完成到 | 能写进论文的 | 不能写的 |
|---|---|---|
| **仅 Job 1** | 第二块芯片上的 5×5 读出指派矩阵(填补 `ChipInfo` 401 的记录缺口)+ **mid-measure 异常是否平台级**的判定 + 实测位序 + 平台少给 ~8% shots 的复现 | 任何关于混入度的话 |
| **+ Job 2** | **M1** 实测 $w$、$r$、purity(8000 shots 时 $r$ 的 95% CI 半宽 0.028,即 $r<1$ 是 5σ 结论);**M2** 线性分数系综等式在真机上成立到 ±0.023(项跨度的 2.3%);**M3** 各向同性替代模型的同深度检验;**M4** 高精度指派矩阵;**M5** 2 个零对照 | 协议基准的跨芯片复现 |
| **+ Job 3** | 把论文的 *4 电路 × 400 shots feasibility* 升级为**第二块芯片上 16 分支 × 2000 shots 的完整协议基准**,并给出 mid/late 的定量对比 | 深度标度 |
| **+ Job 4** | d=1,2,4(,8)的**实测逻辑半径衰减曲线** → 每轮逻辑错误率,可与理论的阈值行为对照 | — |

**明确不追求的**:over-read $\delta^T C\delta$ 的真机数值(§7.3 已给出算术:需要
~$5\times10^7$ shots),以及"4.4% 优于 identity"的显著性宣称——在可达 shots 下
它最多是 ~2σ,论文应继续按量级一致性来陈述,而不是按显著性。

### 8.6 为执行本方案改的代码(全部离线、零机时,已实施并验证)

| # | 改动 | 状态 | 说明 |
|---|---|---|---|
| 1 | `STAGE_SHOTS['probe']=1500→300`,`['bench']=1000→2000`,新增 `['ladder']=4000` | ✅ | canary 必须小到悲观模型下只花 ~11% 预算 |
| 2 | `tomo` 阶段追加 3 条标定电路(16→19 条) | ✅ | Job 2 换芯片/换块后仍自洽 |
| 3 | `analyze_tomo` 过滤 `kind!='tomo'`;`analyze_stage` 优先用本 job 标定并把与 canary 的 `max|Δe01|` 写进 `calibration_source` | ✅ | 两次标定是否一致是**被记录的**,不是被假设的 |
| 4 | 新增 `ladder` 阶段:`DEPTH_SCAN` 网格 @ `LADDER_ROUNDS=(4,)`,`--rounds 4,8` 可加深 | ✅ | `analyze_tomo` 的两个 `for rounds in (1,2)` 循环已泛化为"meta 里出现的所有深度",M3 与 M5 自动覆盖 d=4 |
| 5 | `--mode analyze --stage all` 把 ladder 的 counts **并入 tomo** 后一次分析;缺 dump 的阶段跳过而非报错 | ✅ | ladder 单独分析会被断言拒绝(它没有 d=1 的信号电路),这是有意的 |
| 6 | `--mode cloud --stage all` **硬拒绝**(exit 2) | ✅ | 放弃规则是人看着上一个 job 的实测墙钟做的决定,不能无人值守连跑 |
| 7 | 新增 `--mode plan`:打印预算表、shots↔成本↔精度权衡表、放弃规则,并把表写进 artifact 的 `meta.plan` | ✅ | 表里的电路数/指令数/CNOT 数是**从 builder 现算的**,不会与代码漂移 |
| 8 | `launch_mixedness.sh`:stage 改为**必填**(无 `all`)、支持 `BLOCK=`/`SHOTS=`/`ROUNDS=`、`WK_C180_3` 标注为未核实 | ✅ | 固定 block 才能跨 job 复用标定 |
| 9 | 自检 (h) 扩到 4 个阶段,新增 (i) 断言 tomo 自带 3 条标定、ladder 深度 >2、合并深度 = [1,2,4]、且 `(|0>,Z)` 在每个深度都存在 | ✅ | 第 4 条尤其重要:缺了它 M3 的拟合会**静默跳过**那个深度 |
| — | `run_stage` 记录 `wall_s`/`shots_requested`/`shots_used`/`shots_return_fraction`/`block` | 早已存在 | `stage_record` 已经在做,包括那个 ~92% 的返回率 |

**实施过程中被否决的一个设计**:原本想让 Job 1 在 9 比特块上同时跑 5 比特标定和
9 比特 A/B,以省一次排队。否决理由是"5 比特电路放进 9 比特块后返回几位"**无法离线
验证**,而一旦是 9 位,位序似然和 5×5 指派矩阵会静默失效——这正是 §7.4 记录的第 1 号
陷阱的翻版。A/B 因此留在 bench 的头两条电路。

零机时验证(2026-10-10 全部实跑通过):

```
--mode verify                     exit 0,4 个阶段的指令直方图 + r4 理想参考精确
run_selftests.py hwmixedness      exit 0,1 passed(含新的 (h)(i))
--mode sample --shots 3000        exit 0,位序 margin 124339.6,M2 diff -0.0043+-0.0129,
                                  M3 两深度均 agrees,M5 零对照 max 1.1 sigma
--mode simulate --seeds 6 ×2 芯片 exit 0,即 §7.1 表的两列有效数据
--mode plan                       exit 0
--mode cloud --stage all          exit 2(按设计拒绝)
paper/audit_numbers.py            exit 0,ALL CHECKS PASSED
paper/make_ed.py                  extended_data.tex 逐字节一致(论文未被动过)
```

### 8.7 拿到 API Key 之后的执行顺序

```bash
cd /home/yqc/github/QEC
printf '%s' "$KEY" > .originq_token          # 现存 96 字节那个已失效,全部 Unauthorized

# ---- 第 0 步:全部免费,不花一秒机时 ----
./qenv/bin/python hw_mixedness.py --mode plan           # 打印预算表 + 权衡表
./qenv/bin/python hw_mixedness.py --mode preflight \
    --chip-ids WK_C180_2,WK_C180 --vendor 0.9993,0.9831,0.9612
#  从 preflight 的输出里抄下两个块:
#    B5 = best_qubit_blocks(5)   -> probe / tomo / ladder 都用它
#    B9 = best_qubit_blocks(9)   -> bench 用它
#  同时看:WK_C180_3 是否存在、ChipInfo 是否仍然 401(不 401 就顺手补上 §16 的缺口)

# ---- 第 1 步:canary,2400 shots,悲观模型下 ~34 s ----
CHIP=WK_C180_2 BLOCK="$B5" bash launch_mixedness.sh probe
grep -E 'wall|shots used|refus' qcloud_mixedness_probe.log
#  记下 w1 = 那行 "received 8 circuits in NNN s wall" 的 NNN

# ---- 第 2 步:按 §8.4 的表用 w1 决定 Job 2 的 shots,再往下走 ----
#  w1<=60s -> 8000(默认);60<w1<=180 -> SHOTS=3000;w1>180 -> SHOTS=1500
#  若 §8.3 那个"拒绝不花钱"的假设未经你确认,则一律从 SHOTS=1000 起跳
CHIP=WK_C180_2 BLOCK="$B5" bash launch_mixedness.sh tomo
CHIP=WK_C180_2 BLOCK="$B9" SHOTS=2000 bash launch_mixedness.sh bench   # 条件执行
CHIP=WK_C180_2 BLOCK="$B5" bash launch_mixedness.sh ladder             # 条件执行

# ---- 第 3 步:离线分析,不需要 token,任何时候都能重跑 ----
./qenv/bin/python hw_mixedness.py --mode analyze --stage all
#  ladder 的 counts 会自动并入 tomo,得到 d=1,2,4 的同一条深度曲线
```

**在你给出 Key 之前,我不会执行第 1 步及之后的任何命令。**
