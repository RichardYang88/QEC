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
