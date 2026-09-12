# JOURNAL_CHOICE.md — 目标期刊评估与投稿策略(2026-09)

## 论文现状摘要(决定期刊档位的事实基础)

- **方法**:VSCR = 投影症候测量 + 超网络生成的每分支学习酉恢复(60 参数/分支),
  无监督端到端训练;解码器暖启动(epoch 0 严格等于最优 Pauli 解码器)。
- **理论**:证明 + 数值验证:最优 Pauli 解码器是 Haar 平均分支保真度损失的
  **驻点**(梯度 ≤6e-17,4 类信道)→ 无监督训练"认证"解码器最优性;
  冷启动变分恢复存在不健康分支(最差分支 cf 0.61 vs 0.81)。
- **仿真**:4 信道 × 10 个 p × 200 态,与最优解码器差 <1e-12;3 个 oracle
  误差缓解上界对照;精确电路编译验证 <1e-9(512 维态矢量 vs 密度矩阵)。
- **硬件**:OriginQ 悟空 WK_C180 真机 4 电路 ×400 shots 可行性演示:
  后选择分支保真度 F₁₂=0.80 [0.68,0.88];中间测量失效(0/735);
  辅助比特弛豫卫星峰定量表征。**全量 128 电路基准待购机时**。

## 期刊对比

| 期刊 | 分区/IF 量级 | 首轮周期(典型) | 匹配度评估 |
|---|---|---|---|
| **Nature Communications** | Q1, ~16 | 4–8 周(desk reject 风险中) | 需要"完整故事+实验亮点";当前硬件为可行性级,**若补齐全量基准则匹配**;仅可行性数据时 desk 风险偏高 |
| **PRX Quantum** | Q1, ~20 | 4–6 周 | 量子信息顶刊;重视理论严谨+实验;驻点定理+硬件演示组合对其口味,但实验体量偏小 |
| **npj Quantum Information** | Q1, ~7 | **3–5 周(最快)** | Nature 子刊;对"方法+中小规模硬件演示"接受度最高;**当前数据最稳妥的 Q1 选择** |
| **Quantum Science and Technology** | Q1, ~6 | 4–8 周 | IOP;硬件协议/基准方法论类文章的传统归宿;稳妥备选 |
| **Physical Review Applied** | Q1, ~4 | 3–6 周 | 器件/协议导向;匹配但影响力略低 |
| **Physical Review Letters** | Q1, ~9 | **2–4 周(最快)** | 4+页限制;需"广泛兴趣"单点突破;驻点定理单独或全量基准单点或可,组合故事难塞 |

## 建议策略

1. **现在投(不等机时)**:npj Quantum Information。理由:Q1、Nature 品牌、
   首轮最快之一、对"学习恢复电路+真机可行性+平台表征"的组合接受度高;
   审稿人大概率要求补的正是全量基准——可在修稿期购买机时补做(流程见
   HW_FULL_INTEGRATION.md),把"修稿压力"转化为"机时窗口"。
2. **等全量基准后投**:Nature Communications 或 PRX Quantum。
   全量 128 电路 ×1000 shots 的 HW-vs-ideal 曲线 + F₁₂ 类分支保真度表
   将使实验部分达到 NatComm 门槛;PRX Quantum 则额外欣赏驻点定理。
3. **不建议**:PRL(故事太组合)、PRA(分区/影响力不匹配目标)。

## 投稿包清单(当前已备)

- [x] main.tex(摘要/引言/结果/讨论/方法/图注)+ refs.bib(38 条,Crossref 核实)
- [x] Fig.1 示意图 / Fig.2 训练+分支健康 / Fig.3 基准+LER+相干 / Fig.4 真机可行性
- [x] extended_data.tex(make_ed.py 生成:理想参考表、分支 cf 表、可行性统计、位序证据)
- [x] 数值填充管线(fill_numbers.py ← paper_numbers.json / hw_feasibility_numbers.json)
- [x] 复现脚本链(vscr_paper.py / vscr_paper_coh.py / hw_verify_analysis.py / qcloud_vscr_new.py)
- [x] 期刊评估与策略(本文件)
- [ ] 全量真机数据(待机时;接入流程 HW_FULL_INTEGRATION.md)
- [ ] cover letter(投稿时按期刊模板)
