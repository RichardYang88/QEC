# HW_FULL_INTEGRATION.md — 机时到账后:全量真机基准接入论文的完整流程

论文当前版本(2026-09)使用 4 电路 ×400 shots 的可行性数据(`raw_verify_options.json`)
作为硬件演示。购买 QPU 机时后,按本手册把 128 电路全量基准接入论文。

## 0. 前置事实

- 论文仿真模型 = **warm-start VSCR**,角度文件:
  `vscr_angles_paper_dep.npz`(dep)、`vscr_angles_paper_ad.npz`、`vscr_angles_paper_mixed.npz`。
- 真机管线默认读 `vscr_angles_dep.npz`(v1 快照,可行性数据所用)。
  `qcloud_vscr_new.load_angles` 支持环境变量覆盖:
  `export VSCR_ANGLES_FILE=$PWD/vscr_angles_paper_dep.npz`。
- 全量预算:2 态(0,+)× 2 p(0.02,0.08)× 2 Pauli 帧 × 16 分支 = 128 电路 × 1000 shots。

## 1. 探针(1600 shots,确认管线与配额)

```bash
cd /home/yqc/github/QEC
export VSCR_ANGLES_FILE=$PWD/vscr_angles_paper_dep.npz
./qenv/bin/python -u qcloud_probe_point.py        # 1 数据点 × 100 shots
```

## 2. 全量运行(增量落盘,中途断掉不丢数据)

```bash
bash launch_full.sh        # 内部已带 --late-measure --chunk 32 --dump-raw qcloud_raw_full.json
tail -f qcloud_full.log
```

## 3. 离线分析(无需 token;--analyze 模式)

```bash
./qenv/bin/python -u qcloud_vscr_new.py --mode cloud \
    --analyze qcloud_raw_full.json \
    --p-values 0.02,0.08 --frames 2 --states 0,+ \
    --late-measure --shots 1000 --tag full_paper
```
输出:`qcloud_results_full_paper.json` + `figures/fig_qcloud_hardware_full_paper.png`
(真机点+Wilson 误差棒 vs 同帧理想参考 vs 密集参考曲线)。

## 4. 论文更新清单

1. `paper/main.tex` Results 2.5:
   - 将"feasibility demonstration"段落升级为完整基准:报告每个
     (psi, p) 的 HW vs ideal 与 gap;保留 mid/late 与弛豫卫星峰 characterization
     (改为引用可行性数据作为平台表征)。
   - 摘要中 "$0.80$ ($95\%$ Wilson interval $[0.68,0.88]$)" 替换为全量基准的
     平均 gap / 最低分支保真度。
2. `paper/main.tex` Fig.4 替换为 `fig_qcloud_hardware_full_paper.pdf`
   (用 matplotlib 转 pdf 或改 \includegraphics 为 png);
   可行性四联图降级为 Extended Data Fig.
3. 数值填充:`paper/fill_numbers.py` 已支持 @TOKEN@ 替换;新增全量数值时
   在 `paper_numbers.json` 增加 `hw_full` 字段并扩展 fill 脚本。
4. 重编译:`pdflatex main && bibtex main && pdflatex main && pdflatex main`。

## 5. 一致性校验(提交前必做)

- `VSCR_ANGLES_FILE` 指向 paper 角度时,`--mode verify` 应输出
  worst |circuit-simulator| < 1e-9(论文引用值 @WORSTVER@ 需同步更新)。
- 理想参考曲线 `dense_reference_curves` 与 `paper_numbers.json`
  的 `hardware_frame_numbers` 同源(同一 phi)。
