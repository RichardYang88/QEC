#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the VSCR research-results presentation (16:9, Chinese)."""
import os
from PIL import Image
from lxml import etree
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn

ROOT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(ROOT, "figures")
OUT = os.path.join(ROOT, "report", "VSCR_research_presentation.pptx")

# ---------------- palette & fonts ----------------
NAVY = RGBColor(0x0B, 0x25, 0x45)
NAVY2 = RGBColor(0x16, 0x40, 0x6B)
TEAL = RGBColor(0x1B, 0x9A, 0xAA)
INK = RGBColor(0x22, 0x2A, 0x35)
GRAY = RGBColor(0x5C, 0x66, 0x72)
LIGHT = RGBColor(0xEE, 0xF4, 0xF8)
GOLD = RGBColor(0xE0, 0x8A, 0x1E)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GREEN = RGBColor(0x1F, 0x7A, 0x55)
RED = RGBColor(0xB0, 0x3A, 0x2E)
ROWALT = RGBColor(0xF4, 0xF8, 0xFB)
TEAL_T = RGBColor(0xDD, 0xEF, 0xF3)
CREAM = RGBColor(0xFF, 0xF4, 0xE2)
FONT = "Microsoft YaHei"
MONO = "Consolas"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
_page = [0]


def new_slide():
    return prs.slides.add_slide(BLANK)


def set_font(run, size=16, bold=False, color=INK, name=FONT, italic=False):
    f = run.font
    f.size = Pt(size)
    f.bold = bold
    f.italic = italic
    f.color.rgb = color
    f.name = name
    rPr = run._r.get_or_add_rPr()
    ea = rPr.find(qn('a:ea'))
    if ea is None:
        ea = etree.SubElement(rPr, qn('a:ea'))
    ea.set('typeface', name)


def rect(slide, x, y, w, h, color, shape=MSO_SHAPE.RECTANGLE, line=None):
    sh = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = color
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
        sh.line.width = Pt(1.0)
    sh.shadow.inherit = False
    return sh


def add_text(slide, x, y, w, h, runs, size=16, bold=False, color=INK,
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, name=FONT,
             line_spacing=1.0, italic=False):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = align
    p.line_spacing = line_spacing
    if isinstance(runs, str):
        runs = [(runs, {})]
    for text, ov in runs:
        r = p.add_run()
        r.text = text
        set_font(r, size=ov.get('size', size), bold=ov.get('bold', bold),
                 color=ov.get('color', color), name=ov.get('name', name),
                 italic=ov.get('italic', italic))
    return tb


def add_bullets(slide, x, y, w, h, items, size=15, color=INK,
                space=7, line_spacing=1.14):
    """items: list of (level, runs); runs = str or list[(text, overrides)]"""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    first = True
    for level, runs in items:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.space_before = Pt(space)
        p.space_after = Pt(1)
        p.line_spacing = line_spacing
        sz = size if level == 0 else size - 1.5
        if level == 0:
            r0 = p.add_run()
            r0.text = "▪  "
            set_font(r0, size=sz, bold=True, color=TEAL)
        else:
            r0 = p.add_run()
            r0.text = "      –  "
            set_font(r0, size=sz, bold=True, color=GRAY)
        if isinstance(runs, str):
            runs = [(runs, {})]
        for text, ov in runs:
            r = p.add_run()
            r.text = text
            set_font(r, size=ov.get('size', sz), bold=ov.get('bold', False),
                     color=ov.get('color', color), name=ov.get('name', FONT))
    return tb



def add_fig(slide, fname, x, y, w=None, h=None):
    path = os.path.join(FIG, fname)
    iw, ih = Image.open(path).size
    if w and not h:
        h = w * ih / iw
    if h and not w:
        w = h * iw / ih
    return slide.shapes.add_picture(path, Inches(x), Inches(y),
                                    Inches(w), Inches(h))


def add_card(slide, x, y, w, h, big, label, sub=None, big_color=TEAL,
             fill=LIGHT, big_size=22, label_size=11.5):
    sh = rect(slide, x, y, w, h, fill, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
              line=RGBColor(0xD3, 0xE1, 0xEC))
    sh.adjustments[0] = 0.09
    tf = sh.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Inches(0.1)
    tf.margin_top = Inches(0.06)
    tf.margin_bottom = Inches(0.04)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = big
    set_font(r, size=big_size, bold=True, color=big_color)
    p2 = tf.add_paragraph()
    p2.alignment = PP_ALIGN.CENTER
    p2.space_before = Pt(3)
    r2 = p2.add_run()
    r2.text = label
    set_font(r2, size=label_size, color=INK)
    if sub:
        p3 = tf.add_paragraph()
        p3.alignment = PP_ALIGN.CENTER
        r3 = p3.add_run()
        r3.text = sub
        set_font(r3, size=9.5, color=GRAY)
    return sh


def add_callout(slide, x, y, w, h, runs, fill=CREAM, border=GOLD,
                size=13.5, align=PP_ALIGN.LEFT, tcolor=NAVY, line_spacing=1.12):
    sh = rect(slide, x, y, w, h, fill, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
              line=border)
    sh.adjustments[0] = 0.12
    tf = sh.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Inches(0.18)
    tf.margin_top = tf.margin_bottom = Inches(0.06)
    p = tf.paragraphs[0]
    p.alignment = align
    p.line_spacing = line_spacing
    if isinstance(runs, str):
        runs = [(runs, {})]
    for text, ov in runs:
        r = p.add_run()
        r.text = text
        set_font(r, size=ov.get('size', size), bold=ov.get('bold', True),
                 color=ov.get('color', tcolor))
    return sh


def add_table(slide, x, y, w, h, data, col_widths, hl_col=None,
              font_size=11, header_size=11.5):
    rows, cols = len(data), len(data[0])
    gfx = slide.shapes.add_table(rows, cols, Inches(x), Inches(y),
                                 Inches(w), Inches(h))
    tbl = gfx.table
    tbl.first_row = False
    tbl.horz_banding = False
    for i, cw in enumerate(col_widths):
        tbl.columns[i].width = Inches(cw)
    tbl.rows[0].height = Inches(0.34)
    for i in range(1, rows):
        tbl.rows[i].height = Inches(0.3)
    for ri, row in enumerate(data):
        for ci, val in enumerate(row):
            cell = tbl.cell(ri, ci)
            cell.margin_left = cell.margin_right = Inches(0.03)
            cell.margin_top = cell.margin_bottom = Inches(0.01)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            if ri == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = NAVY
            elif hl_col is not None and ci == hl_col:
                cell.fill.solid()
                cell.fill.fore_color.rgb = TEAL_T
            elif ri % 2 == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = ROWALT
            else:
                cell.fill.solid()
                cell.fill.fore_color.rgb = WHITE
            tf = cell.text_frame
            tf.word_wrap = False
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER
            r = p.add_run()
            r.text = str(val)
            bold = ri == 0 or (hl_col is not None and ci == hl_col)
            ctext = WHITE if ri == 0 else (NAVY if (hl_col is not None and
                                           ci == hl_col) else INK)
            set_font(r, size=header_size if ri == 0 else font_size,
                     bold=bold, color=ctext)
    return tbl


def content_slide(title, kicker=None):
    s = new_slide()
    rect(s, 0, 0, 13.333, 0.14, NAVY)
    rect(s, 0, 0, 2.4, 0.14, TEAL)
    ty = 0.38
    if kicker:
        add_text(s, 0.55, 0.3, 10.0, 0.3, kicker.upper(), size=11,
                 bold=True, color=TEAL)
        ty = 0.56
    add_text(s, 0.55, ty, 12.3, 0.72, title, size=26, bold=True, color=NAVY)
    rect(s, 0.57, ty + 0.66, 1.35, 0.045, GOLD)
    _page[0] += 1
    add_text(s, 12.35, 7.1, 0.75, 0.3, str(_page[0]), size=10, color=GRAY,
             align=PP_ALIGN.RIGHT)
    add_text(s, 0.55, 7.1, 8.5, 0.3,
             "VSCR · Variational Syndrome-Conditioned Recovery · [[5,1,3]]",
             size=9, color=GRAY)
    return s


def chip(slide, x, y, w, h, text, fill=NAVY2, tcolor=WHITE, size=12.5):
    sh = rect(slide, x, y, w, h, fill, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
              line=TEAL)
    sh.adjustments[0] = 0.5
    tf = sh.text_frame
    tf.word_wrap = False
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = text
    set_font(r, size=size, bold=True, color=tcolor)
    return sh


# ---------------- data (from full_run.log / vscr_results.npz) ----------------
DEPOL = [
    ["p", "Raw", "完美码解码器", "ZNE", "VD", "LinDR", "VSCR(本工作)"],
    ["0.005", "0.9752", "0.9998", "1.0000", "1.0000", "0.9793", "0.9993"],
    ["0.010", "0.9510", "0.9993", "0.9999", "0.9998", "0.9844", "0.9983"],
    ["0.020", "0.9039", "0.9974", "0.9996", "0.9993", "0.9930", "0.9955"],
    ["0.040", "0.8154", "0.9903", "0.9967", "0.9969", "1.0040", "0.9868"],
    ["0.070", "0.6958", "0.9721", "0.9838", "0.9895", "1.0068", "0.9669"],
    ["0.100", "0.5908", "0.9470", "0.9570", "0.9758", "0.9962", "0.9406"],
    ["0.150", "0.4447", "0.8942", "0.8770", "0.9330", "0.9563", "0.8869"],
    ["0.200", "0.3298", "0.8339", "0.7559", "0.8532", "0.8982", "0.8267"],
]
AMPD = [
    ["p", "Raw", "完美码解码器", "ZNE", "VD", "LinDR", "VSCR(本工作)"],
    ["0.005", "0.9876", "1.0000", "1.0000", "1.0000", "1.0001", "0.9992"],
    ["0.010", "0.9752", "0.9998", "1.0000", "0.9998", "1.0000", "0.9983"],
    ["0.020", "0.9509", "0.9993", "1.0000", "0.9994", "1.0000", "0.9964"],
    ["0.040", "0.9035", "0.9974", "0.9997", "0.9974", "1.0000", "0.9918"],
    ["0.070", "0.8355", "0.9923", "0.9986", "0.9917", "1.0000", "0.9834"],
    ["0.100", "0.7712", "0.9847", "0.9959", "0.9823", "0.9998", "0.9732"],
    ["0.150", "0.6718", "0.9671", "0.9865", "0.9577", "0.9983", "0.9528"],
    ["0.200", "0.5818", "0.9442", "0.9690", "0.9196", "0.9945", "0.9285"],
]
MIXED = [
    ["p", "Raw", "完美码解码器", "ZNE", "VD", "LinDR", "VSCR(本工作)"],
    ["0.005", "0.9692", "0.9997", "1.0000", "0.9999", "1.0017", "0.9971"],
    ["0.010", "0.9392", "0.9990", "0.9998", "0.9997", "1.0009", "0.9939"],
    ["0.020", "0.8817", "0.9961", "0.9989", "0.9988", "1.0000", "0.9865"],
    ["0.040", "0.7759", "0.9854", "0.9919", "0.9946", "0.9997", "0.9686"],
    ["0.070", "0.6383", "0.9595", "0.9637", "0.9815", "1.0004", "0.9353"],
    ["0.100", "0.5229", "0.9253", "0.9118", "0.9572", "0.9990", "0.8971"],
    ["0.150", "0.3717", "0.8582", "0.7810", "0.8811", "0.9853", "0.8284"],
    ["0.200", "0.2617", "0.7875", "0.6199", "0.7458", "0.9535", "0.7603"],
]
SCHED = [
    ["阶段", "Epochs", "学习率", "噪声范围 p"],
    ["① 热身", "600", "3e-2", "[0.02, 0.08]"],
    ["② 扩展", "400", "3e-2", "[0.02, 0.15]"],
    ["③ 精调", "1200", "1e-3", "[0.02, 0.15]"],
]


# ---------------- slides ----------------
def slide_cover():
    s = new_slide()
    rect(s, 0, 0, 13.333, 7.5, NAVY)
    rect(s, 0, 0, 13.333, 0.09, TEAL)
    rect(s, 0.9, 1.62, 0.09, 1.95, GOLD)
    add_text(s, 0.9, 0.95, 11.5, 0.4,
             "QUANTUM ERROR CORRECTION · 研究成果汇报", size=14, bold=True,
             color=TEAL)
    add_text(s, 1.25, 1.6, 11.3, 1.0,
             "VSCR：变分综合征条件恢复", size=40, bold=True, color=WHITE)
    add_text(s, 1.25, 2.52, 11.3, 0.9,
             "Variational Syndrome-Conditioned Recovery for the [[5,1,3]] Code",
             size=19, color=RGBColor(0xBF, 0xD7, 0xEA), italic=True)
    add_text(s, 1.25, 3.55, 11.3, 0.75,
             "投影式综合征测量 + 每综合征学习一个变分酉恢复 · 无监督训练 · "
             "对标最优解码器与误差缓解基线", size=15,
             color=RGBColor(0x9F, 0xB8, 0xCF))
    chip(s, 1.25, 4.6, 2.7, 0.52, "无监督训练 · 零错误标签")
    chip(s, 4.15, 4.6, 3.3, 0.52, "量子仪器恢复 · 返回真实量子态")
    chip(s, 7.65, 4.6, 3.3, 0.52, "逼近最优解码器性能")
    rect(s, 0.9, 6.35, 11.5, 0.012, RGBColor(0x35, 0x55, 0x78))
    add_text(s, 0.9, 6.55, 11.5, 0.4,
             "仿真平台：PyTorch complex128 密度矩阵可微模拟器（n=5，32×32）· "
             "2026-08", size=11.5, color=RGBColor(0x8F, 0xA8, 0xC0))


def slide_agenda():
    s = content_slide("目录 / Agenda")
    items = [
        ("01", "研究背景与动机"),
        ("02", "纯酉方案（SSVR）为何失败"),
        ("03", "VSCR 方法：量子仪器架构与训练"),
        ("04", "基线方法（Baselines）"),
        ("05", "实验结果：三类噪声基准"),
        ("06", "消融实验与机理分析"),
        ("07", "结论与展望"),
    ]
    for i, (num, text) in enumerate(items):
        col = i % 2
        row = i // 2
        x = 1.1 + col * 6.1
        y = 1.85 + row * 1.22
        sh = rect(s, x, y, 0.58, 0.58, TEAL if i < 6 else NAVY,
                  shape=MSO_SHAPE.ROUNDED_RECTANGLE)
        sh.adjustments[0] = 0.22
        tf = sh.text_frame
        tf.margin_left = tf.margin_right = tf.margin_top = \
            tf.margin_bottom = 0
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = num
        set_font(r, size=17, bold=True, color=WHITE)
        add_text(s, x + 0.82, y + 0.09, 5.1, 0.5, text, size=17.5, bold=True,
                 color=INK)


def slide_background():
    s = content_slide("研究背景：让机器学习一个“真恢复”",
                      kicker="01 · 研究背景")
    add_bullets(s, 0.55, 1.65, 7.35, 4.3, [
        (0, [("量子纠错（QEC）", {'bold': True, 'color': NAVY}),
             ("是容错量子计算的核心：将逻辑量子比特编码进多物理比特纠缠态，"
              "通过综合征测量诊断并纠正错误", {})]),
        (0, [("[[5,1,3]] 完美码", {'bold': True, 'color': NAVY}),
             ("：最小的完美量子码 —— 5 个物理比特编码 1 个逻辑比特，"
              "距离 3，可纠正任意单比特错误", {})]),
        (0, [("现有方案的局限：", {'bold': True, 'color': NAVY}),
             ("传统解码器只给 Pauli 纠正；误差缓解（ZNE / VD）"
              "只修正期望值，不返回量子态", {})]),
        (0, [("研究问题：", {'bold': True, 'color': GOLD}),
             ("能否以无监督、数据驱动的方式学习一个量子恢复操作，"
              "返回真正被纠正的量子态，并逼近最优解码器？", {})]),
    ], size=14.5, space=11)
    rect(s, 8.25, 1.65, 4.5, 3.6, LIGHT, shape=MSO_SHAPE.ROUNDED_RECTANGLE,
         line=RGBColor(0xD3, 0xE1, 0xEC))
    add_text(s, 8.55, 1.88, 4.0, 0.4, "实验平台", size=15, bold=True,
             color=NAVY)
    add_bullets(s, 8.55, 2.35, 3.95, 2.8, [
        (0, "PyTorch complex128 密度矩阵模拟器（n=5，32×32 算符）"),
        (0, "真实 Adam 梯度穿过物理噪声信道反传"),
        (0, "噪声模型：去极化 / 幅值阻尼 / 混合"),
        (0, "评估：随机逻辑态，每个噪声率 40 个测试态"),
    ], size=12, space=8)
    add_callout(s, 0.55, 6.0, 12.2, 0.78,
                [("目标：", {'color': GOLD}),
                 ("一个同时具备 ① 返回真纠正量子态、② 无标签训练、"
                  "③ 对标最优解码器与误差缓解基线 的变分量子–经典恢复方案",
                  {})],
                size=13.5, fill=RGBColor(0xE9, 0xF3, 0xF6), border=TEAL)


def slide_ssvr_fail():
    s = content_slide("动机：为什么纯酉恢复（SSVR）必然失败",
                      kicker="02 · 动机分析")
    add_bullets(s, 0.55, 1.65, 12.2, 3.9, [
        (0, [("原方案 SSVR：", {'bold': True, 'color': NAVY}),
             ("对含噪态 ρ_noisy 直接施加单一变分酉 "
              "R(φ(软综合征))，不做任何测量投影", {})]),
        (0, [("根本障碍：", {'bold': True, 'color': NAVY}),
             ("酉信道保持密度矩阵特征值谱，", {}),
             ("不能投影 / 不能纯化", {'bold': True, 'color': RED}),
             ("；而综合征解码的本质是把含噪态投影到综合征子空间（纯化）"
              "—— 任何酉都无法模拟这一映射", {})]),
        (0, [("优化后果：", {'bold': True, 'color': NAVY}),
             ("对非酉目标信道做最好的酉近似 = “什么都不做” —— "
              "保真度优化必然驱动 ", {}),
             ("R → I", {'bold': True, 'color': RED})]),
        (0, [("实验证实：", {'bold': True, 'color': NAVY}),
             ("在所有噪声模型下，训练后的保真度坍缩到 raw（未纠错）曲线，"
              "与优化器设置无关", {})]),
    ], size=14.5, space=12)
    add_callout(s, 0.55, 5.85, 12.2, 0.95,
                [("结论：", {'color': GOLD, 'size': 15}),
                 ("基于测量的投影不可省略 —— 恢复操作必须是量子仪器"
                  "（quantum instrument），而不是单个酉。",
                  {'size': 15})],
                size=15)


def slide_method():
    s = content_slide("VSCR 方法：投影式综合征仪器 + 变分恢复",
                      kicker="03 · 方法")
    steps = [
        ("1", "投影式综合征测量",
         "4 个稳定子 g₁..g₄ → 16 个综合征投影算符 "
         "P_s = Πᵢ (I + (−1)^sᵢ gᵢ)/2，每个秩为 2，一次性预计算"),
        ("2", "每综合征学习变分恢复",
         "对每个结果 s 施加变分酉 R_s = Π_k exp(−iθ_{s,k}P_k/2)："
         "60 个旋转槽位（Rz/Rx/Rz + Rzz 耦合 × 3 层）"),
        ("3", "超网络参数共享",
         "24 维可学习综合征嵌入 → 超网络生成 60 个转角 "
         "φ_s = φ_base + hnet(emb_s)；一次前向批量构造全部 16 个 R_s"),
    ]
    for i, (num, title, body) in enumerate(steps):
        x = 0.55 + i * 4.16
        rect(s, x, 1.62, 3.97, 2.42, LIGHT,
             shape=MSO_SHAPE.ROUNDED_RECTANGLE,
             line=RGBColor(0xD3, 0xE1, 0xEC))
        circ = rect(s, x + 0.16, 1.8, 0.46, 0.46, TEAL, shape=MSO_SHAPE.OVAL)
        tf = circ.text_frame
        tf.margin_left = tf.margin_right = tf.margin_top = \
            tf.margin_bottom = 0
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = num
        set_font(r, size=17, bold=True, color=WHITE)
        add_text(s, x + 0.74, 1.86, 3.1, 0.4, title, size=14.5, bold=True,
                 color=NAVY)
        add_text(s, x + 0.2, 2.42, 3.6, 1.55, body, size=11.5, color=INK,
                 line_spacing=1.15)
    add_callout(s, 0.55, 4.35, 12.23, 1.05,
                [("恢复态：", {'color': TEAL}),
                 ("ρ_rec = Σ_s R_s P_s ρ P_s R_s†      ", {}),
                 ("（先测量投影、再按结果恢复 —— 合法量子仪器的 CP 分解）",
                  {'size': 11.5, 'color': GRAY, 'bold': False})],
                fill=RGBColor(0xE9, 0xF3, 0xF6), border=TEAL, size=14)
    add_callout(s, 0.55, 5.62, 12.23, 1.05,
                [("训练目标（无标签）：", {'color': GOLD}),
                 ("F = Σ_s ⟨ψ| R_s P_s ρ_noisy P_s R_s† |ψ⟩，"
                  "对随机逻辑态 |ψ⟩ 与噪声率 p 最大化期望恢复保真度 —— "
                  "全程不需要错误标签", {'bold': False})],
                size=14)


def slide_training():
    s = content_slide("训练策略：课程学习 + 多种子无标签选择",
                      kicker="03 · 方法")
    add_bullets(s, 0.55, 1.7, 7.1, 4.4, [
        (0, [("课程式训练（右表）：", {'bold': True, 'color': NAVY}),
             ("先窄噪声范围热身，逐步扩展，最后低学习率精调", {})]),
        (0, [("幅值阻尼噪声例外：", {'bold': True, 'color': NAVY}),
             ("直接在全噪声范围训练 —— 其最优纠正一开始就是非 Pauli 的，"
              "课程反而有害", {})]),
        (0, [("多种子训练（1234 / 2024 / 777）：", {'bold': True,
             'color': NAVY}),
             ("16 个综合征中偶尔有 1 个陷入坏局部最优，且依赖种子", {})]),
        (0, [("无标签综合征选择：", {'bold': True, 'color': GOLD}),
             ("以最差综合征条件保真度 cf_s = E[⟨ψ|R_sP_sρP_sR_s†|ψ⟩]/"
              "E[Tr(P_sρ)] 排序候选（纯自计算诊断，不用解码表），"
              "并通过验证保真度门限（≥0.85）剔除退化解", {})]),
        (0, [("全流程无监督：", {'bold': True, 'color': NAVY}),
             ("不使用错误标签、不使用解码器表", {})]),
    ], size=13.5, space=9)
    add_table(s, 8.0, 1.85, 4.75, 1.7, SCHED,
              [1.35, 1.05, 0.95, 1.4], font_size=11.5, header_size=12)
    add_text(s, 8.0, 3.62, 4.8, 0.6, "去极化噪声课程表（VSCR_SCHEDULES）",
             size=10.5, color=GRAY, align=PP_ALIGN.CENTER)
    add_card(s, 8.0, 4.35, 2.3, 1.55, "3 seeds", "多种子训练",
             sub="1234 / 2024 / 777", big_size=19)
    add_card(s, 10.45, 4.35, 2.3, 1.55, "min cf_s", "最差综合征条件保真度",
             sub="无标签自诊断排序", big_size=19)


def slide_baselines():
    s = content_slide("基线方法（Baselines）", kicker="04 · 基线")
    rows = [
        ("Raw（无恢复）", "下界", "不做任何恢复操作"),
        ("Perfect-code 解码器", "上界 · Pauli 仪器",
         "最小权重单错查找表纠正，Pauli 型仪器的理论上限"),
        ("ZNE", "Oracle 式缓解", "精确噪声尺度外推（零噪声极限）"),
        ("Virtual Distillation (VD)", "纯化式缓解",
         "Tr[ρⁿO]/Tr[ρⁿ] 纯化，n = 2"),
        ("LinDR (vnCDR 式)", "线性恢复",
         "全 Pauli 基 1024 特征，岭正则最小二乘（验证集选正则）"),
    ]
    y = 1.7
    for name, tag, desc in rows:
        rect(s, 0.55, y, 12.23, 0.66, WHITE if (rows.index((name, tag, desc))
             % 2 == 0) else ROWALT, line=RGBColor(0xD9, 0xE4, 0xEE))
        add_text(s, 0.75, y + 0.13, 3.6, 0.45, name, size=14, bold=True,
                 color=NAVY)
        chip(s, 4.5, y + 0.1, 2.15, 0.44, tag, fill=TEAL, size=11)
        add_text(s, 6.85, y + 0.15, 5.85, 0.45, desc, size=12, color=INK)
        y += 0.75
    add_callout(s, 0.55, 5.75, 12.23, 0.98,
                [("注意：", {'color': GOLD}),
                 ("ZNE / VD / LinDR 输出的是修正期望值或线性重构，"
                  "并非物理恢复信道（LinDR 重构不保证半正定，中等噪声下 "
                  ">1.0 属 oracle 式拟合）；", {}),
                 ("VSCR 是唯一返回真纠正密度矩阵的方法。",
                  {'color': NAVY})], size=13)


def slide_depolarizing_curves():
    s = content_slide("结果① 去极化噪声：保真度曲线与逻辑错误率",
                      kicker="05 · 实验结果")
    add_fig(s, "fig_fidelity_depolarizing.png", 0.55, 1.55, w=6.0)
    add_fig(s, "fig_ler_depolarizing.png", 6.78, 1.55, w=6.0)
    add_bullets(s, 0.55, 5.85, 12.2, 1.2, [
        (0, [("VSCR 全程紧贴完美码解码器曲线（最大差距仅 0.6%）；", {}),
             ("高噪声区 p ≥ 0.15 超过 ZNE", {'bold': True, 'color': GREEN}),
             ("（0.8869 vs 0.8770；0.8267 vs 0.7559）", {})]),
        (0, [("p=0.10 时较 raw 绝对提升 +0.3498（0.5908 → 0.9406）；", {}),
             ("LinDR 中等噪声 >1.0 = 线性重构非物理（非 PSD），属 oracle 式拟合",
              {'color': GRAY})]),
    ], size=13, space=6)


def slide_depolarizing_table():
    s = content_slide("结果① 去极化噪声：完整数据与 p=0.10 对比",
                      kicker="05 · 实验结果")
    add_table(s, 0.55, 1.6, 6.45, 3.2, DEPOL,
              [0.72, 0.95, 1.28, 0.82, 0.82, 0.9, 0.96], hl_col=6,
              font_size=11, header_size=11.5)
    add_fig(s, "fig_summary_depolarizing_p010.png", 7.45, 1.75, w=5.3)
    add_text(s, 7.45, 5.0, 5.3, 0.4, "p = 0.10 各方法平均保真度对比（越高越好）",
             size=11, color=GRAY, align=PP_ALIGN.CENTER)
    add_card(s, 0.55, 5.3, 2.95, 1.4, "+0.3498", "vs Raw @ p=0.10",
             sub="0.5908 → 0.9406", big_color=GREEN)
    add_card(s, 3.65, 5.3, 2.95, 1.4, "−0.6%", "vs 最优解码器（最大差距）",
             sub="全程 8 个噪声点", big_color=TEAL)
    add_card(s, 6.75, 5.3, 2.95, 1.4, "0.8869", "vs ZNE 0.8770 @ p=0.15",
             sub="高噪声区胜出", big_color=GREEN)
    add_card(s, 9.85, 5.3, 2.93, 1.4, ">1.0", "LinDR @ 中等噪声",
             sub="非物理重构（非 PSD）", big_color=RED, big_size=19)


def slide_amplitude_damping():
    s = content_slide("结果② 幅值阻尼噪声：学习非 Pauli 纠正",
                      kicker="05 · 实验结果")
    add_fig(s, "fig_fidelity_amplitude_damping.png", 0.55, 1.6, w=6.5)
    add_bullets(s, 7.35, 1.85, 5.45, 3.6, [
        (0, [("VSCR 全程与完美码解码器相差约 1.2% 以内", {})]),
        (0, [("p = 0.20 超过 Virtual Distillation",
              {'bold': True, 'color': GREEN}),
             ("（0.9285 vs 0.9196）", {})]),
        (0, [("这是学习信道胜过 Pauli 查找表的唯一区间：",
              {'bold': True, 'color': NAVY}),
             ("幅值阻尼的最优纠正本质上是连续、非 Pauli 的，"
              "查找表已先让出部分优势 —— 正是变分恢复的价值所在", {})]),
        (0, [("p=0.10：0.7712 → 0.9732（+0.2021，相对 +26.2%）", {})]),
    ], size=13.5, space=10)
    add_callout(s, 7.35, 5.55, 5.45, 1.0,
                [("启示：", {'color': GOLD}),
                 ("噪声越“非 Pauli”，学习式恢复的优势越大", {})],
                size=13)


def slide_mixed():
    s = content_slide("结果③ 混合噪声（去极化 + 幅值阻尼）",
                      kicker="05 · 实验结果")
    add_fig(s, "fig_fidelity_mixed.png", 0.55, 1.6, w=6.5)
    add_bullets(s, 7.35, 1.85, 5.45, 3.9, [
        (0, [("最困难的基准：", {'bold': True, 'color': NAVY}),
             ("两条噪声信道叠加，与解码器差距扩大到约 2.5–3%", {})]),
        (0, [("原因：通常有一个综合征无法完全“锁定”"
              "（精细逻辑梯度信号仅 O(p)）", {})]),
        (0, [("仍显著优于缓解类基线：", {'bold': True, 'color': NAVY}),
             ("p ≥ 0.15 起超过 ZNE（0.8284 vs 0.7810 @0.15）；"
              "p ≥ 0.20 超过 VD（0.7603 vs 0.7458）", {})]),
        (0, [("对复合噪声保持稳健，验证方法的普适性", {})]),
    ], size=13.5, space=10)
    add_callout(s, 7.35, 5.65, 5.45, 0.95,
                [("p=0.10：0.5229 → 0.8971（+0.3742 绝对提升）", {})],
                size=13, fill=RGBColor(0xE9, 0xF3, 0xF6), border=TEAL)



def slide_ablation():
    s = content_slide("消融实验：真实梯度 vs 随机游走“假梯度”",
                      kicker="06 · 消融与分析")
    add_fig(s, "fig_training_curves.png", 0.55, 1.6, w=6.5)
    add_card(s, 7.4, 1.85, 2.6, 1.5, "0.9738", "真实梯度 · 验证保真度",
             sub="穿过物理信道反传", big_color=GREEN)
    add_card(s, 10.15, 1.85, 2.6, 1.5, "0.5955", "随机游走 · 验证保真度",
             sub="复现参考实现的失效", big_color=RED)
    add_bullets(s, 7.4, 3.75, 5.4, 2.3, [
        (0, [("随机游走消融：", {'bold': True, 'color': NAVY}),
             ("以 np.random.randn 冒充梯度（复现参考实现 ACE-QEC 中损坏的 "
              "train_*_step 更新规则）—— 保真度始终停滞在 raw 水平，"
              "从未学会纠错", {})]),
        (0, [("真实梯度：", {'bold': True, 'color': NAVY}),
             ("Adam 梯度穿过物理噪声信道精确反传，是恢复能够被学会的必要条件",
              {})]),
    ], size=13, space=9)
    add_callout(s, 0.55, 6.15, 12.23, 0.68,
                [("差距 +0.3783：", {'color': GOLD}),
                 ("“能微分”与“假装微分”的分水岭 —— 可微仿真器是本工作的基础设施",
                  {})], size=13)


def slide_diagnostics():
    s = content_slide("机理诊断：差距从哪来？", kicker="06 · 消融与分析")
    add_bullets(s, 0.55, 1.7, 12.2, 4.2, [
        (0, [("为什么个别综合征“锁不住”？", {'bold': True, 'color': NAVY}),
             (" [[5,1,3]] 距离为 3：投影后任意纠正都把权重 ≤2 的错误映回码空间，"
              "残余误差只来自落错逻辑态的双错事件 —— 每个综合征只有 O(p) 的"
              "精细逻辑梯度信号，训练偶尔欠拟合", {})]),
        (0, [("解码器替换实验：", {'bold': True, 'color': NAVY}),
             ("把解码器纠正 C_s 直接插入学习仪器，精确复现完美解码器曲线 —— "
              "剩余差距完全来自学习到的酉，而非仪器设计", {})]),
        (0, [("子空间作用分析：", {'bold': True, 'color': NAVY}),
             ("学到的 R_s 与解码器 Pauli C_s 仅差保真度中性的综合征相关相位"
              "（±i、±1）；15/16 综合征功能上完全正确，其余由多种子选择兜底",
              {})]),
        (0, [("超参数经验：", {'bold': True, 'color': NAVY}),
             ("码空间惩罚项 lam > 0 会破坏高学习率训练，lam = 0 最佳", {})]),
    ], size=14, space=12)
    add_callout(s, 0.55, 5.95, 12.23, 0.8,
                [("结论：仪器结构（测量 + 分综合征恢复）是正确且充分的，"
                  "优化是唯一瓶颈，且已用课程 + 多种子选择系统性缓解", {})],
                size=13.5, fill=RGBColor(0xE9, 0xF3, 0xF6), border=TEAL)



def slide_conclusion():
    s = content_slide("结论：关键贡献与核心数字", kicker="07 · 结论")
    add_text(s, 0.55, 1.6, 6.4, 0.4, "关键贡献", size=16, bold=True,
             color=NAVY)
    add_bullets(s, 0.55, 2.05, 6.5, 4.3, [
        (0, [("核心洞见：", {'bold': True, 'color': GOLD}),
             ("恢复必须是量子仪器 —— 测量投影不可被纯酉学习替代"
              "（SSVR 坍缩到 R→I 的反面教材）", {})]),
        (0, [("VSCR 架构：", {'bold': True, 'color': NAVY}),
             ("投影式综合征测量 + 每综合征变分酉恢复 + 超网络参数共享", {})]),
        (0, [("完全无监督：", {'bold': True, 'color': NAVY}),
             ("无错误标签、无解码器表，纯自计算的综合征诊断与模型选择", {})]),
        (0, [("唯一性：", {'bold': True, 'color': NAVY}),
             ("所有基线中唯一返回真纠正密度矩阵的方法", {})]),
    ], size=13.5, space=10)
    add_text(s, 7.4, 1.6, 5.4, 0.4, "核心数字", size=16, bold=True,
             color=NAVY)
    add_card(s, 7.4, 2.1, 2.6, 1.45, "0.6%", "距最优解码器最大差距",
             sub="去极化噪声全程", big_color=TEAL)
    add_card(s, 10.2, 2.1, 2.6, 1.45, "+0.35", "vs Raw @ p=0.10",
             sub="去极化 · 绝对保真度增益", big_color=GREEN)
    add_card(s, 7.4, 3.7, 2.6, 1.45, "0.8869", "p=0.15 超过 ZNE",
             sub="ZNE = 0.8770", big_color=GREEN)
    add_card(s, 10.2, 3.7, 2.6, 1.45, "0.9738", "真实梯度训练保真度",
             sub="随机游走仅 0.5955", big_color=TEAL)
    add_callout(s, 0.55, 6.05, 12.23, 0.72,
                [("一句话总结：", {'color': GOLD}),
                 ("在 [[5,1,3]] 码上，无监督训练的 VSCR 达到与最优查找表解码器"
                  "几乎相同的恢复保真度，并在高噪声 / 非 Pauli 噪声区胜过 "
                  "ZNE / VD 等误差缓解基线。", {'bold': False})], size=13)


def slide_reproducibility():
    s = content_slide("可复现性与未来工作", kicker="07 · 结论")
    rect(s, 0.55, 1.7, 6.3, 2.6, NAVY, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
    cmds = [
        "cd QEC && source qenv/bin/activate",
        "python smoke_test.py    # 冒烟测试 ~30 s",
        "python quick_eval.py    # 快速评估 ~12 min",
        "python ssvr_qec.py      # 完整基准 ~25 min",
    ]
    for i, c in enumerate(cmds):
        add_text(s, 0.85, 1.95 + i * 0.55, 5.8, 0.45, c, size=12.5,
                 color=WHITE, name=MONO)
    add_bullets(s, 0.55, 4.55, 6.3, 2.0, [
        (0, [("固定种子：", {'bold': True, 'color': NAVY}),
             ("SEED=1234；训练种子 1234 / 2024 / 777", {})]),
        (0, [("完整运行总耗时约 20 分钟（单卡）", {})]),
        (0, [("产物：", {'bold': True, 'color': NAVY}),
             ("figures/*.png + vscr_results.npz + report/", {})]),
    ], size=12.5, space=7)
    add_text(s, 7.35, 1.65, 5.4, 0.4, "未来工作", size=16, bold=True,
             color=NAVY)
    add_bullets(s, 7.35, 2.15, 5.4, 3.6, [
        (0, [("扩展到更大的码：表面码 / LDPC 码、多逻辑比特", {})]),
        (0, [("硬件友好的软综合征 / POVM 变体（减少测量开销）", {})]),
        (1, [("将超网络改为以噪声模型为条件，实现“一次训练、多噪声复用”",
              {})]),
        (0, [("与误差缓解 / 容错计算管线端到端集成", {})]),
        (0, [("迁移到真实量子硬件（采样版训练）", {})]),
    ], size=13, space=9)


def slide_thanks():
    s = new_slide()
    rect(s, 0, 0, 13.333, 7.5, NAVY)
    rect(s, 0, 0, 13.333, 0.09, TEAL)
    add_text(s, 0, 2.7, 13.333, 1.1, "谢谢 · 欢迎提问", size=44, bold=True,
             color=WHITE, align=PP_ALIGN.CENTER)
    add_text(s, 0, 3.95, 13.333, 0.5,
             "VSCR: Variational Syndrome-Conditioned Recovery for the "
             "[[5,1,3]] Code", size=15, italic=True,
             color=RGBColor(0xBF, 0xD7, 0xEA), align=PP_ALIGN.CENTER)
    rect(s, 5.87, 4.75, 1.6, 0.012, GOLD)
    add_text(s, 0, 5.15, 13.333, 0.9,
             "代码：ssvr_qec.py（完整基准） · 数据：vscr_results.npz · "
             "报告：report/VSCR_final_report.md", size=12.5,
             color=RGBColor(0x9F, 0xB8, 0xCF), align=PP_ALIGN.CENTER)


# ---------------- assemble ----------------
def main():
    slide_cover()
    slide_agenda()
    slide_background()
    slide_ssvr_fail()
    slide_method()
    slide_training()
    slide_baselines()
    slide_depolarizing_curves()
    slide_depolarizing_table()
    slide_amplitude_damping()
    slide_mixed()
    slide_ablation()
    slide_diagnostics()
    slide_conclusion()
    slide_reproducibility()
    slide_thanks()
    prs.save(OUT)
    print("Saved:", OUT)
    print("Slides:", len(prs.slides._sldIdLst))


if __name__ == "__main__":
    main()
