"""Build the customer-facing model comparison report."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor, Twips


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "reports/customer_model_comparison"
ASSET_DIR = OUTPUT_DIR / "assets"
OUTPUT_DOCX = OUTPUT_DIR / "管道泄漏智能检测系统_模型方案对比分析报告_客户版.docx"

SKILL_ROOT = Path(
    r"C:\Users\WingYouther\.codex\plugins\cache\openai-primary-runtime\documents\26.715.12143\skills\documents"
)
sys.path.insert(0, str(SKILL_ROOT / "scripts"))
from table_geometry import apply_table_geometry  # noqa: E402


BLUE = "2E74B5"
DARK_BLUE = "1F4D78"
NAVY = "0B2545"
LIGHT_BLUE = "E8EEF5"
LIGHT_GRAY = "F2F4F7"
MID_GRAY = "6B7280"
GRID = "D9DEE5"
GREEN = "1F6B45"
GOLD = "7A5A00"
RED = "9B1C1C"
WHITE = "FFFFFF"
BLACK = "111827"


def read_json(relative_path: str) -> dict:
    return json.loads((PROJECT_ROOT / relative_path).read_text(encoding="utf-8"))


def set_run_font(run, size: float | None = None, *, bold: bool | None = None,
                 color: str | None = None, italic: bool | None = None) -> None:
    run.font.name = "Calibri"
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), "Calibri")
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), "Calibri")
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if color is not None:
        run.font.color.rgb = RGBColor.from_string(color)


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_border(cell, *, color: str = GRID, size: int = 6) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.find(qn("w:tcBorders"))
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = qn(f"w:{edge}")
        element = borders.find(tag)
        if element is None:
            element = OxmlElement(f"w:{edge}")
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), str(size))
        element.set(qn("w:color"), color)


def prevent_row_split(row, repeat_header: bool = False) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    cant_split = OxmlElement("w:cantSplit")
    tr_pr.append(cant_split)
    if repeat_header:
        header = OxmlElement("w:tblHeader")
        header.set(qn("w:val"), "true")
        tr_pr.append(header)


def set_cell_text(cell, text: str, *, bold: bool = False, size: float = 9.2,
                  color: str = BLACK, align=WD_ALIGN_PARAGRAPH.CENTER) -> None:
    cell.text = ""
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    paragraph = cell.paragraphs[0]
    paragraph.alignment = align
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.space_after = Pt(0)
    paragraph.paragraph_format.line_spacing = 1.08
    run = paragraph.add_run(text)
    set_run_font(run, size, bold=bold, color=color)


def build_table(doc: Document, headers: list[str], rows: list[list[str]], widths: list[int],
                *, first_col_left: bool = True, highlight_row: int | None = None):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    header_row = table.rows[0]
    for index, text in enumerate(headers):
        set_cell_text(header_row.cells[index], text, bold=True, size=9.0, color=NAVY)
        set_cell_shading(header_row.cells[index], LIGHT_GRAY)
        set_cell_border(header_row.cells[index])
    prevent_row_split(header_row, repeat_header=True)

    for row_index, values in enumerate(rows):
        row = table.add_row()
        for col_index, value in enumerate(values):
            align = WD_ALIGN_PARAGRAPH.LEFT if first_col_left and col_index == 0 else WD_ALIGN_PARAGRAPH.CENTER
            set_cell_text(row.cells[col_index], value, size=9.0, align=align)
            if highlight_row is not None and row_index == highlight_row:
                set_cell_shading(row.cells[col_index], LIGHT_BLUE)
            set_cell_border(row.cells[col_index])
        prevent_row_split(row)
    apply_table_geometry(
        table,
        widths,
        table_width_dxa=9360,
        indent_dxa=120,
        cell_margins_dxa={"top": 80, "bottom": 80, "start": 120, "end": 120},
    )
    return table


def add_paragraph(doc: Document, text: str = "", *, size: float = 11, bold: bool = False,
                  color: str = BLACK, align=WD_ALIGN_PARAGRAPH.LEFT, before: float = 0,
                  after: float = 6, line_spacing: float = 1.10, italic: bool = False,
                  keep_with_next: bool = False):
    paragraph = doc.add_paragraph()
    paragraph.alignment = align
    paragraph.paragraph_format.space_before = Pt(before)
    paragraph.paragraph_format.space_after = Pt(after)
    paragraph.paragraph_format.line_spacing = line_spacing
    paragraph.paragraph_format.keep_with_next = keep_with_next
    if text:
        run = paragraph.add_run(text)
        set_run_font(run, size, bold=bold, color=color, italic=italic)
    return paragraph


def add_label_value(doc: Document, label: str, value: str, *, after: float = 3) -> None:
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.space_after = Pt(after)
    paragraph.paragraph_format.line_spacing = 1.08
    label_run = paragraph.add_run(label)
    set_run_font(label_run, 10.3, bold=True, color=NAVY)
    value_run = paragraph.add_run(value)
    set_run_font(value_run, 10.3, color=BLACK)


def add_bullet(doc: Document, text: str, *, color: str = BLACK) -> None:
    paragraph = doc.add_paragraph(style="List Bullet")
    paragraph.paragraph_format.left_indent = Inches(0.5)
    paragraph.paragraph_format.first_line_indent = Inches(-0.25)
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.space_after = Pt(8)
    paragraph.paragraph_format.line_spacing = 1.167
    run = paragraph.add_run(text)
    set_run_font(run, 10.8, color=color)


def add_number(doc: Document, text: str) -> None:
    paragraph = doc.add_paragraph(style="List Number")
    paragraph.paragraph_format.left_indent = Inches(0.5)
    paragraph.paragraph_format.first_line_indent = Inches(-0.25)
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.space_after = Pt(8)
    paragraph.paragraph_format.line_spacing = 1.167
    run = paragraph.add_run(text)
    set_run_font(run, 10.8, color=BLACK)


def add_callout(doc: Document, label: str, text: str, *, fill: str = LIGHT_BLUE,
                accent: str = BLUE) -> None:
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.space_before = Pt(5)
    paragraph.paragraph_format.space_after = Pt(10)
    paragraph.paragraph_format.left_indent = Inches(0.12)
    paragraph.paragraph_format.right_indent = Inches(0.12)
    paragraph.paragraph_format.line_spacing = 1.12
    p_pr = paragraph._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    p_pr.append(shd)
    borders = OxmlElement("w:pBdr")
    left = OxmlElement("w:left")
    left.set(qn("w:val"), "single")
    left.set(qn("w:sz"), "18")
    left.set(qn("w:color"), accent)
    left.set(qn("w:space"), "8")
    borders.append(left)
    p_pr.append(borders)
    label_run = paragraph.add_run(label)
    set_run_font(label_run, 11, bold=True, color=accent)
    text_run = paragraph.add_run(text)
    set_run_font(text_run, 11, color=BLACK)


def add_heading(doc: Document, text: str, level: int = 1) -> None:
    paragraph = doc.add_paragraph(text, style=f"Heading {level}")
    paragraph.paragraph_format.keep_with_next = True


def add_page_number(paragraph) -> None:
    run = paragraph.add_run("第 ")
    set_run_font(run, 9, color=MID_GRAY)
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = "1"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, instr, separate, text, end])
    tail = paragraph.add_run(" 页")
    set_run_font(tail, 9, color=MID_GRAY)


def configure_document(doc: Document) -> None:
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)
    section.header_distance = Inches(0.492)
    section.footer_distance = Inches(0.492)

    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Calibri"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    normal.font.size = Pt(11)
    normal.font.color.rgb = RGBColor.from_string(BLACK)
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.10

    heading_tokens = {
        1: (16, BLUE, 16, 8),
        2: (13, BLUE, 12, 6),
        3: (12, DARK_BLUE, 8, 4),
    }
    for level, (size, color, before, after) in heading_tokens.items():
        style = styles[f"Heading {level}"]
        style.font.name = "Calibri"
        style._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
        style._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(color)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True

    for style_name in ("List Bullet", "List Number"):
        style = styles[style_name]
        style.font.name = "Calibri"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
        style.font.size = Pt(10.8)
        style.paragraph_format.left_indent = Inches(0.5)
        style.paragraph_format.first_line_indent = Inches(-0.25)
        style.paragraph_format.space_after = Pt(8)
        style.paragraph_format.line_spacing = 1.167

    header = section.header
    hp = header.paragraphs[0]
    hp.alignment = WD_ALIGN_PARAGRAPH.LEFT
    hp.paragraph_format.space_after = Pt(0)
    hr = hp.add_run("管道泄漏智能检测系统  |  模型方案评审")
    set_run_font(hr, 9, color=MID_GRAY)

    footer = section.footer
    fp = footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    fp.paragraph_format.space_before = Pt(0)
    add_page_number(fp)

    core = doc.core_properties
    core.title = "管道泄漏智能检测系统模型方案对比分析报告"
    core.subject = "客户评审版模型对比与部署建议"
    core.author = "项目交付团队"
    core.keywords = "YOLO, LoRA, 管道泄漏, 模型评估"


def font(size: int, bold: bool = False):
    candidates = [
        r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def make_comparison_chart(path: Path, models: list[tuple[str, dict, str]]) -> None:
    width, height = 1600, 900
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = font(38, True)
    body_font = font(24)
    small_font = font(21)
    value_font = font(20, True)
    draw.text((70, 38), "五组正式实验固定测试集指标对比", fill="#0B2545", font=title_font)
    draw.text((70, 93), "数值越高越好；纵轴从 70% 起，用于放大五组实验之间的差异", fill="#667085", font=small_font)

    metrics = [
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("mAP50", "map50"),
        ("mAP50-95", "map50_95"),
    ]
    left, right, top, bottom = 125, 60, 175, 205
    chart_w = width - left - right
    chart_h = height - top - bottom
    y_min, y_max = 0.70, 0.95
    for tick in (0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
        y = top + chart_h - (tick - y_min) / (y_max - y_min) * chart_h
        draw.line((left, y, width - right, y), fill="#D9DEE5", width=2)
        label = f"{tick * 100:.0f}%"
        bbox = draw.textbbox((0, 0), label, font=small_font)
        draw.text((left - 18 - (bbox[2] - bbox[0]), y - 13), label, fill="#667085", font=small_font)

    group_width = chart_w / len(metrics)
    bar_width = 46 if len(models) > 3 else 72
    gap = 10 if len(models) > 3 else 18
    for metric_index, (label, key) in enumerate(metrics):
        center = left + group_width * (metric_index + 0.5)
        total_bars = len(models) * bar_width + (len(models) - 1) * gap
        start_x = center - total_bars / 2
        for model_index, (_, values, color) in enumerate(models):
            value = float(values[key])
            x0 = start_x + model_index * (bar_width + gap)
            x1 = x0 + bar_width
            y0 = top + chart_h - (value - y_min) / (y_max - y_min) * chart_h
            draw.rounded_rectangle((x0, y0, x1, top + chart_h), radius=7, fill=color)
            text = f"{value * 100:.1f}"
            bbox = draw.textbbox((0, 0), text, font=value_font)
            draw.text((x0 + (bar_width - (bbox[2] - bbox[0])) / 2, y0 - 30), text, fill="#111827", font=value_font)
        bbox = draw.textbbox((0, 0), label, font=body_font)
        draw.text((center - (bbox[2] - bbox[0]) / 2, top + chart_h + 22), label, fill="#111827", font=body_font)

    legend_y = height - 105
    for index, (name, _, color) in enumerate(models):
        legend_x = 125 + (index % 3) * 500
        row_y = legend_y + (index // 3) * 45
        draw.rounded_rectangle((legend_x, row_y, legend_x + 26, row_y + 26), radius=5, fill=color)
        draw.text((legend_x + 38, row_y - 2), name, fill="#344054", font=small_font)
    image.save(path, format="PNG", dpi=(180, 180))


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def pp(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value * 100:.2f} 个百分点"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)

    v2 = read_json("models/metrics/notebook_lora_v2_internal_metrics.json")
    v3 = read_json("models/metrics/notebook_lora_v3_formal_internal_metrics.json")
    y26_full = read_json("models/metrics/yolo26s_full_768_internal_metrics.json")
    y26_post = read_json("models/metrics/yolo26s_lora_formal_internal_metrics.json")
    y26_pre = read_json("models/metrics/yolo26s_lora_from_pretrained_formal_internal_metrics.json")
    audit = read_json("models/metrics/dataset_audit.json")

    chart_path = ASSET_DIR / "top_candidate_metrics.png"
    make_comparison_chart(
        chart_path,
        [
            ("YOLO26s 全参数", y26_full, "#2E74B5"),
            ("YOLO11s LoRA v2", v3, "#1F6B45"),
            ("YOLO26s 后置 LoRA", y26_post, "#C58A1C"),
            ("YOLO11s LoRA v1", v2, "#7A5195"),
            ("YOLO26s 预训练 LoRA", y26_pre, "#D95F59"),
        ],
    )

    doc = Document()
    configure_document(doc)

    # Page 1: customer-pack opening and decision summary.
    add_paragraph(doc, "客户评审资料", size=10.5, bold=True, color=GOLD, after=7)
    add_paragraph(doc, "管道泄漏智能检测系统", size=28, bold=True, color=NAVY, after=4)
    add_paragraph(doc, "模型方案对比分析报告", size=20, bold=True, color=BLUE, after=10)
    add_paragraph(
        doc,
        "基于固定测试集的模型精度、漏检风险、推理效率与后续适配成本综合评估",
        size=12.5,
        color=MID_GRAY,
        after=22,
    )
    add_label_value(doc, "评审范围：", "YOLO26s 全参数、YOLO11s LoRA v2/v3、YOLO26s LoRA 两类方案")
    add_label_value(doc, "测试基准：", "固定测试集 251 张图像，统一检测 crack、water、leak 三类目标")
    add_label_value(doc, "报告版本：", "客户评审版 V1.0")
    add_label_value(doc, "报告日期：", "2026 年 7 月 19 日", after=15)

    add_heading(doc, "执行摘要", 1)
    add_callout(
        doc,
        "推荐结论：",
        "采用 YOLO26s 全参数模型作为当前客户演示与生产部署主模型；采用 YOLO11s LoRA v3 作为低成本领域适配和后续快速迭代方案。",
        fill=LIGHT_BLUE,
        accent=BLUE,
    )
    add_paragraph(
        doc,
        "YOLO26s 全参数模型在严格定位指标 mAP50-95 和单帧推理速度上保持第一，适合作为当前稳定交付基线。"
        "YOLO11s LoRA v3 的 Recall 略高 0.66 个百分点，mAP50 仅低 0.14 个百分点，综合性能已经非常接近主模型，同时当前实验训练耗时明显更低。",
        after=7,
    )
    add_bullet(doc, "主部署方案：YOLO26s 全参数模型，mAP50 为 92.05%，mAP50-95 为 85.47%。")
    add_bullet(doc, "低成本适配方案：YOLO11s LoRA v3，Recall 为 84.75%，高于当前全部核心候选模型。")
    add_bullet(doc, "YOLO11s LoRA v2 已由 v3 全面替代；YOLO26s 直接 LoRA 的漏检风险偏高，不建议作为当前主模型。")

    doc.add_page_break()

    # Page 2: evaluation scope, table and visual.
    add_heading(doc, "1. 测试范围与评价原则", 1)
    train_count = audit["splits"]["train"]["images"]
    valid_count = audit["splits"]["valid"]["images"]
    test_count = audit["splits"]["test"]["images"]
    class_objects = audit["splits"]["test"]["class_objects"]
    negative_images = audit["splits"]["test"]["negative_images"]
    add_paragraph(
        doc,
        f"本次对比使用同一数据处理规则。训练集 {train_count} 张、验证集 {valid_count} 张、固定测试集 {test_count} 张。"
        f"测试集包含 crack {class_objects['crack']} 个、water {class_objects['water']} 个、leak {class_objects['leak']} 个目标，"
        f"并保留 {negative_images} 张无目标图片，用于检验背景误报。",
    )
    add_paragraph(
        doc,
        "Precision 反映误报控制能力，Recall 反映漏检控制能力，mAP50 衡量常规检测精度，mAP50-95 采用更严格的定位要求。"
        "管道泄漏场景优先关注 Recall，同时结合 mAP50-95 和视频推理效率选择最终方案。",
    )

    add_heading(doc, "2. 正式模型核心指标", 1)
    source_p = add_paragraph(doc, "表 1  固定测试集统一评估结果", size=9.5, bold=True, color=MID_GRAY, before=4, after=4)
    source_p.paragraph_format.keep_with_next = True
    comparison_rows = [
        ["YOLO26s 全参数", pct(y26_full["precision"]), pct(y26_full["recall"]), pct(y26_full["map50"]), pct(y26_full["map50_95"])],
        ["YOLO11s LoRA v3", pct(v3["precision"]), pct(v3["recall"]), pct(v3["map50"]), pct(v3["map50_95"])],
        ["YOLO26s 后置 LoRA", pct(y26_post["precision"]), pct(y26_post["recall"]), pct(y26_post["map50"]), pct(y26_post["map50_95"])],
        ["YOLO11s LoRA v2", pct(v2["precision"]), pct(v2["recall"]), pct(v2["map50"]), pct(v2["map50_95"])],
        ["YOLO26s 预训练 LoRA", pct(y26_pre["precision"]), pct(y26_pre["recall"]), pct(y26_pre["map50"]), pct(y26_pre["map50_95"])],
    ]
    build_table(
        doc,
        ["模型 / 方法", "Precision", "Recall", "mAP50", "mAP50-95"],
        comparison_rows,
        [2700, 1500, 1500, 1500, 2160],
        highlight_row=0,
    )
    note = add_paragraph(doc, "注：所有指标越高越好；后置 LoRA 指在已完成的 YOLO26s 全参数模型上继续训练适配器。", size=9, color=MID_GRAY, before=4, after=7)

    doc.add_picture(str(chart_path), width=Inches(6.35))
    chart_shape = doc.inline_shapes[-1]
    chart_shape._inline.docPr.set("descr", "YOLO26s 全参数、YOLO11s LoRA v3 与 YOLO26s 后置 LoRA 的 Precision、Recall、mAP50 和 mAP50-95 对比柱状图")
    chart_shape._inline.docPr.set("title", "核心候选模型固定测试集指标")
    caption = add_paragraph(doc, "图 1  核心候选模型精度对比", size=9.5, color=MID_GRAY, align=WD_ALIGN_PARAGRAPH.CENTER, after=3)
    caption.paragraph_format.keep_with_next = False

    doc.add_page_break()

    # Page 3: method-by-method analysis.
    add_heading(doc, "3. 各模型与训练方法分析", 1)
    add_heading(doc, "3.1 YOLO26s 全参数微调", 2)
    add_paragraph(
        doc,
        "该方案更新模型全部参数，是当前综合精度最高的方案。固定测试集 mAP50-95 为 85.47%，"
        "单张图片模型推理约 6.56 ms，在本次评估中兼顾了定位精度和视频处理速度。其主要代价是训练时间和计算资源较高，"
        "本次正式训练约 5.85 小时。",
    )
    add_callout(doc, "适用定位：", "作为当前客户演示、图片检测与视频检测的主部署模型。", fill="EEF5FB", accent=BLUE)

    add_heading(doc, "3.2 YOLO11s LoRA v3", 2)
    add_paragraph(
        doc,
        "v3 采用低秩适配器并重新训练三分类检测头，估算可训练参数约占模型参数的 17.36%。"
        "其 Recall 为 84.75%，比 YOLO26s 全参数模型高 0.66 个百分点；mAP50 为 91.92%，与主模型仅差 0.14 个百分点。"
        "正式训练完成 100 个 epoch，最佳验证结果出现在第 98 个 epoch，训练过程持续收敛。",
    )
    add_paragraph(
        doc,
        "LoRA 的主要价值是后续针对新工地、新相机或新环境只需分发较小的适配器。v3 适配器约 3.79 MB；"
        "但完整独立部署权重仍约 20.41 MB，因此 LoRA 节省的是增量适配和分发成本，并不会自动缩小完整部署模型。",
    )
    add_callout(doc, "适用定位：", "作为低成本领域适配方案，并保留为主模型的高 Recall 备选。", fill="EEF8F2", accent=GREEN)

    add_heading(doc, "3.3 YOLO26s 全参数模型后追加 LoRA", 2)
    add_paragraph(
        doc,
        "该方案从已经训练完成的 YOLO26s 全参数模型继续训练 LoRA。追加训练约 0.68 小时，但 Precision、Recall、mAP50 和 mAP50-95 均未超过原始全参数模型。"
        "因此它不应被解释为精度提升方案，其更合理的用途是未来面对新场景时做小规模快速适配。",
    )

    add_heading(doc, "3.4 从官方预训练模型直接做 LoRA", 2)
    add_paragraph(
        doc,
        "YOLO26s 直接 LoRA 的 Precision 达到 93.14%，但 Recall 只有 75.93%，说明模型更保守、误报较少但漏检明显。"
        "在泄漏检测这种漏检代价较高的场景中，该权衡不符合当前业务优先级。该实验还表明，减少可训练参数不一定减少实际训练时间："
        "受训练轮次、混合精度和自定义计算路径影响，本次耗时约 6.66 小时。",
    )

    doc.add_page_break()

    # Page 4: v3 evidence and error pattern.
    add_heading(doc, "4. LoRA v3 改进效果", 1)
    add_paragraph(
        doc,
        "v3 针对 v2 的学习率偏高、提前停止过早和增强策略未充分收敛等问题，采用更低初始学习率、rank 16、较低 dropout，"
        "并在最后 40 个 epoch 关闭 mosaic。结果显示 v3 在所有核心指标和所有类别上均超过 v2。",
    )
    add_paragraph(doc, "表 2  YOLO11s LoRA v2 与 v3 对比", size=9.5, bold=True, color=MID_GRAY, before=4, after=4).paragraph_format.keep_with_next = True
    v3_v2_rows = [
        ["Precision", pct(v2["precision"]), pct(v3["precision"]), pp(v3["precision"] - v2["precision"])],
        ["Recall", pct(v2["recall"]), pct(v3["recall"]), pp(v3["recall"] - v2["recall"])],
        ["mAP50", pct(v2["map50"]), pct(v3["map50"]), pp(v3["map50"] - v2["map50"])],
        ["mAP50-95", pct(v2["map50_95"]), pct(v3["map50_95"]), pp(v3["map50_95"] - v2["map50_95"])],
    ]
    build_table(doc, ["指标", "LoRA v2", "LoRA v3", "v3 提升"], v3_v2_rows, [2500, 1900, 1900, 3060], highlight_row=1)

    add_heading(doc, "4.1 分类表现", 2)
    class_rows = []
    class_names = {"crack": "裂纹 crack", "water": "积水 water", "leak": "泄漏 leak"}
    for key in ("crack", "water", "leak"):
        old = v2["per_class_map50_95"][key]
        new = v3["per_class_map50_95"][key]
        class_rows.append([class_names[key], pct(old), pct(new), pp(new - old)])
    build_table(doc, ["目标类别", "LoRA v2", "LoRA v3", "提升"], class_rows, [2500, 1900, 1900, 3060], highlight_row=2)

    add_heading(doc, "4.2 主要误差来源", 2)
    add_paragraph(
        doc,
        "v3 的类别间混淆较少，主要剩余问题来自复杂背景上的误报，其中 leak 类占背景误报的较大比例。"
        "这说明下一阶段的收益更可能来自现场负样本覆盖、视频连续帧确认和阈值策略，而不是继续单纯增加训练轮次。",
    )
    add_bullet(doc, "按综合 F1 平衡，v3 的参考置信度约为 0.56。")
    add_bullet(doc, "若业务优先减少漏检，建议将前端置信度设置在 0.25-0.35，并通过连续帧确认抑制视频误报。")
    add_bullet(doc, "crack 测试目标仅 18 个，类别指标对少量样本变化较敏感，现场复核时应重点关注。")

    doc.add_page_break()

    # Page 5: recommendation, deployment and caveats.
    add_heading(doc, "5. 部署建议", 1)
    add_number(doc, "生产与客户演示优先部署 YOLO26s 全参数模型，作为统一图片和视频检测基线。")
    add_number(doc, "保留 YOLO11s LoRA v3 作为领域适配模型，用于新相机、新工地或环境变化后的快速更新。")
    add_number(doc, "视频检测采用逐帧或抽帧推理，并增加连续帧确认、告警合并和最短持续时间，避免单帧误报造成告警闪烁。")
    add_number(doc, "在客户现场样本上完成独立验收后，再冻结置信度阈值、IoU 阈值和最终部署权重。")

    add_heading(doc, "5.1 推荐模型组合", 2)
    recommendation_rows = [
        ["当前主模型", "YOLO26s 全参数", "综合定位精度最高、推理最快", "生产部署 / 客户演示"],
        ["适配模型", "YOLO11s LoRA v3", "Recall 略高、训练和增量分发成本低", "新场景适配 / 快速迭代"],
        ["归档模型", "YOLO11s LoRA v2", "已被 v3 全面超过", "仅保留历史对照"],
        ["研究模型", "YOLO26s 预训练 LoRA", "漏检偏高，不符合当前优先级", "方法验证"],
    ]
    build_table(doc, ["角色", "模型", "选择依据", "建议用途"], recommendation_rows, [1600, 2100, 3400, 2260], highlight_row=0)

    add_heading(doc, "5.2 上线前验收建议", 2)
    add_bullet(doc, "使用客户现场独立采集的图片和视频进行最终验收，避免只依赖现有离线测试集。")
    add_bullet(doc, "分别统计白天、夜间、反光、遮挡、积水和复杂背景条件下的 Recall 与误报率。")
    add_bullet(doc, "推理时间为模型核心耗时，不等于完整视频端到端帧率；还需计入解码、绘制、传输和存储。")
    add_bullet(doc, "mAP50-95 使用更严格的多阈值定位要求，通常低于 mAP50，不应将所有指标机械要求为同一百分比。")

    add_callout(
        doc,
        "最终建议：",
        "当前版本以 YOLO26s 全参数模型完成稳定交付，以 YOLO11s LoRA v3 建立后续低成本适配能力。两者形成“主模型 + 适配模型”的双层方案。",
        fill=LIGHT_BLUE,
        accent=BLUE,
    )

    add_heading(doc, "附录：数据口径与可追溯性", 1)
    add_paragraph(
        doc,
        "本报告指标来自项目固定 test 划分的正式评估记录。YOLO11s LoRA v3 完成 100 个 epoch 并导出 PyTorch 与 ONNX 权重；"
        "YOLO26s 全参数模型完成 139 个 epoch，最佳验证结果位于第 114 个 epoch。所有 smoke、预检和已中止实验均未纳入正式结论。",
        size=9.8,
        color=MID_GRAY,
    )
    add_paragraph(
        doc,
        "关键评估记录：notebook_lora_v3_formal_internal_metrics.json、notebook_lora_v2_internal_metrics.json、"
        "yolo26s_full_768_internal_metrics.json、yolo26s_lora_formal_internal_metrics.json、"
        "yolo26s_lora_from_pretrained_formal_internal_metrics.json。",
        size=9.2,
        color=MID_GRAY,
        after=0,
    )

    doc.save(OUTPUT_DOCX)
    print(OUTPUT_DOCX)


if __name__ == "__main__":
    main()
