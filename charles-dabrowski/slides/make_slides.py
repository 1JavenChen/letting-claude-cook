"""Build the 10-slide step-by-step project deck from results/*.json, models/*.history.json and results/cnn_v4/predictions.csv.

    .venv/bin/python slides/make_slides.py
"""
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from lxml import etree
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION, XL_MARKER_STYLE
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "slides" / "FYS_speaker_counter.pptx"

# ---------------------------------------------------------------- palette (magma-like, after the spectrogram plots)
BG = RGBColor(0xFF, 0xFF, 0xFF)
SURFACE = RGBColor(0xF3, 0xF0, 0xF9)
TEXT = RGBColor(0x1E, 0x18, 0x38)
MUTED = RGBColor(0x5E, 0x58, 0x78)
GRID = RGBColor(0xDD, 0xD8, 0xEA)
ORANGE = RGBColor(0xD9, 0x6F, 0x1A)
MAGENTA = RGBColor(0xB8, 0x2E, 0x5C)
PURPLE = RGBColor(0x6E, 0x55, 0xA8)
HEAD_FONT, BODY_FONT = "Cambria", "Calibri"

# ---------------------------------------------------------------- data
def metrics(name):
    return json.load(open(ROOT / "results" / name / "metrics.json"))

V1, V1B, V2, V3 = metrics("cnn_v1"), metrics("cnn_v1b"), metrics("cnn_v2"), metrics("cnn_v3")
V4, V4_2 = metrics("cnn_v4"), metrics("cnn_v4_on_v2test")
V5, V5_2 = metrics("cnn_v5_stft"), metrics("cnn_v5_stft_on_v2test")
H4 = json.load(open(ROOT / "models" / "cnn_v4.history.json"))
H5 = json.load(open(ROOT / "models" / "cnn_v5_stft.history.json"))
PRED = list(csv.DictReader(open(ROOT / "results/cnn_v4/predictions.csv")))
CM = np.zeros((5, 5), int)
for r in PRED:
    CM[int(r["true_n_speakers"]), int(r["pred_n_speakers"])] += 1
OFFSETS = Counter(int(r["pred_n_speakers"]) - int(r["true_n_speakers"]) for r in PRED)
V3_TRAIN = [r for r in csv.DictReader(open(ROOT / "data/mixtures_v3/labels.csv")) if r["split"] == "train"]
N_DIST = Counter(int(r["n_speakers"]) for r in V3_TRAIN)
pct = lambda x, d=1: f"{x*100:.{d}f}%"

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
SLIDE_NO = 0


# ---------------------------------------------------------------- helpers
def new_slide(notes):
    global SLIDE_NO
    SLIDE_NO += 1
    s = prs.slides.add_slide(BLANK)
    bg = s.background.fill; bg.solid(); bg.fore_color.rgb = BG
    s.notes_slide.notes_text_frame.text = notes
    text(s, 12.2, 7.0, 0.55, 0.3, str(SLIDE_NO), size=10, color=MUTED, align=PP_ALIGN.RIGHT)
    return s


def text(slide, x, y, w, h, runs, size=14, color=TEXT, bold=False, font=BODY_FONT,
         align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, space_after=0, bullets=False):
    """runs: str, or list of paragraphs; each paragraph is str or list of (text, {opts})."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    paras = runs if isinstance(runs, list) else runs.split("\n")
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space_after)
        if bullets:
            pPr = p._p.get_or_add_pPr()
            pPr.set("marL", str(int(Inches(0.22)))); pPr.set("indent", str(-int(Inches(0.22))))
            bu = etree.SubElement(pPr, qn("a:buClr")); etree.SubElement(bu, qn("a:srgbClr"), val="D96F1A")
            etree.SubElement(pPr, qn("a:buChar"), char="•")
        for seg in (para if isinstance(para, list) else [(para, {})]):
            t, o = seg if isinstance(seg, tuple) else (seg, {})
            r = p.add_run(); r.text = t
            f = r.font; f.name = o.get("font", font); f.size = Pt(o.get("size", size))
            f.bold = o.get("bold", bold); f.color.rgb = o.get("color", color)
    return tb


def title(slide, t, size=30):
    text(slide, 0.6, 0.45, 12.13, 1.15, t, size=size, font=HEAD_FONT, bold=True)


def card(slide, x, y, w, h, fill=SURFACE):
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.adjustments[0] = 0.06
    sh.fill.solid(); sh.fill.fore_color.rgb = fill; sh.line.fill.background(); sh.shadow.inherit = False
    return sh


def arrow(slide, x, y, w=0.24, h=0.28):
    ar = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(y), Inches(w), Inches(h))
    ar.fill.solid(); ar.fill.fore_color.rgb = MUTED; ar.line.fill.background(); ar.shadow.inherit = False


def labeled_card(slide, x, y, w, h, head, body, head_color=ORANGE, body_size=12, head_size=15, bullets=False, space_after=None):
    card(slide, x, y, w, h)
    text(slide, x + 0.22, y + 0.2, w - 0.44, 0.4, head, size=head_size, bold=True, color=head_color)
    text(slide, x + 0.22, y + 0.64, w - 0.44, h - 0.78, body, size=body_size, color=TEXT, bullets=bullets, space_after=(4 if bullets else 0) if space_after is None else space_after)


def style_axis(ax, fmt=None, gridlines=False, size=11):
    ax.tick_labels.font.size = Pt(size); ax.tick_labels.font.color.rgb = MUTED; ax.tick_labels.font.name = BODY_FONT
    if fmt:
        ax.tick_labels.number_format = fmt; ax.tick_labels.number_format_is_linked = False
    ax.format.line.color.rgb = GRID
    ax.has_major_gridlines = gridlines
    if gridlines:
        ax.major_gridlines.format.line.color.rgb = GRID
    ax.has_minor_gridlines = False


def chart_title(ch, t):
    ch.has_title = True
    tf = ch.chart_title.text_frame; tf.text = t
    p = tf.paragraphs[0]; p.font.size = Pt(14); p.font.bold = True; p.font.color.rgb = TEXT; p.font.name = BODY_FONT


def bar_chart(slide, x, y, w, h, cats, vals, t, colors=None, ymax=1.0, fmt="0%", unit=0.25, gap=60):
    cd = CategoryChartData(); cd.categories = cats; cd.add_series("v", vals)
    gf = slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(x), Inches(y), Inches(w), Inches(h), cd)
    ch = gf.chart; ch.has_legend = False; chart_title(ch, t); ch.font.name = BODY_FONT
    plot = ch.plots[0]; plot.gap_width = gap; plot.vary_by_categories = False
    ser = plot.series[0]; ser.format.fill.solid(); ser.format.fill.fore_color.rgb = ORANGE
    if colors:
        for i, c in enumerate(colors):
            pt = ser.points[i]; pt.format.fill.solid(); pt.format.fill.fore_color.rgb = c
    plot.has_data_labels = True
    dl = plot.data_labels; dl.number_format = fmt; dl.number_format_is_linked = False
    dl.position = XL_LABEL_POSITION.OUTSIDE_END; dl.font.size = Pt(11); dl.font.color.rgb = TEXT; dl.font.name = BODY_FONT
    va = ch.value_axis; va.minimum_scale = 0; va.maximum_scale = ymax; va.major_unit = unit
    style_axis(va, fmt, gridlines=True); va.format.line.fill.background()
    style_axis(ch.category_axis)
    return ch


def line_chart(slide, x, y, w, h, cats, series, t, ymin, ymax, unit, xlabel="epoch"):
    cd = CategoryChartData(); cd.categories = cats
    for name, vals, _ in series:
        cd.add_series(name, vals)
    gf = slide.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(x), Inches(y), Inches(w), Inches(h), cd)
    ch = gf.chart; ch.font.name = BODY_FONT; chart_title(ch, t)
    ch.has_legend = True; ch.legend.position = XL_LEGEND_POSITION.BOTTOM; ch.legend.include_in_layout = False
    ch.legend.font.size = Pt(12); ch.legend.font.color.rgb = TEXT; ch.legend.font.name = BODY_FONT
    for ser, (_, _, col) in zip(ch.plots[0].series, series):
        ser.smooth = False
        ser.format.line.color.rgb = col; ser.format.line.width = Pt(2.5)
        ser.marker.style = XL_MARKER_STYLE.CIRCLE; ser.marker.size = 5
        ser.marker.format.fill.solid(); ser.marker.format.fill.fore_color.rgb = col; ser.marker.format.line.color.rgb = col
    va = ch.value_axis; va.minimum_scale = ymin; va.maximum_scale = ymax; va.major_unit = unit
    style_axis(va, "0%", gridlines=True); va.format.line.fill.background()
    ca = ch.category_axis; style_axis(ca)
    ca.has_title = True; ca.axis_title.text_frame.text = xlabel
    p = ca.axis_title.text_frame.paragraphs[0]; p.font.size = Pt(11); p.font.color.rgb = MUTED; p.font.name = BODY_FONT; p.font.bold = False
    return ch


def set_borders(cell, color_hex="DDD8EA", width_pt=0.75):
    tcPr = cell._tc.get_or_add_tcPr()
    for tag in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
        for old in tcPr.findall(qn(tag)):
            tcPr.remove(old)
        ln = etree.SubElement(tcPr, qn(tag), w=str(int(width_pt * 12700)), cap="flat", cmpd="sng", algn="ctr")
        sf = etree.SubElement(ln, qn("a:solidFill")); etree.SubElement(sf, qn("a:srgbClr"), val=color_hex)
        etree.SubElement(ln, qn("a:prstDash"), val="solid")
    for fill in tcPr.findall(qn("a:solidFill")):   # borders must precede the fill
        tcPr.remove(fill); tcPr.append(fill)


def table(slide, x, y, col_w, row_h, rows, cell_fn):
    """rows: list of lists of str. cell_fn(i, j) -> dict(fill, color, bold, size, align)."""
    tbl = slide.shapes.add_table(len(rows), len(col_w), Inches(x), Inches(y), Inches(sum(col_w)), Inches(sum(row_h))).table
    for j, w in enumerate(col_w):
        tbl.columns[j].width = Inches(w)
    for i, h in enumerate(row_h):
        tbl.rows[i].height = Inches(h)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            o = cell_fn(i, j)
            c = tbl.cell(i, j); c.fill.solid(); c.fill.fore_color.rgb = o.get("fill", BG)
            c.margin_left = c.margin_right = Inches(0.08); c.margin_top = c.margin_bottom = Inches(0.03)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = c.text_frame; tf.word_wrap = True
            for k, ln in enumerate(str(val).split("\n")):
                p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
                p.alignment = o.get("align", PP_ALIGN.LEFT)
                r = p.add_run(); r.text = ln
                r.font.name = BODY_FONT; r.font.size = Pt(o.get("size", 12))
                r.font.bold = o.get("bold", False); r.font.color.rgb = o.get("color", TEXT)
            set_borders(c)
    tblPr = tbl._tbl.tblPr; tblPr.set("bandRow", "0"); tblPr.set("firstRow", "0")
    return tbl


def blend(a, b, t):
    return RGBColor(*(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3)))


def bullets(slide, x, y, w, h, items, size=14):
    text(slide, x, y, w, h, items, size=size, bullets=True, space_after=8)

# ================================================================ 1. title
s = new_slide("Project: a neural network that listens to 4 seconds of a noisy room and reports how many people are talking, "
              "and how many are male and female. Four steps: get data, build the input, train the network, compare two inputs.")
title(s, "Counting voices in a noisy room", size=44)
text(s, 0.6, 1.55, 12.13, 0.6, "A neural network that listens to 4 seconds of audio and reports how many people are talking, and how many are male or female",
     size=20, color=MUTED)
IMG_W = 12.13; IMG_H = IMG_W * 420 / 2400
s.shapes.add_picture(str(ROOT / "data/demo/examples.png"), Inches(0.6), Inches(2.7), Inches(IMG_W), Inches(IMG_H))
text(s, 0.6, 2.7 + IMG_H + 0.1, 12.13, 0.35, "Five generated clips as the network sees them: 0, 1, 2, 3 and 4 people talking over background noise", size=11, color=MUTED)
steps = [("1", "Data", "a public speech corpus, turned into 20,000 synthetic clips"), ("2", "Input", "each clip becomes a mel spectrogram"),
         ("3", "Model", "a small CNN trained and tested on unseen voices"), ("4", "Experiment", "mel vs plain spectrogram as input"),
         ("5", "Real data", "tested and retrained on real meeting recordings")]
SW, SG, SY = 2.3, 0.16, 5.55
for i, (n, hd, body) in enumerate(steps):
    x = 0.6 + i * (SW + SG)
    text(s, x, SY, 0.5, 0.6, n, size=30, bold=True, font=HEAD_FONT, color=ORANGE)
    text(s, x + 0.5, SY + 0.04, SW - 0.5, 0.35, hd, size=15, bold=True)
    text(s, x + 0.5, SY + 0.4, SW - 0.5, 0.7, body, size=11, color=MUTED)
text(s, 0.6, 6.85, 8, 0.4, "Karol Dabrowski  ·  First-Year Seminar  ·  October 2026", size=12, color=MUTED)

# ================================================================ 2. data
s = new_slide("Step one, data. Source: LibriSpeech, a public corpus of audiobook readings, 291 speakers each recorded alone, "
              "labelled male or female. Since it has no overlap or noise, I generated my own clips: pick 0 to 4 speakers, overlay "
              "4-second excerpts, add reverb half the time, add white, pink or babble noise at 0 to 20 dB SNR. Labels are exact by "
              "construction. 20,000 training clips, 1,500 validation, 2,000 test, with test speakers held out entirely.")
title(s, "Step 1: a public speech corpus, turned into 20,000 synthetic noisy clips")
labeled_card(s, 0.6, 1.8, 5.6, 2.45, "Source: LibriSpeech", [
    "Free public corpus of audiobook readings (LibriVox), widely used in speech research",
    "291 speakers used, 146 male / 145 female, each recorded alone in a quiet room",
    "No overlapping speech and no noise, so it cannot be used directly",
], body_size=13, bullets=True)
labeled_card(s, 0.6, 4.45, 5.6, 2.45, "Generation recipe", [
    "Pick 0 to 4 speakers; overlay a 4-second excerpt from each",
    "Add reverb with probability 0.5; add white, pink or babble noise at 0 to 20 dB SNR",
    "Labels (people, male, female) are exact by construction; test speakers are never seen in training",
], body_size=13, bullets=True)
bar_chart(s, 6.6, 1.75, 6.13, 2.5, ["training", "validation", "test (unseen speakers)"], [20000, 1500, 2000],
          "Clips generated", colors=[ORANGE, PURPLE, PURPLE], ymax=25000, fmt="#,##0", unit=5000)
bar_chart(s, 6.6, 4.4, 6.13, 2.5, ["0 (noise only)", "1", "2", "3", "4"], [N_DIST[k] / len(V3_TRAIN) for k in range(5)],
          "Share of training clips by number of speakers", ymax=0.3, fmt="0%", unit=0.1)

# ================================================================ 3. input + model
s = new_slide("Step two, input and model. Each clip is converted to a spectrogram: time left to right, frequency bottom to top, "
              "brightness is energy. I used the mel version, 64 bands spaced like human hearing. The model is a small "
              "convolutional neural network: four stages that detect patterns and halve the picture, a pooled summary, and three "
              "output heads for people, male and female, each a 5-way choice. About 580,000 parameters.")
title(s, "Step 2: each clip becomes a spectrogram, read by a small convolutional network")
bullets(s, 0.6, 1.8, 5.4, 3.3, [
    [("Spectrogram: ", {"bold": True}), ("time left to right, frequency bottom to top, brightness is energy. A voice is a stack of horizontal harmonics; two voices interleave.", {})],
    [("Mel scale: ", {"bold": True}), ("64 bands spaced like human hearing, fine at low frequencies and coarse at high. The plain version (top) keeps 128 equal bins; I compare both in step 4.", {})],
    [("Model: ", {"bold": True}), ("a CNN, the standard architecture for images. Four stages detect patterns and halve the picture, then three output heads each pick a count from 0 to 4. About 580,000 parameters.", {})],
], size=13.5)
s.shapes.add_picture(str(ROOT / "slides/assets/mel_vs_stft_light.png"), Inches(6.73), Inches(1.7), Inches(6.0), Inches(6.0 * 4.9 / 7.2))
chain = [("Spectrogram", "64 × 401"), ("Stage 1", "32 filters"), ("Stage 2", "64 filters"), ("Stage 3", "128 filters"),
         ("Stage 4", "128 filters"), ("Pooled summary", "128 numbers"), ("3 heads", "people, male, female")]
CW, CG, CY, CH = 1.55, 0.213, 6.0, 0.85
for i, (hd, body) in enumerate(chain):
    x = 0.6 + i * (CW + CG)
    card(s, x, CY, CW, CH, fill=SURFACE if i not in (0, 6) else BG)
    if i in (0, 6):
        sh = s.shapes[-1]; sh.line.color.rgb = GRID; sh.line.width = Pt(1)
    text(s, x + 0.08, CY + 0.12, CW - 0.16, 0.3, hd, size=12, bold=True, color=ORANGE, align=PP_ALIGN.CENTER)
    text(s, x + 0.08, CY + 0.46, CW - 0.16, 0.3, body, size=10.5, color=TEXT, align=PP_ALIGN.CENTER)
    if i < 6:
        arrow(s, x + CW, CY + CH / 2 - 0.1, w=0.21, h=0.2)

# ================================================================ 4. training + results
by_n = V4["count_accuracy_by_n_speakers"]
s = new_slide("Step three, training and results. Thirty passes over the 20,000 clips; after each pass the model is scored on "
              "the 1,500 validation clips and the best version is kept. Final test on 2,000 clips from unseen speakers: exact "
              "speaker count 80 percent, within one 99 percent, male and female counts 75 and 78 percent. Three simultaneous "
              "speakers is the hardest case; low SNR, reverb and babble noise hurt most.")
title(s, "Step 3: trained for 30 passes, tested on 2,000 clips from unseen speakers")
tiles = [(pct(V4["n_speakers"]["accuracy"], 0), "exact speaker count"), (pct(V4["n_speakers"]["within_1"], 0), "within one speaker"),
         (f"{pct(V4['n_male']['accuracy'], 0)} / {pct(V4['n_female']['accuracy'], 0)}", "exact male / female count")]
TX, TW, TY0, TH, TG = 0.6, 3.3, 1.8, 1.25, 0.2
for i, (big, label) in enumerate(tiles):
    y = TY0 + i * (TH + TG)
    card(s, TX, y, TW, TH)
    text(s, TX + 0.25, y + 0.12, TW - 0.5, 0.7, big, size=34 if i < 2 else 27, bold=True, font=HEAD_FONT, color=ORANGE, anchor=MSO_ANCHOR.MIDDLE)
    text(s, TX + 0.25, y + 0.82, TW - 0.5, 0.35, label, size=12, color=MUTED)
bullets(s, TX, TY0 + 3 * (TH + TG) + 0.05, TW, 1.6, [
    "30 passes over the training clips; the version with the best validation score is kept",
    "About 1.5 h per model on a laptop GPU",
], size=11.5)
bar_chart(s, 4.4, 1.65, 8.33, 2.8, ["0 (noise only)", "1 speaker", "2 speakers", "3 speakers", "4 speakers"],
          [v[0] for v in by_n.values()], "Exact-count accuracy by true number of speakers")
line_chart(s, 4.4, 4.5, 8.33, 2.45, [e["epoch"] for e in H4],
           [("training", [e["train_n_speakers"] for e in H4], PURPLE), ("validation", [e["val_n_speakers"] for e in H4], ORANGE)],
           "Exact-count accuracy per pass", 0.2, 0.9, 0.2, xlabel="pass")
text(s, 0.6, 6.98, 12.1, 0.3, "Hardest case: three simultaneous speakers (56%). Low SNR, reverb and babble noise hurt most.", size=11, color=MUTED)

# ================================================================ 5. comparison
s = new_slide("Step four, the experiment. Same data, network, settings and seed; only the input changed from the 64-band mel "
              "spectrogram to the 128-bin plain spectrogram. Mel learned faster and finished 1 to 2 points ahead on both test sets, "
              "at half the training time per pass. One run each, so the gap is suggestive rather than proven.")
title(s, "Step 4: mel vs plain spectrogram as input, everything else identical")
line_chart(s, 0.6, 1.7, 7.3, 5.2, [e["epoch"] for e in H4],
           [("mel spectrogram, 64 bands", [e["val_n_speakers"] for e in H4], ORANGE),
            ("plain spectrogram, 128 bins", [e["val_n_speakers"] for e in H5], PURPLE)],
           "Validation accuracy per pass", 0.2, 0.85, 0.1, xlabel="pass")
rows = [["", "mel", "plain"],
        ["input size", "64 × 401", "128 × 401"],
        ["exact count, 2,000-clip test", pct(V4["n_speakers"]["accuracy"]), pct(V5["n_speakers"]["accuracy"])],
        ["exact count, 1,000-clip test", pct(V4_2["n_speakers"]["accuracy"]), pct(V5_2["n_speakers"]["accuracy"])],
        ["within one speaker", pct(V4["n_speakers"]["within_1"]), pct(V5["n_speakers"]["within_1"])],
        ["time per pass", "165 s", "~320 s"]]
def mv_cell(i, j):
    o = {"size": 12 if j == 0 else 13, "align": PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER}
    if i == 0 or j == 0: o.update(fill=SURFACE, bold=(i == 0))
    else: o.update(color=ORANGE if j == 1 else PURPLE, bold=i in (2, 3))
    return o
row_h = [0.5] + [0.48] * 5
table(s, 8.2, 1.75, [2.45, 1.04, 1.04], row_h, rows, mv_cell)
bullets(s, 8.2, 1.75 + sum(row_h) + 0.3, 4.53, 2.3, [
    [("Only the input changed: ", {"bold": True}), ("same clips, network, settings and random seed.", {})],
    [("Mel is 1 to 2 points better ", {"bold": True}), ("and trains twice as fast, since its input is half the size.", {})],
    [("One run each, ", {"bold": True}), ("so the gap is suggestive, not proven.", {})],
], size=12.5)

# ================================================================ 6. real data
def ami(name):
    return json.load(open(ROOT / "results" / name / "metrics.json"))
A4, A6, A7 = ami("cnn_v4_on_ami_test"), ami("cnn_v6_ami_on_ami_test"), ami("cnn_v7_ft_on_ami_test")
import csv as _csv
_rows = list(_csv.DictReader(open(ROOT / "results/cnn_v4_on_ami_test/predictions.csv")))
_t = np.array([int(r["true_n_speakers"]) for r in _rows]); BASE_EXACT, BASE_W1 = float(np.mean(_t == 1)), float(np.mean(abs(_t - 1) <= 1))
s = new_slide("Step five, real recordings. The synthetic clips felt artificial, so I tested on the AMI Meeting Corpus: about 100 "
              "hours of real four-person meetings with time-stamped speaker turns and each participant's sex, which gives exact "
              "labels per 4-second window. On 8,000 real windows from unseen meetings, the synthetic-only model gets just 25 "
              "percent exact: it hears most real speech as silence, because far-field room audio is quieter and more echoey "
              "than anything it trained on. Retraining the same network on 42,000 real windows lifts that to 68 percent exact "
              "and 97 percent within one. The comparison that matters is against always answering one, which already scores 60 "
              "percent because real meetings are mostly one person at a time. Two and four simultaneous speakers remain hard.")
title(s, "Step 5: on real meeting recordings the synthetic model collapses; retraining on real data recovers it")
bullets(s, 0.6, 1.8, 5.6, 2.6, [
    [("Real data: ", {"bold": True}), ("the AMI Meeting Corpus, about 100 h of real 4-person meetings with time-stamped speaker turns and participant sex. Each 4 s window gets exact labels from the turns.", {})],
    [("Setup: ", {"bold": True}), ("42,000 balanced training windows from 136 meetings (far-field table mic and headset mix); 8,000 test windows from 16 unseen meetings at the natural distribution.", {})],
    [("Same network, same recipe; ", {"bold": True}), ("only the training audio changed.", {})],
], size=13)
rows = [["model", "trained on", "exact", "within one"],
        ["cnn_v4", "synthetic clips only", pct(A4["n_speakers"]["accuracy"], 0), pct(A4["n_speakers"]["within_1"], 0)],
        ["cnn_v6", "real meetings, from scratch", pct(A6["n_speakers"]["accuracy"], 0), pct(A6["n_speakers"]["within_1"], 0)],
        ["cnn_v7", "cnn_v4 fine-tuned on real", pct(A7["n_speakers"]["accuracy"], 0), pct(A7["n_speakers"]["within_1"], 0)],
        ["baseline", "always answer \"1\"", pct(BASE_EXACT, 0), pct(BASE_W1, 0)]]
def rd_cell(i, j):
    o = {"size": 12, "align": PP_ALIGN.LEFT if j < 2 else PP_ALIGN.CENTER}
    if i == 0: o.update(fill=SURFACE, bold=True)
    elif j == 0: o.update(fill=SURFACE, bold=True, color=ORANGE if i == 2 else TEXT)
    elif j >= 2: o.update(bold=True, color=ORANGE if i == 2 else (MUTED if i == 4 else TEXT))
    else: o.update(color=MUTED)
    return o
table(s, 0.6, 4.55, [1.0, 2.5, 1.05, 1.05], [0.42] * 5, rows, rd_cell)
cd = CategoryChartData(); cd.categories = ["synthetic only", "real, scratch", "fine-tuned", "always 1"]
cd.add_series("exact", [A4["n_speakers"]["accuracy"], A6["n_speakers"]["accuracy"], A7["n_speakers"]["accuracy"], BASE_EXACT])
cd.add_series("within one", [A4["n_speakers"]["within_1"], A6["n_speakers"]["within_1"], A7["n_speakers"]["within_1"], BASE_W1])
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(6.6), Inches(1.65), Inches(6.13), Inches(3.4), cd)
ch = gf.chart; ch.font.name = BODY_FONT; chart_title(ch, "Speaker-count accuracy on 8,000 real meeting windows")
ch.has_legend = True; ch.legend.position = XL_LEGEND_POSITION.BOTTOM; ch.legend.include_in_layout = False
ch.legend.font.size = Pt(11); ch.legend.font.color.rgb = TEXT; ch.legend.font.name = BODY_FONT
plot = ch.plots[0]; plot.gap_width = 70; plot.overlap = -10
for ser, col in zip(plot.series, [ORANGE, PURPLE]):
    ser.format.fill.solid(); ser.format.fill.fore_color.rgb = col
plot.has_data_labels = True
dl = plot.data_labels; dl.number_format = "0%"; dl.number_format_is_linked = False
dl.position = XL_LABEL_POSITION.OUTSIDE_END; dl.font.size = Pt(10); dl.font.color.rgb = TEXT; dl.font.name = BODY_FONT
va = ch.value_axis; va.minimum_scale = 0; va.maximum_scale = 1.0; va.major_unit = 0.25
style_axis(va, "0%", gridlines=True); va.format.line.fill.background(); style_axis(ch.category_axis, size=10)
by6 = A6["count_accuracy_by_n_speakers"]
bullets(s, 6.6, 5.2, 6.13, 1.8, [
    [("Why cnn_v4 fails: ", {"bold": True}), ("it labels most real speech as silence. Far-field room audio is quieter and more echoey than the synthetic clips.", {})],
    [("Retrained model by true count: ", {"bold": True}), (f"0: {pct(by6['0'][0], 0)}, 1: {pct(by6['1'][0], 0)}, 2: {pct(by6['2'][0], 0)}, 3: {pct(by6['3'][0], 0)}, 4: {pct(by6['4'][0], 0)} (only 89 four-speaker windows exist).", {})],
    [("Fine-tuning from cnn_v4 did not help ", {"bold": True}), ("and that model fell to 38% on the synthetic test: it forgot the old domain.", {})],
], size=12)

# ================================================================ 6b. real data by speaker count
import collections as _collections
BYN = {m: {int(k): v for k, v in json.load(open(ROOT / "results" / f"{m}_on_ami_test/metrics.json"))["count_accuracy_by_n_speakers"].items()}
       for m in ("cnn_v6_ami", "cnn_v7_ft", "cnn_v4")}
NWIN = {n: BYN["cnn_v6_ami"][n][1] for n in range(5)}
_p4 = _collections.Counter(int(r["pred_n_speakers"]) for r in _csv.DictReader(open(ROOT / "results/cnn_v6_ami_on_ami_test/predictions.csv")) if r["true_n_speakers"] == "4")
s = new_slide("The two real-data models by number of speakers. A word on how they were trained: cnn_v6 started from random weights and saw "
              "the real windows twenty times. cnn_v7 started from cnn_v4, the synthetic model, and saw the same real windows eight times at "
              "a lower learning rate, the standard fine-tuning recipe, so it began knowing what voices look like and only had to adapt to "
              "real rooms. One speaker, the common case in meetings, is where the scratch "
              "model is clearly better: 77 versus 75 percent. Two speakers is the big weakness for both, 45 and 40 percent, "
              "usually heard as one. At three speakers the fine-tuned model is slightly ahead, 61 versus 57. Four speakers is "
              "essentially unsolved, 3 and 8 percent, but there are only 89 such windows and almost all of them are counted as "
              "three, so the models are off by one rather than lost. The synthetic-only model is below 10 percent at every count "
              "from two up. The baseline of always answering one scores 60 percent only because single-speaker windows dominate.")
title(s, "Step 5, by count: both real-data models fail on 2 and 4 speakers")
bullets(s, 0.6, 1.75, 5.6, 2.2, [
    [("Same 8,000 real test windows, ", {"bold": True}), ("split by the true number of speakers. Natural distribution, so one speaker dominates.", {})],
    [("cnn_v6, from scratch: ", {"bold": True}), ("random starting weights, 20 passes over the 42,500 real windows, learning rate 5e-4.", {})],
    [("cnn_v7, fine-tuned: ", {"bold": True}), ("starts from cnn_v4's trained weights, then 8 passes over the same real windows at a lower learning rate (2e-4), so the synthetic knowledge is adjusted rather than overwritten. All layers updated, no synthetic clips mixed in.", {})],
], size=12)
hdr = ["model", *[f"{n} speaker{'s' if n > 1 else ''}" for n in range(1, 5)]]
sub = ["windows", *[f"{NWIN[n]:,}" for n in range(1, 5)]]
rows = [hdr, sub] + [[lab, *[pct(BYN[m][n][0], 0) for n in range(1, 5)]]
                     for lab, m in [("cnn_v6 (scratch)", "cnn_v6_ami"), ("cnn_v7 (fine-tuned)", "cnn_v7_ft"), ("cnn_v4 (synthetic)", "cnn_v4")]]
def bn_cell(i, j):
    o = {"size": 12, "align": PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER}
    if i == 0: o.update(fill=SURFACE, bold=True)
    elif i == 1: o.update(fill=SURFACE, color=MUTED, size=11)
    elif j == 0: o.update(fill=SURFACE, bold=True, color=ORANGE if i == 2 else TEXT)
    else:
        best = i in (2, 3) and BYN["cnn_v6_ami" if i == 2 else "cnn_v7_ft"][j][0] > BYN["cnn_v7_ft" if i == 2 else "cnn_v6_ami"][j][0]
        o.update(bold=True, color=ORANGE if best else (MUTED if i == 4 else TEXT))
    return o
table(s, 0.6, 4.0, [1.9, 1.0, 1.0, 1.0, 1.0], [0.4] * 5, rows, bn_cell)
text(s, 0.6, 6.1, 5.6, 0.8, f"Exact count accuracy on the natural AMI test set. Orange marks the better of the two real-data models at each count. "
     f"Silent windows ({NWIN[0]}) are near 98% for both and are left out.", size=11, color=MUTED)
cd = CategoryChartData(); cd.categories = [f"{n} speaker{'s' if n > 1 else ''}" for n in range(1, 5)]
for lab, m in [("cnn_v6 (scratch)", "cnn_v6_ami"), ("cnn_v7 (fine-tuned)", "cnn_v7_ft"), ("cnn_v4 (synthetic only)", "cnn_v4")]:
    cd.add_series(lab, [BYN[m][n][0] for n in range(1, 5)])
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(6.6), Inches(1.65), Inches(6.13), Inches(3.4), cd)
ch = gf.chart; ch.font.name = BODY_FONT; chart_title(ch, "Real windows: exact accuracy by true speaker count")
ch.has_legend = True; ch.legend.position = XL_LEGEND_POSITION.BOTTOM; ch.legend.include_in_layout = False
ch.legend.font.size = Pt(11); ch.legend.font.color.rgb = TEXT; ch.legend.font.name = BODY_FONT
plot = ch.plots[0]; plot.gap_width = 70; plot.overlap = -10
for ser, col in zip(plot.series, [ORANGE, PURPLE, GRID]):
    ser.format.fill.solid(); ser.format.fill.fore_color.rgb = col
plot.has_data_labels = True
dl = plot.data_labels; dl.number_format = "0%"; dl.number_format_is_linked = False
dl.position = XL_LABEL_POSITION.OUTSIDE_END; dl.font.size = Pt(10); dl.font.color.rgb = TEXT; dl.font.name = BODY_FONT
va = ch.value_axis; va.minimum_scale = 0; va.maximum_scale = 1.0; va.major_unit = 0.25
style_axis(va, "0%", gridlines=True); va.format.line.fill.background(); style_axis(ch.category_axis, size=10)
bullets(s, 6.6, 5.2, 6.13, 1.8, [
    [("One speaker: ", {"bold": True}), (f"the scratch model wins, {pct(BYN['cnn_v6_ami'][1][0], 0)} vs {pct(BYN['cnn_v7_ft'][1][0], 0)}, and this is most of a real meeting.", {})],
    [("Two speakers is the real weakness: ", {"bold": True}), (f"{pct(BYN['cnn_v6_ami'][2][0], 0)} and {pct(BYN['cnn_v7_ft'][2][0], 0)}; most misses are heard as one person, since real overlaps are brief.", {})],
    [("Four speakers: ", {"bold": True}), (f"only {NWIN[4]} windows, and {_p4[3]} of them are counted as three. Off by one, not lost.", {})],
], size=12)

# ================================================================ 7. single-sex models
def ss(m, t):
    return json.load(open(ROOT / "results" / f"{m}_on_{t}" / "metrics.json"))["n_speakers"]
SS = {(m, t): ss(m, t) for m in ("cnn_v4", "cnn_v8_male", "cnn_v9_female") for t in ("male", "female")}
SS[("cnn_v4", "v3test")] = V4["n_speakers"]
SS[("cnn_v8_male", "v3test")] = ss("cnn_v8_male", "v3test"); SS[("cnn_v9_female", "v3test")] = ss("cnn_v9_female", "v3test")
s = new_slide("Step six, does counting get easier when all the voices are the same sex? I generated two more synthetic sets "
              "with the same recipe, one using only the male LibriSpeech speakers and one using only the female ones, 20,000 "
              "training clips each, and trained the same network on each. On its own sex's test set, each specialist beats "
              "the mixed model by about two and a half points: 82 versus 80 percent for male voices, 79 versus 77 for female. "
              "Swap the sexes and the specialists lose five to nine points, so the gain is real but modest and it does not "
              "transfer. Female voices are harder for every model: two or three overlapping female voices score about ten "
              "points below the same counts for male voices, probably because female pitch ranges overlap more.")
title(s, "Step 6: single-sex training helps a little, and only for that sex")
bullets(s, 0.6, 1.8, 5.6, 2.35, [
    [("Question: ", {"bold": True}), ("is counting easier when every voice in the clip is the same sex?", {})],
    [("Setup: ", {"bold": True}), ("two new synthetic sets with the same recipe, one from the 146 male speakers only and one from the 145 female speakers only; 20,000 training clips each, separate test speakers.", {})],
    [("Same network, same recipe; ", {"bold": True}), ("three models compared on three test sets.", {})],
], size=13)
rows = [["model", "trained on", "male test", "female test", "mixed test"]] + [
    [lab, tr, pct(SS[(m, "male")]["accuracy"], 0), pct(SS[(m, "female")]["accuracy"], 0), pct(SS[(m, "v3test")]["accuracy"], 0)]
    for lab, tr, m in [("cnn_v4", "both sexes", "cnn_v4"), ("cnn_v8", "male voices only", "cnn_v8_male"), ("cnn_v9", "female voices only", "cnn_v9_female")]]
def ss_cell(i, j):
    o = {"size": 12, "align": PP_ALIGN.LEFT if j < 2 else PP_ALIGN.CENTER}
    best = (i == 2 and j == 2) or (i == 3 and j == 3)
    if i == 0: o.update(fill=SURFACE, bold=True)
    elif j == 0: o.update(fill=SURFACE, bold=True)
    elif j >= 2: o.update(bold=True, color=ORANGE if best else TEXT)
    else: o.update(color=MUTED)
    return o
table(s, 0.6, 3.75, [0.95, 1.75, 0.95, 1.05, 1.05], [0.42] * 4, rows, ss_cell)
text(s, 0.6, 5.55, 5.6, 0.7, "Exact speaker-count accuracy, 2,000 clips per test set. Within-one accuracy is 96 to 98% for every pair.",
     size=11, color=MUTED)
cd = CategoryChartData(); cd.categories = ["male-only clips", "female-only clips", "mixed clips"]
for lab, m in [("cnn_v4 (both)", "cnn_v4"), ("male-only model", "cnn_v8_male"), ("female-only model", "cnn_v9_female")]:
    cd.add_series(lab, [SS[(m, t)]["accuracy"] for t in ("male", "female", "v3test")])
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(6.6), Inches(1.65), Inches(6.13), Inches(3.4), cd)
ch = gf.chart; ch.font.name = BODY_FONT; chart_title(ch, "Exact count accuracy by test set")
ch.has_legend = True; ch.legend.position = XL_LEGEND_POSITION.BOTTOM; ch.legend.include_in_layout = False
ch.legend.font.size = Pt(11); ch.legend.font.color.rgb = TEXT; ch.legend.font.name = BODY_FONT
plot = ch.plots[0]; plot.gap_width = 70; plot.overlap = -10
for ser, col in zip(plot.series, [MUTED, ORANGE, MAGENTA]):
    ser.format.fill.solid(); ser.format.fill.fore_color.rgb = col
plot.has_data_labels = True
dl = plot.data_labels; dl.number_format = "0%"; dl.number_format_is_linked = False
dl.position = XL_LABEL_POSITION.OUTSIDE_END; dl.font.size = Pt(10); dl.font.color.rgb = TEXT; dl.font.name = BODY_FONT
va = ch.value_axis; va.minimum_scale = 0.6; va.maximum_scale = 0.9; va.major_unit = 0.1
style_axis(va, "0%", gridlines=True); va.format.line.fill.background(); style_axis(ch.category_axis, size=10)
gm = SS[("cnn_v8_male", "male")]["accuracy"] - SS[("cnn_v4", "male")]["accuracy"]
gfm = SS[("cnn_v9_female", "female")]["accuracy"] - SS[("cnn_v4", "female")]["accuracy"]
bullets(s, 6.6, 5.2, 6.13, 1.8, [
    [("Specialists win on their own sex, ", {"bold": True}), (f"by {gm*100:.1f} points (male) and {gfm*100:.1f} points (female): a real but small gain.", {})],
    [("They do not transfer: ", {"bold": True}), ("each loses 5 to 9 points on the other sex, so one mixed model is the safer choice.", {})],
    [("Female voices are harder for every model: ", {"bold": True}), ("two or three overlapping female voices score about 10 points below the same counts for male voices.", {})],
], size=12)

# ================================================================ 8. male vs female patterns
s = new_slide("What is the difference between a male and a female voice on a spectrogram? One clean voice of each, from the "
              "LibriSpeech readings, zoomed into the lowest kilohertz. A voice is a stack of harmonics: a bottom line at the "
              "pitch and copies at every multiple of it. This man's pitch is 110 hertz, so his lines are 110 hertz apart and "
              "nine of them fit below one kilohertz. This woman's pitch is 211 hertz, so her lines are twice as far apart and "
              "only four fit. On the right is the spectrum at the dashed instant: the same thing as a row of evenly spaced "
              "peaks. That spacing is what tells the sexes apart, and it is also why two female voices are harder to count: "
              "fewer lines per voice means less evidence that a second voice is present.")
title(s, "Step 6, continued: male vs female voices on the spectrogram")
IW = 10.8; IH = IW * 7.2 / 14
s.shapes.add_picture(str(ROOT / "slides/assets/male_vs_female_light.png"), Inches((13.333 - IW) / 2), Inches(1.3), Inches(IW), Inches(IH))
text(s, 0.6, 1.3 + IH + 0.08, 12.13, 0.35, "A voice is a stack of harmonics spaced by its pitch. Male: ~110 Hz apart, 9 lines below 1 kHz. Female: ~210 Hz apart, only 4. Clean LibriSpeech audio, 100 ms analysis window.",
     size=11, color=MUTED, align=PP_ALIGN.CENTER)

# ================================================================ 9. sex composition on real data
def comp(r):
    n, m, f = int(r["true_n_speakers"]), int(r["true_n_male"]), int(r["true_n_female"])
    return "silence" if n == 0 else "all male" if m == n else "all female" if f == n else "mixed"
_r6 = list(csv.DictReader(open(ROOT / "results/cnn_v6_ami_on_ami_test/predictions.csv")))
RC = {}   # (composition, n_speakers or "all") -> (accuracy, count)
for c in ("all male", "all female", "mixed"):
    rs = [r for r in _r6 if comp(r) == c]
    RC[(c, "all")] = (np.mean([r["true_n_speakers"] == r["pred_n_speakers"] for r in rs]), len(rs))
    for n in (1, 2, 3):
        rn = [r for r in rs if int(r["true_n_speakers"]) == n]
        RC[(c, n)] = (np.mean([r["true_n_speakers"] == r["pred_n_speakers"] for r in rn]), len(rn)) if rn else (None, 0)
s = new_slide("Does the sex pattern hold on real recordings? The AMI labels say who is talking in every window, so I split the "
              "real-data model's test predictions into windows where only men talk, only women talk, or both. Overall, all-male "
              "windows score 72 percent, all-female 68 and mixed 49, but that mostly reflects how many people were talking: "
              "mixed windows always have at least two speakers. At a fixed count the picture is different. With two speakers, "
              "mixed equals all-male at 46 percent and beats all-female at 40. With three, mixed is the best of the three, "
              "though the single-sex groups have only about 70 windows each. So a man and a woman together are, if anything, "
              "slightly easier to count than two voices of the same sex, and female-only remains the hardest case on real "
              "data too, by about five points.")
title(s, "Step 6 on real data: how many talk matters more than who")
bullets(s, 0.6, 1.8, 5.6, 2.0, [
    [("Same real-data model (cnn_v6), ", {"bold": True}), ("8,000 real meeting windows split by who was talking: only men, only women, or both. No retraining.", {})],
    [("Mixed windows always contain 2 or more speakers, ", {"bold": True}), ("so compare at a fixed count, not overall.", {})],
], size=13)
hdr = ["windows", "overall", "1 speaker", "2 speakers", "3 speakers"]
rows = [hdr] + [[f"{c} ({RC[(c, 'all')][1]:,})"] + [pct(RC[(c, k)][0], 0) if RC[(c, k)][0] is not None else "n/a" for k in ("all", 1, 2, 3)]
                for c in ("all male", "all female", "mixed")]
def rc_cell(i, j):
    o = {"size": 12, "align": PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER}
    if i == 0: o.update(fill=SURFACE, bold=True)
    elif j == 0: o.update(fill=SURFACE, bold=True)
    elif j == 1: o.update(color=MUTED)
    else: o.update(bold=True, color=ORANGE if (i == 2 and j in (2, 3)) else TEXT)
    return o
table(s, 0.6, 3.4, [1.75, 0.95, 1.0, 1.05, 1.05], [0.42] * 4, rows, rc_cell)
text(s, 0.6, 5.2, 5.6, 0.7, "Exact count accuracy of cnn_v6 on the natural AMI test set. The 3-speaker single-sex groups have only 71 and 77 windows; "
     "4-speaker windows are too rare to score.", size=11, color=MUTED)
cd = CategoryChartData(); cd.categories = ["1 speaker", "2 speakers", "3 speakers"]
for c in ("all male", "all female", "mixed"):
    cd.add_series(c, [RC[(c, n)][0] for n in (1, 2, 3)])
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(6.6), Inches(1.65), Inches(6.13), Inches(3.4), cd)
ch = gf.chart; ch.font.name = BODY_FONT; chart_title(ch, "Real windows: exact accuracy by who is talking")
ch.has_legend = True; ch.legend.position = XL_LEGEND_POSITION.BOTTOM; ch.legend.include_in_layout = False
ch.legend.font.size = Pt(11); ch.legend.font.color.rgb = TEXT; ch.legend.font.name = BODY_FONT
plot = ch.plots[0]; plot.gap_width = 70; plot.overlap = -10
for ser, col in zip(plot.series, [ORANGE, MAGENTA, PURPLE]):
    ser.format.fill.solid(); ser.format.fill.fore_color.rgb = col
plot.has_data_labels = True
dl = plot.data_labels; dl.number_format = "0%"; dl.number_format_is_linked = False
dl.position = XL_LABEL_POSITION.OUTSIDE_END; dl.font.size = Pt(10); dl.font.color.rgb = TEXT; dl.font.name = BODY_FONT
va = ch.value_axis; va.minimum_scale = 0; va.maximum_scale = 1.0; va.major_unit = 0.25
style_axis(va, "0%", gridlines=True); va.format.line.fill.background(); style_axis(ch.category_axis, size=10)
bullets(s, 6.6, 5.2, 6.13, 1.8, [
    [("Female-only is hardest on real data too: ", {"bold": True}), (f"{pct(RC[('all female', 1)][0], 0)} vs {pct(RC[('all male', 1)][0], 0)} for one speaker, {pct(RC[('all female', 2)][0], 0)} vs {pct(RC[('all male', 2)][0], 0)} for two.", {})],
    [("Mixing sexes does not hurt: ", {"bold": True}), ("at 2 speakers mixed equals all-male; at 3 it is the best group. A male and a female voice sit in different pitch ranges and are easier to tell apart.", {})],
    [("Overall gaps are mostly about count: ", {"bold": True}), ("real meetings are mostly one person at a time, and single-speaker windows are the easy ones.", {})],
], size=12)

prs.save(OUT)
print("saved", OUT, "slides:", SLIDE_NO)
