"""Slides for the 2026-10-28 session only: more data, CLAP, fresh test, final comparison."""
import json, csv
from collections import Counter
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION

ROOT = Path("/Users/karoldabrowski/Desktop/College/FYS/genre")
INK, MUTED, PURPLE, ORANGE, TINT, TEAL = RGBColor(0x1F,0x1D,0x2B), RGBColor(0x6B,0x68,0x7A), RGBColor(0x4B,0x2C,0x7F), RGBColor(0xE8,0x70,0x2A), RGBColor(0xF4,0xF1,0xF9), RGBColor(0x2A,0x9D,0x8F)
HEAD, BODY = "Cambria", "Calibri"
c = json.load(open(ROOT/"results/comparison.json")); G = ["classical","pop","jazz","hiphop","rock","electronic","country","score"]
L = [g.replace("hiphop","hip-hop").replace("score","film score") for g in G]
cz_t = json.load(open(ROOT/"results/clap_larger_clap_general/metrics_test.json")); cz_f = json.load(open(ROOT/"results/fresh_test/metrics.json"))["clap_larger_clap_general"]
rows = list(csv.DictReader(open(ROOT/"data/labels.csv"))); old = list(csv.DictReader(open(ROOT/"data/labels_before_extend_backup.csv")))
cnt_new, cnt_old = Counter(r["genre"] for r in rows if r["split"]=="train"), Counter(r["genre"] for r in old if r["split"]=="train")

prs = Presentation(); prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5); blank = prs.slide_layouts[6]
def text(s, x, y, w, h, t, size=16, bold=False, color=INK, font=BODY, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0; tf.vertical_anchor = anchor
    for i, line in enumerate(t if isinstance(t, list) else [t]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.alignment = align; p.space_after = Pt(6)
        for rt, opt in (line if isinstance(line, list) else [(line, {})]):
            r = p.add_run(); r.text = rt; f = r.font; f.name = opt.get("font", font); f.size = Pt(opt.get("size", size)); f.bold = opt.get("bold", bold); f.color.rgb = opt.get("color", color)
    return tb
def title(s, t): text(s, 0.6, 0.45, 12.1, 0.9, t, size=32, bold=True, font=HEAD)
def card(s, x, y, w, h, fill=TINT):
    sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)); sh.adjustments[0] = 0.08
    sh.fill.solid(); sh.fill.fore_color.rgb = fill; sh.line.fill.background(); sh.shadow.inherit = False; return sh
def stat(s, x, y, w, big, label, color=PURPLE, h=1.5):
    card(s, x, y, w, h); text(s, x+0.25, y+0.15, w-0.5, 0.8, big, size=38, bold=True, color=color, font=HEAD); text(s, x+0.25, y+0.95, w-0.5, 0.5, label, size=12.5, color=MUTED)
def footer(s, t): text(s, 0.6, 7.0, 12.1, 0.3, t, size=10, color=MUTED)
def style_chart(ch, title_txt, size=12):
    ch.has_title = True; ch.chart_title.text_frame.text = title_txt; tp = ch.chart_title.text_frame.paragraphs[0]
    tp.runs[0].font.size = Pt(size); tp.runs[0].font.bold = True; tp.runs[0].font.name = BODY; tp.runs[0].font.color.rgb = INK
    ca = ch.category_axis; ca.tick_labels.font.size = Pt(11); ca.tick_labels.font.name = BODY; ca.tick_labels.font.color.rgb = INK; ca.format.line.fill.background(); ca.has_major_gridlines = False
    va = ch.value_axis; va.visible = False; va.has_major_gridlines = False

# ---------------------------------------------------------------- 1 what changed
s = prs.slides.add_slide(blank)
text(s, 0.6, 0.8, 12, 1.0, "A slightly better model, and a different approach", size=38, bold=True, font=HEAD)
text(s, 0.6, 1.85, 12, 0.5, "Genre classifier · 28 October 2026", size=18, color=MUTED)
v3t, v4t = c['cnn_v3_8cls']['test']['accuracy']*100, c['cnn_v4_w48']['test']['accuracy']*100
stat(s, 0.6, 2.9, 3.9, f"{v3t:.0f} → {v4t:.0f} %", "the CNN, after doubling the training data (new artists only) and widening the net", h=1.7)
stat(s, 4.7, 2.9, 3.9, f"{c['clap_head']['test']['accuracy']*100:.0f} %", "a tiny classifier on top of CLAP, a public audio model. Trained in 3 minutes.", color=TEAL, h=1.7)
stat(s, 8.8, 2.9, 3.95, f"{c['ensemble:cnn_v4_w48+clap_head']['fresh']['accuracy']*100:.0f} %", "both together, on 504 brand-new songs from artists no model had seen (was 78 %)", color=ORANGE, h=1.7)
text(s, 0.6, 5.0, 12.1, 1.2, ["Same test songs as before, so the numbers are comparable. Chance is 12.5 %.",
                               "More data helped a little. Borrowing a model that has already heard a lot of music helped more."], size=15, color=INK)
footer(s, "Numbers: results/comparison.json · test split 1 204 clips, fresh set 504 clips, 8 genres")

# ---------------------------------------------------------------- 2 the different approach
s = prs.slides.add_slide(blank); title(s, "The different approach: let CLAP do the listening")
text(s, 0.6, 1.5, 6.3, 2.2, ["CLAP (LAION) was trained on hundreds of thousands of audio clips with captions. It turns any clip into 512 numbers that describe how it sounds.",
                              "We embed each of our clips once, then train a small network (512 → 256 → 8) on those numbers with our labels. CLAP stays frozen."], size=14)
card(s, 0.6, 3.9, 3.0, 1.9); text(s, 0.8, 4.05, 2.6, 0.4, "Our CNN", size=15, bold=True, font=HEAD)
text(s, 0.8, 4.5, 2.6, 1.2, "learns what music sounds like from 23 000 clips, from scratch. 5.5 M parameters, 2 hours.", size=12, color=MUTED)
card(s, 3.9, 3.9, 3.0, 1.9); text(s, 4.1, 4.05, 2.6, 0.4, "CLAP + classifier", size=15, bold=True, font=HEAD, color=TEAL)
text(s, 4.1, 4.5, 2.6, 1.2, "already knows what music sounds like; our labels only teach it where the genre lines are. 130 k parameters, 3 minutes.", size=12, color=MUTED)
s.shapes.add_picture(str(ROOT/"results/confusion_clap_head_test.png"), Inches(7.3), Inches(1.5), height=Inches(4.4))
text(s, 0.6, 6.05, 12.1, 0.7, [[("Same mistakes as the CNN, just fewer of them: ", {"bold": True}), ("pop still leaks into rock and country, classical and film score still trade places.", {})]], size=13)
footer(s, "Zero-shot CLAP (no training, text prompts only) was also tried: 69 %, below our CNN. Not shown further.")

# ---------------------------------------------------------------- 3 where things stand
s = prs.slides.add_slide(blank); title(s, "Where things stand")
names = [("cnn_v3_8cls", "CNN v3\n(before)"), ("cnn_v4_w48", "CNN v4\n(2x data)"), ("clap_head", "CLAP +\nclassifier"), ("ensemble:cnn_v4_w48+clap_head", "both\ntogether")]
cd = CategoryChartData(); cd.categories = [n for _, n in names]
cd.add_series("test split (1 204 clips)", [round(c[k]["test"]["accuracy"], 3) for k, _ in names]); cd.add_series("fresh songs (504)", [round(c[k]["fresh"]["accuracy"], 3) for k, _ in names])
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.6), Inches(1.45), Inches(7.4), Inches(5.0), cd); ch = gf.chart; style_chart(ch, "Accuracy, 8 genres (chance 12.5 %)")
ch.has_legend = True; ch.legend.position = XL_LEGEND_POSITION.TOP; ch.legend.include_in_layout = False; ch.legend.font.size = Pt(11); ch.legend.font.name = BODY
pl = ch.plots[0]; pl.gap_width = 80; pl.overlap = -5; pl.has_data_labels = True; dl = pl.data_labels; dl.font.size = Pt(11); dl.font.name = BODY; dl.position = XL_LABEL_POSITION.OUTSIDE_END; dl.number_format = '0%'; dl.number_format_is_linked = False
for ser, col in zip(pl.series, (PURPLE, ORANGE)): ser.format.fill.solid(); ser.format.fill.fore_color.rgb = col
ch.value_axis.minimum_scale = 0.6; ch.value_axis.maximum_scale = 0.9; ch.category_axis.tick_labels.font.size = Pt(11)
items = [("More data: +2 points", "Useful, not dramatic. The curve says 100k clips would reach the low 80s."),
         ("Pretrained model: +4 points", "Three minutes of training beat two hours from scratch."),
         ("Combining them: +0.3", "They get the same songs wrong, the genuinely ambiguous ones."),
         ("Next", "Fine-tune CLAP itself, clean up the pop labels, try full songs.")]
y = 1.5
for h, b in items:
    card(s, 8.3, y, 4.45, 1.15); text(s, 8.5, y+0.12, 4.1, 0.35, h, size=14, bold=True, font=HEAD); text(s, 8.5, y+0.5, 4.1, 0.6, b, size=11.5, color=MUTED); y += 1.27
footer(s, "Demo app updated: compare mode (CNN vs CLAP classifier) and an ensemble option · report: genre/report/final_report.pdf")

out = ROOT/"slides/today_2026-10-28.pptx"; prs.save(out); print("saved", out)
