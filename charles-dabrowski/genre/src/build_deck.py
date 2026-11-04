import csv, json
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_TICK_LABEL_POSITION

ROOT = Path("/Users/karoldabrowski/Desktop/College/FYS/genre")
INK, MUTED, PURPLE, ORANGE, TINT = RGBColor(0x1F,0x1D,0x2B), RGBColor(0x6B,0x68,0x7A), RGBColor(0x4B,0x2C,0x7F), RGBColor(0xE8,0x70,0x2A), RGBColor(0xF4,0xF1,0xF9)
HEAD, BODY = "Cambria", "Calibri"

rows = list(csv.DictReader(open(ROOT/"data/labels.csv")))
n_clips, n_artists = len(rows), len({r["artist_id"] for r in rows})
m2 = json.load(open(ROOT/"results/cnn_v2/metrics_test.json")); m3 = json.load(open(ROOT/"results/cnn_v3_8cls/metrics_test.json"))
cl = json.load(open(ROOT/"results/cnn_v3_8cls/acc_vs_cliplen.json"))
import numpy as np
cm = np.array(m3["confusion"], float); cm = cm / cm.sum(1, keepdims=True)
G8 = ["classical","pop","jazz","hiphop","rock","electronic","country","score"]

prs = Presentation(); prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
blank = prs.slide_layouts[6]

def text(slide, x, y, w, h, s, size=16, bold=False, color=INK, font=BODY, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, italic=False):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)); tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0; tf.vertical_anchor = anchor
    lines = s if isinstance(s, list) else [s]
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.alignment = align
        runs = line if isinstance(line, list) else [(line, {})]
        for rt, opt in runs:
            r = p.add_run(); r.text = rt; f = r.font; f.name = opt.get("font", font); f.size = Pt(opt.get("size", size))
            f.bold = opt.get("bold", bold); f.italic = opt.get("italic", italic); f.color.rgb = opt.get("color", color)
        p.space_after = Pt(6)
    return tb

def title(slide, s):
    text(slide, 0.6, 0.45, 12.1, 0.9, s, size=34, bold=True, font=HEAD)

def card(slide, x, y, w, h, fill=TINT):
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.adjustments[0] = 0.08; sh.fill.solid(); sh.fill.fore_color.rgb = fill; sh.line.fill.background(); sh.shadow.inherit = False
    return sh

def stat(slide, x, y, w, big, label, color=PURPLE):
    card(slide, x, y, w, 1.55)
    text(slide, x+0.25, y+0.15, w-0.5, 0.85, big, size=40, bold=True, color=color, font=HEAD)
    text(slide, x+0.25, y+1.0, w-0.5, 0.45, label, size=13, color=MUTED)

def circle_num(slide, x, y, n):
    c = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(0.5), Inches(0.5)); c.fill.solid(); c.fill.fore_color.rgb = PURPLE; c.line.fill.background()
    tf = c.text_frame; tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0; p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = str(n); r.font.size = Pt(14); r.font.bold = True; r.font.color.rgb = RGBColor(255,255,255); r.font.name = BODY

def footer(slide, s):
    text(slide, 0.6, 7.0, 12.1, 0.3, s, size=10, color=MUTED)

# ------------------------------------------------------------------ slide 1: title + headline numbers
s = prs.slides.add_slide(blank)
text(s, 0.6, 0.7, 12, 1.2, "Which genre is this song?", size=48, bold=True, font=HEAD)
text(s, 0.6, 1.75, 12, 0.6, "A convolutional net that reads the spectrogram of a 30-second clip and names the genre", size=20, color=MUTED)
text(s, 0.6, 2.3, 12, 0.4, "FYS · Karol Dąbrowski · 21 October 2026", size=14, color=MUTED)
stat(s, 0.6, 3.0, 2.9, f"{n_clips:,}".replace(",", " "), "30 s clips scraped from Deezer")
stat(s, 3.7, 3.0, 2.9, "8", "genres, from classical to film score")
stat(s, 6.8, 3.0, 2.9, f"{n_artists:,}".replace(",", " "), "artists, split so test artists are never seen in training")
stat(s, 9.9, 3.0, 2.85, f"{m3['accuracy']*100:.0f} %", "test accuracy across 8 genres (chance = 12.5 %)", color=ORANGE)
s.shapes.add_picture(str(ROOT/"results/dataset/typical_spectrograms.png"), Inches(0.6), Inches(4.8), width=Inches(12.1))
footer(s, "The most typical clip per genre (closest to the genre's average spectrum), same colour scale. Classical and film score are sparse and quiet, pop / rock / hip-hop are dense walls of sound, electronic has hard-edged blocks and a steady beat.")

# ------------------------------------------------------------------ slide 2: method, mostly figures
s = prs.slides.add_slide(blank); title(s, "Method: a 30-second preview in, a genre out")
steps = [("Collect", "Deezer API · album genre = label"), ("Represent", "log-mel spectrogram, 128 × 646"), ("Train", "CNN, 2.5 M params, 10 s crops"), ("Test", "artists never seen in training")]
for k, (h, b) in enumerate(steps):
    x = 0.6 + k * 3.075
    card(s, x, 1.35, 2.9, 0.85)
    circle_num(s, x + 0.15, 1.52, k + 1)
    text(s, x + 0.78, 1.42, 2.05, 0.35, h, size=15, bold=True, font=HEAD)
    text(s, x + 0.78, 1.78, 2.05, 0.35, b, size=11, color=MUTED)
pic = s.shapes.add_picture(str(ROOT/"results/dataset/pipeline_example.png"), Inches(0.6), Inches(2.35), width=Inches(12.1))
if pic.height > Inches(2.7): pic.height = Inches(2.7); pic.width = int(pic.height * 2161 / 649); pic.left = int((Inches(13.333) - pic.width) / 2)
# bottom left: clips per genre (native bar chart)
from collections import Counter
cnt = Counter(r["genre"] for r in rows)
cd = CategoryChartData(); cd.categories = [g.replace("hiphop","hip-hop").replace("score","film score") for g in G8]; cd.add_series("clips", [cnt[g] for g in G8])
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.6), Inches(5.15), Inches(5.9), Inches(1.8), cd); ch = gf.chart
ch.has_legend = False; ch.has_title = True; ch.chart_title.text_frame.text = f"Training data: {n_clips:,} clips from {n_artists:,} artists".replace(",", " ")
tp = ch.chart_title.text_frame.paragraphs[0]; tp.runs[0].font.size = Pt(12); tp.runs[0].font.bold = True; tp.runs[0].font.name = BODY; tp.runs[0].font.color.rgb = INK
pl = ch.plots[0]; pl.gap_width = 50; pl.has_data_labels = True; dl = pl.data_labels; dl.font.size = Pt(9); dl.font.name = BODY; dl.font.color.rgb = INK; dl.position = XL_LABEL_POSITION.OUTSIDE_END; dl.number_format = '# ##0'; dl.number_format_is_linked = False
ser = pl.series[0]; ser.format.fill.solid(); ser.format.fill.fore_color.rgb = PURPLE
pt = ser.points[7]; pt.format.fill.solid(); pt.format.fill.fore_color.rgb = ORANGE
ca = ch.category_axis; ca.tick_labels.font.size = Pt(9.5); ca.tick_labels.font.name = BODY; ca.tick_labels.font.color.rgb = INK; ca.format.line.fill.background(); ca.has_major_gridlines = False
va = ch.value_axis; va.visible = False; va.has_major_gridlines = False; va.minimum_scale = 0; va.maximum_scale = 2100
# bottom right: accuracy vs clip length (native line chart)
secs = ["2", "3", "5", "10", "15", "30"]
cd = CategoryChartData(); cd.categories = [f"{v} s" for v in secs]; cd.add_series("accuracy", [round(cl[v], 3) for v in secs])
gf = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(6.8), Inches(5.15), Inches(5.9), Inches(1.8), cd); ch = gf.chart
ch.has_legend = False; ch.has_title = True; ch.chart_title.text_frame.text = "How much audio does it need? Test accuracy vs clip length"
tp = ch.chart_title.text_frame.paragraphs[0]; tp.runs[0].font.size = Pt(12); tp.runs[0].font.bold = True; tp.runs[0].font.name = BODY; tp.runs[0].font.color.rgb = INK
pl = ch.plots[0]; pl.has_data_labels = True; dl = pl.data_labels; dl.font.size = Pt(9); dl.font.name = BODY; dl.font.color.rgb = INK; dl.position = XL_LABEL_POSITION.ABOVE; dl.number_format = '0%'; dl.number_format_is_linked = False
ser = pl.series[0]; ser.format.line.color.rgb = PURPLE; ser.format.line.width = Pt(2.25); ser.smooth = False
ser.marker.style = 8; ser.marker.size = 7; ser.marker.format.fill.solid(); ser.marker.format.fill.fore_color.rgb = ORANGE; ser.marker.format.line.color.rgb = ORANGE
ca = ch.category_axis; ca.tick_labels.font.size = Pt(9.5); ca.tick_labels.font.name = BODY; ca.tick_labels.font.color.rgb = INK; ca.format.line.color.rgb = RGBColor(0xC9,0xC3,0xD8); ca.has_major_gridlines = False
va = ch.value_axis; va.visible = False; va.has_major_gridlines = False; va.minimum_scale = 0.4; va.maximum_scale = 0.9
footer(s, "Split by artist, 80 / 10 / 10 per genre. Trained with SpecAugment masking, AdamW and a one-cycle schedule: 40 epochs on 7 genres, then 9 more after adding film score.")

# ------------------------------------------------------------------ slide 3: results
s = prs.slides.add_slide(blank); title(s, f"{m3['accuracy']*100:.0f} % of songs by unseen artists land in the right genre")
cd = CategoryChartData(); cd.categories = [g.replace("hiphop","hip-hop").replace("score","film score") for g in G8]; cd.add_series("F1", [round(m3["f1"][g], 2) for g in G8])
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.6), Inches(1.5), Inches(6.2), Inches(4.3), cd); ch = gf.chart
ch.has_legend = False; ch.has_title = True; ch.chart_title.text_frame.text = f"Per-genre F1 score on the test set (n = {m3['n']:,}".replace(",", " ") + ")"
tp = ch.chart_title.text_frame.paragraphs[0]; tp.runs[0].font.size = Pt(14); tp.runs[0].font.bold = True; tp.runs[0].font.name = BODY; tp.runs[0].font.color.rgb = INK
pl = ch.plots[0]; pl.gap_width = 70; pl.has_data_labels = True; dl = pl.data_labels; dl.font.size = Pt(11); dl.font.name = BODY; dl.font.color.rgb = INK; dl.position = XL_LABEL_POSITION.OUTSIDE_END; dl.number_format = '0.00'; dl.number_format_is_linked = False
ser = pl.series[0]; ser.format.fill.solid(); ser.format.fill.fore_color.rgb = PURPLE
for k in (1, 7):
    pt = ser.points[k]; pt.format.fill.solid(); pt.format.fill.fore_color.rgb = ORANGE
ca = ch.category_axis; ca.tick_labels.font.size = Pt(11); ca.tick_labels.font.name = BODY; ca.tick_labels.font.color.rgb = INK; ca.format.line.fill.background(); ca.has_major_gridlines = False
va = ch.value_axis; va.visible = False; va.has_major_gridlines = False; va.minimum_scale = 0; va.maximum_scale = 1.05
s.shapes.add_picture(str(ROOT/"results/cnn_v3_8cls/confusion_test.png"), Inches(7.3), Inches(1.45), height=Inches(4.4))
card(s, 0.6, 5.95, 6.2, 0.95)
text(s, 0.85, 6.05, 5.8, 0.8, [[("Pop is the catch-all. ", {"bold": True}), (f"{cm[1,4]*100:.0f} % of pop songs are called rock and {cm[1,6]*100:.0f} % country; rock, hip-hop and electronic each leak {cm[5,1]*100:.0f}–{cm[4,1]*100:.0f} % into pop. Country, hip-hop and jazz are the easy ones.", {})]], size=12.5)
card(s, 7.3, 5.95, 5.45, 0.95)
text(s, 7.55, 6.05, 5.0, 0.8, [[("Classical vs film score is the hard pair. ", {"bold": True}), (f"{cm[0,7]*100:.0f} % of classical clips are called a film score and {cm[7,0]*100:.0f} % of scores classical; scores also drift to electronic ({cm[7,5]*100:.0f} %). Same orchestra, different production.", {})]], size=12.5)
footer(s, f"Confusion matrix: rows = true genre, columns = prediction, normalised per row. Chance = 12.5 %. Accuracy vs clip length: 2 s {cl['2']*100:.0f} %, 5 s {cl['5']*100:.0f} %, 10 s {cl['10']*100:.0f} %, 30 s {cl['30']*100:.0f} %.")

out = ROOT/"slides/genre_cnn.pptx"; prs.save(out); print("saved", out)
