# generate_presentation.py
# Run: pip install python-pptx && python generate_presentation.py
# Output: management_presentation.pptx

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
import copy

# ── Colour Palette ──────────────────────────────────────────────────────────
GREEN_DARK   = RGBColor(0x1B, 0x5E, 0x20)   # deep forest green  #1B5E20
GREEN_MID    = RGBColor(0x2E, 0x7D, 0x32)   # medium green       #2E7D32
GREEN_LIGHT  = RGBColor(0x43, 0xA0, 0x47)   # accent green       #43A047
GREEN_PALE   = RGBColor(0xE8, 0xF5, 0xE9)   # slide background   #E8F5E9
WHITE        = RGBColor(0xFF, 0xFF, 0xFF)
CHARCOAL     = RGBColor(0x21, 0x21, 0x21)
AMBER        = RGBColor(0xFF, 0x8F, 0x00)   # highlight/warning  #FF8F00
LIGHT_GRAY   = RGBColor(0xF5, 0xF5, 0xF5)
MID_GRAY     = RGBColor(0xBD, 0xBD, 0xBD)

prs = Presentation()
prs.slide_width  = Inches(13.33)
prs.slide_height = Inches(7.5)

BLANK = prs.slide_layouts[6]   # completely blank layout

# ── Helper utilities ─────────────────────────────────────────────────────────

def add_rect(slide, l, t, w, h, fill_color=None, line_color=None, line_width=Pt(0)):
    shape = slide.shapes.add_shape(1, Inches(l), Inches(t), Inches(w), Inches(h))
    shape.line.width = line_width
    if fill_color:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill_color
    else:
        shape.fill.background()
    if line_color:
        shape.line.color.rgb = line_color
    else:
        shape.line.fill.background()
    return shape


def add_textbox(slide, l, t, w, h, text, font_size=Pt(12), bold=False,
                color=CHARCOAL, align=PP_ALIGN.LEFT, wrap=True,
                font_name="Calibri", italic=False):
    txb = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    txb.word_wrap = wrap
    tf = txb.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = font_size
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    run.font.name = font_name
    return txb


def header_bar(slide, title, subtitle=None):
    """Dark green top bar with white title."""
    add_rect(slide, 0, 0, 13.33, 1.35, fill_color=GREEN_DARK)
    add_textbox(slide, 0.3, 0.12, 12.7, 0.75, title,
                font_size=Pt(30), bold=True, color=WHITE,
                align=PP_ALIGN.LEFT, font_name="Calibri")
    if subtitle:
        add_textbox(slide, 0.3, 0.82, 12.7, 0.42, subtitle,
                    font_size=Pt(14), bold=False, color=GREEN_PALE,
                    align=PP_ALIGN.LEFT, font_name="Calibri", italic=True)


def slide_bg(slide):
    """Pale green full-slide background."""
    add_rect(slide, 0, 0, 13.33, 7.5, fill_color=GREEN_PALE)


def footer_bar(slide, slide_num, total=16):
    add_rect(slide, 0, 7.1, 13.33, 0.4, fill_color=GREEN_MID)
    add_textbox(slide, 0.2, 7.12, 8, 0.28,
                "AI-Powered Plant Disease Detection  |  PlantVillage Project",
                font_size=Pt(9), color=WHITE, font_name="Calibri")
    add_textbox(slide, 11.5, 7.12, 1.6, 0.28,
                f"{slide_num} / {total}",
                font_size=Pt(9), color=WHITE,
                align=PP_ALIGN.RIGHT, font_name="Calibri")


def bullet_block(slide, l, t, w, h, items, font_size=Pt(13),
                 color=CHARCOAL, bullet_color=GREEN_MID, indent=0.25):
    """Render a list of strings as bullet points inside a textbox."""
    txb = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    txb.word_wrap = True
    tf = txb.text_frame
    tf.word_wrap = True
    first = True
    for item in items:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.space_before = Pt(4)
        bullet_run = p.add_run()
        bullet_run.text = "● "
        bullet_run.font.size = font_size
        bullet_run.font.color.rgb = bullet_color
        bullet_run.font.name = "Calibri"
        text_run = p.add_run()
        text_run.text = item
        text_run.font.size = font_size
        text_run.font.color.rgb = color
        text_run.font.name = "Calibri"
    return txb


def section_label(slide, l, t, w, text, bg=GREEN_LIGHT, fg=WHITE):
    add_rect(slide, l, t, w, 0.32, fill_color=bg)
    add_textbox(slide, l + 0.1, t + 0.03, w - 0.2, 0.28, text,
                font_size=Pt(11), bold=True, color=fg, font_name="Calibri")


def table_block(slide, l, t, col_widths, headers, rows,
                header_fill=GREEN_MID, row_fill=WHITE,
                alt_fill=GREEN_PALE, font_size=Pt(11)):
    """Draw a simple styled table using rectangles and textboxes."""
    row_h = 0.32
    # header row
    x = l
    for i, (hdr, cw) in enumerate(zip(headers, col_widths)):
        add_rect(slide, x, t, cw, row_h, fill_color=header_fill,
                 line_color=WHITE, line_width=Pt(1))
        add_textbox(slide, x + 0.05, t + 0.04, cw - 0.1, row_h - 0.06,
                    hdr, font_size=font_size, bold=True,
                    color=WHITE, font_name="Calibri", align=PP_ALIGN.CENTER)
        x += cw
    # data rows
    for ri, row in enumerate(rows):
        y = t + row_h * (ri + 1)
        fill = alt_fill if ri % 2 == 0 else row_fill
        x = l
        for ci, (cell, cw) in enumerate(zip(row, col_widths)):
            add_rect(slide, x, y, cw, row_h, fill_color=fill,
                     line_color=MID_GRAY, line_width=Pt(0.5))
            bold_cell = (ci == 0)
            add_textbox(slide, x + 0.05, y + 0.04, cw - 0.1, row_h - 0.06,
                        str(cell), font_size=font_size, bold=bold_cell,
                        color=CHARCOAL, font_name="Calibri",
                        align=PP_ALIGN.CENTER)
            x += cw


def stat_card(slide, l, t, w, h, value, label, val_color=GREEN_DARK):
    add_rect(slide, l, t, w, h, fill_color=WHITE,
             line_color=GREEN_LIGHT, line_width=Pt(1.5))
    add_textbox(slide, l + 0.1, t + 0.12, w - 0.2, h * 0.55,
                value, font_size=Pt(26), bold=True,
                color=val_color, align=PP_ALIGN.CENTER, font_name="Calibri")
    add_textbox(slide, l + 0.05, t + h * 0.58, w - 0.1, h * 0.38,
                label, font_size=Pt(10), bold=False,
                color=CHARCOAL, align=PP_ALIGN.CENTER, font_name="Calibri")


# ════════════════════════════════════════════════════════════════════════════
# SLIDE 1 — Title / Cover
# ════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(BLANK)
add_rect(slide, 0, 0, 13.33, 7.5, fill_color=GREEN_DARK)
add_rect(slide, 0, 4.8, 13.33, 2.7, fill_color=GREEN_MID)

# decorative accent bars
add_rect(slide, 0, 4.72, 13.33, 0.12, fill_color=GREEN_LIGHT)
add_rect(slide, 0, 4.58, 13.33, 0.08, fill_color=AMBER)

add_textbox(slide, 0.6, 0.8, 12, 0.6,
            "AI-POWERED PLANT DISEASE DETECTION",
            font_size=Pt(13), bold=True, color=GREEN_LIGHT,
            font_name="Calibri", align=PP_ALIGN.CENTER)
add_textbox(slide, 0.6, 1.35, 12, 1.6,
            "Putting a Plant Doctor in Every Farmer's Pocket",
            font_size=Pt(38), bold=True, color=WHITE,
            font_name="Calibri", align=PP_ALIGN.CENTER)
add_textbox(slide, 0.6, 2.95, 12, 0.55,
            "PlantVillage Dataset  |  FastAPI + PyTorch + MobileNetV2",
            font_size=Pt(15), bold=False, color=GREEN_PALE,
            font_name="Calibri", italic=True, align=PP_ALIGN.CENTER)

add_textbox(slide, 0.6, 5.05, 12, 0.45,
            "Dr. Amish Jain  |  Ayman Rabaya  |  Shannon Coutinho  |  Aayush Sharma",
            font_size=Pt(14), bold=True, color=WHITE,
            font_name="Calibri", align=PP_ALIGN.CENTER)
add_textbox(slide, 0.6, 5.55, 12, 0.38,
            "Computer Vision Final Project  —  2026",
            font_size=Pt(12), bold=False, color=GREEN_PALE,
            font_name="Calibri", align=PP_ALIGN.CENTER)
add_textbox(slide, 0.6, 6.1, 12, 0.38,
            "99.76% Accuracy  |  38 Disease Classes  |  Mobile-First Deployment",
            font_size=Pt(12), bold=False, color=AMBER,
            font_name="Calibri", align=PP_ALIGN.CENTER)
footer_bar(slide, 1)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 2 — Executive Summary
# ════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(BLANK)
slide_bg(slide)
header_bar(slide, "Executive Summary", "Why This Matters")
footer_bar(slide, 2)

# 4 KPI cards
cards = [
    ("10–40%",  "Annual crop yield lost\nto plant diseases"),
    ("38",       "Disease classes\ndetected & classified"),
    ("99.76%",  "Test accuracy\n(MobileNetV2)"),
    ("< 0.5s",  "Inference time\nper image"),
]
for i, (val, lbl) in enumerate(cards):
    stat_card(slide, 0.35 + i * 3.15, 1.55, 2.9, 1.35, val, lbl)

add_textbox(slide, 0.35, 3.05, 12.6, 0.35,
            "SOLUTION  →  IMPACT",
            font_size=Pt(11), bold=True, color=GREEN_DARK, font_name="Calibri")
add_rect(slide, 0.35, 3.4, 12.63, 0.04, fill_color=GREEN_LIGHT)

items = [
    "Business Problem:  Smallholder farmers suffer avoidable crop losses due to delayed disease "
    "diagnosis — no accessible, fast expert tool exists in the field.",
    "Our Solution:  Mobile web AI prototype that classifies plant leaf diseases from a smartphone "
    "photo in seconds, with per-leaf explainability via Grad-CAM visual heatmaps.",
    "Key Result:  MobileNetV2 achieves 99.76% accuracy and 0.9950 macro-F1 across 38 PlantVillage "
    "disease/healthy classes — no expert intermediary required.",
    "Business Impact:  Faster triage, better treatment timing, reduced crop loss risk, and a "
    "scalable advisory workflow deployable to any smartphone browser.",
]
bullet_block(slide, 0.35, 3.5, 12.6, 3.3, items, font_size=Pt(12.5))

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 3 — Team Introduction
# ════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(BLANK)
slide_bg(slide)
header_bar(slide, "Our Team", "Roles & Responsibilities")
footer_bar(slide, 3)

team = [
    ("Dr. Amish Jain",    "ML Architecture & Model Strategy",
     "Model design, training pipeline, architecture comparison & selection"),
    ("Ayman Rabaya",      "Backend API & Deployment",
     "FastAPI development, inference service, analytics logging, deployment"),
    ("Shannon Coutinho",  "Frontend & Demo",
     "Mobile UI, camera/upload integration, demo recording & narration"),
    ("Aayush Sharma",     "Documentation & Reporting",
     "Technical synopsis, Jupyter Book, presentation, compliance mapping"),
]
col_w = [2.4, 3.0, 6.9]
table_block(slide, 0.35, 1.5,
            col_w,
            ["Team Member", "Role Stream", "Responsibilities"],
            [[n, r, d] for n, r, d in team],
            font_size=Pt(11.5))

add_textbox(slide, 0.35, 4.0, 12.6, 0.38,
            "★  All team members present at least one section of this deck.",
            font_size=Pt(13), bold=True, color=GREEN_DARK,
            font_name="Calibri", align=PP_ALIGN.CENTER)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 4 — Problem Statement
# ════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(BLANK)
slide_bg(slide)
header_bar(slide, "The Problem", "Delayed Diagnosis Costs Farmers")
footer_bar(slide, 4)

# pain flow boxes
steps = [
    ("👁️  Farmer\nSpots Symptom", GREEN_MID),
    ("📅  Days /\nWeeks Waiting", AMBER),
    ("🔬  Expert\nUnavailable", RGBColor(0xC6, 0x28, 0x28)),
    ("💸  Treatment\nToo Late", RGBColor(0xB7, 0x1C, 0x1C)),
    ("📉  Yield\nLoss", RGBColor(0x7F, 0x00, 0x00)),
]
for i, (lbl, col) in enumerate(steps):
    add_rect(slide, 0.35 + i * 2.52, 1.55, 2.2, 1.15, fill_color=col)
    add_textbox(slide, 0.35 + i * 2.52 + 0.05, 1.6, 2.1, 1.05,
                lbl, font_size=Pt(12), bold=True, color=WHITE,
                font_name="Calibri", align=PP_ALIGN.CENTER)
    if i < 4:
        add_textbox(slide, 0.35 + i * 2.52 + 2.2, 1.9, 0.32, 0.55,
                    "→", font_size=Pt(22), bold=True,
                    color=GREEN_DARK, font_name="Calibri")

items = [
    "800M+ smallholder farmers globally depend on crops as their primary livelihood source.",
    "Plant diseases are responsible for an estimated 10–40% annual crop yield loss worldwide.",
    "Expert agronomists are scarce, expensive, and simply unavailable in remote field conditions.",
    "By the time visible symptoms are obvious, disease spread is often already beyond early-stage control.",
    "No fast, affordable, and accessible tool currently exists for in-field disease identification at scale.",
]
bullet_block(slide, 0.35, 2.85, 12.6, 4.0, items, font_size=Pt(13))

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 5 — Scope & Success Criteria
# ════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(BLANK)
slide_bg(slide)
header_bar(slide, "Project Scope & Success Criteria", "Version 1 MVP Definition")
footer_bar(slide, 5)

# IN SCOPE
add_rect(slide, 0.35, 1.5, 6.0, 0.38, fill_color=GREEN_MID)
add_textbox(slide, 0.45, 1.53, 5.8, 0.32, "✅  IN SCOPE — Version 1",
            font_size=Pt(12), bold=True, color=WHITE, font_name="Calibri")
in_scope = [
    "Mobile browser-based camera capture & file upload (no app install)",
    "Leaf presence validation before inference",
    "Multi-leaf segmentation — up to 5 leaves per image",
    "Per-leaf independent classification (Crop + Disease label)",
    "Confidence-gated output — retake if model is uncertain",
    "Grad-CAM visual explainability per accepted leaf",
    "PlantVillage dataset  |  38 classes  |  English language",
]
bullet_block(slide, 0.35, 1.92, 6.0, 4.5, in_scope, font_size=Pt(12))

# OUT OF SCOPE
add_rect(slide, 6.7, 1.5, 6.3, 0.38, fill_color=RGBColor(0xC6, 0x28, 0x28))
add_textbox(slide, 6.8, 1.53, 6.1, 0.32, "🚫  OUT OF SCOPE — Version 1",
            font_size=Pt(12), bold=True, color=WHITE, font_name="Calibri")
out_scope = [
    "Native mobile app / offline PWA mode",
    "Non-English language support",
    "Real-field (non-PlantVillage) training data",
    "Treatment recommendation engine",
    "Cloud autoscaling / production hardening",
]
bullet_block(slide, 6.7, 1.92, 6.3, 3.5, out_scope,
             font_size=Pt(12), bullet_color=RGBColor(0xC6, 0x28, 0x28))

add_rect(slide, 0.35, 5.85, 12.63, 0.38, fill_color=GREEN_DARK)
add_textbox(slide, 0.45, 5.88, 12.4, 0.32,
            "SUCCESS CRITERIA:  >95% test accuracy  |  Functional multi-leaf pipeline  |  "
            "Live API with Grad-CAM explainability",
            font_size=Pt(11), bold=True, color=WHITE, font_name="Calibri",
            align=PP_ALIGN.CENTER)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 6 — Data Profile
# ════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(BLANK)
slide_bg(slide)
header_bar(slide, "Data Profile", "PlantVillage Dataset")
footer_bar(slide, 6)

table_block(slide, 0.35, 1.5,
            [3.0, 9.3],
            ["Property", "Value"],
            [
                ["Source",           "PlantVillage (open benchmark dataset)"],
                ["Total Classes",    "38 (crop + disease combinations)"],
                ["Label Format",     "Crop___Disease  (e.g. Tomato___Late_blight)"],
                ["Healthy Label",    "Healthy — No disease detected"],
                ["Image Format",     "JPEG, folder-per-class structure"],
                ["Split Strategy",   "Stratified  80% train  /  10% validation  /  10% test"],
                ["Imbalance Handling", "Class-weighted cross-entropy loss"],
                ["Preprocessing",   "Resize 384×384  |  ImageNet normalisation  |  Standard augmentation"],
            ],
            font_size=Pt(12))

add_textbox(slide, 0.35, 5.4, 12.6, 0.35,
            "Sample classes:  Tomato Late Blight  |  Apple Scab  |  Corn Common Rust  |  "
            "Potato Early Blight  |  Grape Black Rot  |  Pepper Bacterial Spot",
            font_size=Pt(11.5), bold=False, color=GREEN_DARK, font_name="Calibri",
            italic=True)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 7 — EDA
# ════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(BLANK)
slide_bg(slide)
header_bar(slide, "Exploratory Data Analysis", "What the Data Told Us")
footer_bar(slide, 7)

eda_items = [
    ("Class Imbalance Detected",
     "Significant variation in sample counts across 38 classes was found → directly drove the "
     "adoption of class-weighted cross-entropy loss rather than standard unweighted loss."),
    ("Conservative Augmentation Chosen",
     "Augmentation limited to horizontal/vertical flip, rotation, brightness jitter, and scale "
     "only. Aggressive augmentation risked unrealistic artifacts given PlantVillage's "
     "controlled, lab-quality capture conditions."),
    ("Domain Shift Risk Identified Early",
     "PlantVillage images are lab-captured with clean backgrounds and controlled lighting. "
     "Real field photos have variable lighting, angles, and background clutter → this finding "
     "directly drove the confidence thresholding and retake-guidance design."),
    ("Resolution Standardised at 384×384",
     "Chosen to balance input image quality against inference latency on mobile-targeted "
     "deployment. Larger sizes improved feature detail; smaller sizes hurt fine-grained "
     "lesion recognition."),
]
y = 1.5
for title, body in eda_items:
    add_rect(slide, 0.35, y, 0.18, 0.38, fill_color=GREEN_LIGHT)
    add_textbox(slide, 0.62, y, 12.3, 0.28, title,
                font_size=Pt(12.5), bold=True, color=GREEN_DARK, font_name="Calibri")
    add_textbox(slide, 0.62, y + 0.3, 12.3, 0.55, body,
                font_size=Pt(11.5), color=CHARCOAL, font_name="Calibri")
    y += 1.12

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 8 — System Architecture
# ════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(BLANK)
slide_bg(slide)
header_bar(slide, "Technology Platform & System Architecture", "How the System Works")
footer_bar(slide, 8)

layers = [
    ("📱  Client Layer",
     "Mobile Browser (HTML/CSS/JS)",
     "Camera capture, image preview, compression, upload, result rendering",
     GREEN_LIGHT),
    ("⚡  API Layer",
     "FastAPI (Python)",
     "GET /health  |  POST /infer  |  GET /analytics/summary",
     GREEN_MID),
    ("🧠  CV / ML Services",
     "PyTorch + OpenCV + torchvision",
     "Leaf validation  →  Segmentation  →  Classification  →  Grad-CAM  →  Disease info",
     GREEN_DARK),
    ("💾  Storage & Logging",
     "File system + JSONL analytics",
     "Uploaded images  |  Grad-CAM outputs  |  Inference events  |  Application logs",
     RGBColor(0x1A, 0x23, 0x7E)),
]
for i, (layer, tech, resp, col) in enumerate(layers):
    add_rect(slide, 0.35, 1.5 + i * 1.3, 12.63, 1.15, fill_color=col)
    add_textbox(slide, 0.5, 1.53 + i * 1.3, 3.0, 0.38, layer,
                font_size=Pt(12), bold=True, color=WHITE, font_name="Calibri")
    add_textbox(slide, 0.5, 1.88 + i * 1.3, 3.0, 0.32, tech,
                font_size=Pt(11), color=GREEN_PALE, font_name="Calibri", italic=True)
    add_rect(slide, 3.55, 1.5 + i * 1.3, 0.04, 1.15, fill_color=WHITE)
    add_textbox(slide, 3.7, 1.6 + i * 1.3, 9.1, 0.75, resp,
                font_size=Pt(12), color=WHITE, font_name="Calibri")

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 9 — CV Pipeline Methodology
# ════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(BLANK)
slide_bg(slide)
header_bar(slide, "Methodology", "End-to-End Computer Vision Pipeline")
footer_bar(slide, 9)

steps = [
    ("1", "📱 Capture /\nUpload",    "Farmer uses mobile browser to capture or upload leaf photo",           GREEN_MID),
    ("2", "🔍 Leaf\nValidation",     "HSV green-ratio heuristic checks leaf presence; rejects non-leaf inputs", GREEN_MID),
    ("3", "✂️ Multi-Leaf\nSegment",  "OpenCV threshold + contour detection isolates up to 5 individual leaves", GREEN_MID),
    ("4", "🧠 Per-Leaf\nClassify",   "MobileNetV2 classifies each leaf independently as Crop + Disease label",  GREEN_DARK),
    ("5", "📊 Confidence\nFilter",   "Predictions below 70% threshold trigger retake rather than forced guess", AMBER),
    ("6", "🌡️ Grad-CAM\nHeatmap",   "Visual heatmap highlights leaf regions that drove the prediction",        GREEN_DARK),
    ("7", "📋 Structured\nResponse", "Per-leaf results, confidence, description, and heatmap returned to user", GREEN_MID),
]
box_w = 1.7
for i, (num, title, desc, col) in enumerate(steps):
    x = 0.2 + i * 1.87
    add_rect(slide, x, 1.5, box_w, 0.95, fill_color=col)
    add_textbox(slide, x + 0.05, 1.53, box_w - 0.1, 0.9,
                title, font_size=Pt(10), bold=True, color=WHITE,
                font_name="Calibri", align=PP_ALIGN.CENTER)
    if i < 6:
        add_textbox(slide, x + box_w, 1.85, 0.17, 0.38,
                    "→", font_size=Pt(14), bold=True,
                    color=GREEN_DARK, font_name="Calibri")
    add_textbox(slide, x, 2.55, box_w, 0.95, desc,
                font_size=Pt(9.5), color=CHARCOAL, font_name="Calibri",
                align=PP_ALIGN.CENTER)

add_textbox(slide, 0.2, 3.6, 12.9, 0.32,
            "4 Integrated CV Techniques:  Leaf Validation  |  Classical Segmentation  |  "
            "Deep Learning Classification  |  Grad-CAM Explainability",
            font_size=Pt(12), bold=True, color=GREEN_DARK,
            font_name="Calibri", align=PP_ALIGN.CENTER)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 10 — Model Strategy & Training
# ════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(BLANK)
slide_bg(slide)
header_bar(slide, "Model Strategy & Training Approach", "How We Built and Selected Our Model")
footer_bar(slide, 10)

# Model cards
models = [
    ("CNN Baseline",  "Custom from scratch",
     ["Lightweight benchmark", "Fast inference (127ms)", "No pretrained weights", "Smallest size (1.79 MB)"],
     GREEN_LIGHT, CHARCOAL),
    ("MobileNetV2\n✅ SELECTED",  "ImageNet pretrained +\nfine-tuned head",
     ["Best Accuracy 99.76%", "Best macro-F1  0.9950", "8.90 MB — deployable", "Transfer learning benefit"],
     GREEN_DARK, WHITE),
    ("Hybrid Model",  "MobileNetV2 +\nresidual branch fusion",
     ["Strong 99.48% accuracy", "Largest size (18.35 MB)", "Slowest inference", "Comparison experiment"],
     GREEN_MID, WHITE),
]
for i, (name, strategy, pts, bg, fg) in enumerate(models):
    x = 0.35 + i * 4.3
    add_rect(slide, x, 1.5, 3.95, 3.5, fill_color=bg,
             line_color=GREEN_LIGHT, line_width=Pt(1.5))
    add_textbox(slide, x + 0.1, 1.55, 3.75, 0.55, name,
                font_size=Pt(13), bold=True, color=fg,
                font_name="Calibri", align=PP_ALIGN.CENTER)
    add_textbox(slide, x + 0.1, 2.1, 3.75, 0.45, strategy,
                font_size=Pt(11), color=fg, font_name="Calibri",
                italic=True, align=PP_ALIGN.CENTER)
    add_rect(slide, x + 0.1, 2.55, 3.75, 0.04, fill_color=WHITE)
    y = 2.65
    for pt in pts:
        add_textbox(slide, x + 0.2, y, 3.55, 0.35,
                    "• " + pt, font_size=Pt(11), color=fg, font_name="Calibri")
        y += 0.38

add_textbox(slide, 0.35, 5.1, 12.6, 0.35,
            "TRAINING POLICY:  Adam Optimizer  |  Max 50 Epochs  |  Early Stopping (patience=8)  |  "
            "Best-Checkpoint Restore  |  Class-Weighted Loss",
            font_size=Pt(11.5), bold=True, color=GREEN_DARK,
            font_name="Calibri", align=PP_ALIGN.CENTER)

table_block(slide, 0.35, 5.55,
            [2.55, 2.1, 2.1, 2.1, 2.1, 2.08],
            ["Hyperparameter", "CNN", "MobileNetV2", "Hybrid", "—", "—"],
            [
                ["Image Size",    "384×384", "384×384", "384×384", "", ""],
                ["Batch Size",    "16",      "16",      "12",      "", ""],
                ["Learning Rate", "1e-3",    "1e-4",    "1e-4",    "", ""],
            ],
            font_size=Pt(10.5))

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 11 — Literature Review
# ════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(BLANK)
slide_bg(slide)
header_bar(slide, "Literature Review & Benchmark Positioning", "Building on Established Research")
footer_bar(slide, 11)

refs = [
    ("Mohanty et al. (2016)",
     "Using Deep Learning for Image-Based Plant Disease Detection",
     "Pioneering PlantVillage deep learning study achieving ~99% accuracy under controlled "
     "conditions — established the benchmark this project targets."),
    ("Howard et al. (2017)",
     "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications",
     "Introduced MobileNetV2 as an efficient, lightweight architecture for mobile vision tasks "
     "— our primary deployment model."),
    ("Selvaraju et al. (2017)",
     "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization",
     "Proposed Grad-CAM for visual explainability in CNNs — directly adopted in our inference "
     "pipeline for per-leaf heatmap generation."),
    ("Our Contribution",
     "End-to-End Deployable Prototype",
     "Multi-leaf segmentation + transfer learning classification + Grad-CAM explainability in "
     "a single mobile API — with practical confidence thresholding for field-safe inference."),
]
for i, (author, title, body) in enumerate(refs):
    y = 1.5 + i * 1.35
    col = GREEN_DARK if i == 3 else GREEN_MID
    add_rect(slide, 0.35, y, 0.2, 0.8, fill_color=col)
    add_textbox(slide, 0.65, y, 12.0, 0.32, author,
                font_size=Pt(12), bold=True, color=GREEN_DARK, font_name="Calibri")
    add_textbox(slide, 0.65, y + 0.3, 12.0, 0.28, title,
                font_size=Pt(11), bold=False, color=CHARCOAL,
                font_name="Calibri", italic=True)
    add_textbox(slide, 0.65, y + 0.57, 12.0, 0.55, body,
                font_size=Pt(11), color=CHARCOAL, font_name="Calibri")

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 12 — Results & Model Selection
# ════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(BLANK)
slide_bg(slide)
header_bar(slide, "Results & Model Selection", "MobileNetV2 Delivers Best Performance")
footer_bar(slide, 12)

table_block(slide, 0.35, 1.5,
            [2.5, 1.6, 1.6, 1.6, 1.6, 1.7, 1.63],
            ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "Speed (ms)", "Size (MB)"],
            [
                ["CNN Baseline",    "98.78%", "98.44%", "98.82%", "98.61%", "127.8", "1.79"],
                ["MobileNetV2 ✅",  "99.76%", "99.38%", "99.65%", "99.50%", "201.4", "8.90"],
                ["Hybrid",          "99.48%", "98.99%", "99.39%", "99.17%", "206.4", "18.35"],
            ],
            font_size=Pt(12))

add_textbox(slide, 0.35, 3.65, 12.6, 0.38,
            "WHY MobileNetV2 WAS SELECTED FOR DEPLOYMENT",
            font_size=Pt(12), bold=True, color=GREEN_DARK, font_name="Calibri")

reasons = [
    "Highest observed Accuracy (99.76%) and macro-F1 (0.9950) among all three compared models.",
    "Better deployability than Hybrid — 2× smaller model size (8.90 MB vs 18.35 MB) with comparable quality.",
    "Slightly faster benchmarked inference (201 ms) than Hybrid (206 ms) in recorded evaluation runs.",
    "Best production trade-off of quality, speed, and footprint for mobile-first field deployment.",
]
bullet_block(slide, 0.35, 4.08, 12.6, 3.0, reasons, font_size=Pt(12.5))

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 13 — Demo Evidence
# ════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(BLANK)
slide_bg(slide)
header_bar(slide, "Demo Evidence", "The System in Action — Four Scenarios")
footer_bar(slide, 13)

scenarios = [
    ("✅", "Healthy Leaf",       GREEN_MID,
     "No disease detected\nConfidence shown\nGrad-CAM rendered"),
    ("🦠", "Diseased Leaf",      RGBColor(0xC6, 0x28, 0x28),
     "Crop + Disease label\nConfidence ≥ 70%\nGrad-CAM heatmap"),
    ("🌿", "Multi-Leaf Mixed",   RGBColor(0x1A, 0x23, 0x7E),
     "Per-leaf results\nMixed healthy / diseased\nUp to 5 leaves"),
    ("❌", "Invalid Input",      CHARCOAL,
     '"No leaf detected.\nPlease retake\nthe photo."'),
]
for i, (icon, title, col, output) in enumerate(scenarios):
    x = 0.35 + i * 3.18
    add_rect(slide, x, 1.5, 2.9, 1.0, fill_color=col)
    add_textbox(slide, x + 0.1, 1.55, 2.7, 0.42,
                icon + "  " + title, font_size=Pt(13), bold=True,
                color=WHITE, font_name="Calibri", align=PP_ALIGN.CENTER)
    add_textbox(slide, x + 0.1, 1.93, 2.7, 0.52, output,
                font_size=Pt(11), color=WHITE, font_name="Calibri",
                align=PP_ALIGN.CENTER)

add_textbox(slide, 0.35, 2.65, 12.6, 0.32,
            "KEY FEATURE HIGHLIGHTS",
            font_size=Pt(12), bold=True, color=GREEN_DARK, font_name="Calibri")

highlights = [
    "Grad-CAM heatmaps show exactly which part of the leaf drove the diagnosis — "
    "decision-support evidence visible to farmers and extension officers.",
    "Confidence scores (0–100%) give farmers a reliability signal before acting "
    "on a recommendation — transparent uncertainty quantification.",
    "Edge case handling prevents silent misclassification: the system asks for a "
    "better photo rather than returning a wrong answer at low confidence.",
    "Multi-leaf support processes up to 5 leaves per image independently — useful "
    "for batch field triage of multiple plants in a single photo.",
]
bullet_block(slide, 0.35, 3.02, 12.6, 3.8, highlights, font_size=Pt(12.5))

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 14 — Risks, Limitations & Mitigations
# ════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(BLANK)
slide_bg(slide)
header_bar(slide, "Risks, Limitations & Mitigations",
           "Honest Assessment — We Built the System to Fail Safely")
footer_bar(slide, 14)

table_block(slide, 0.35, 1.5,
            [3.8, 1.5, 7.48],
            ["Risk / Limitation", "Severity", "Mitigation"],
            [
                ["PlantVillage → Field Domain Shift",
                 "🔴 HIGH",
                 "Confidence thresholding + retake guidance; field data collection roadmap for V2"],
                ["Classical Segmentation Failure\n(clutter / heavy shadow)",
                 "🟡 MED",
                 "Confidence filtering rejects uncertain outputs; V2 roadmap: deep segmentation (SAM)"],
                ["Limited to 38 PlantVillage Classes",
                 "🟡 MED",
                 "Architecture is fully extensible — adding classes requires retraining only"],
                ["No Offline / Native App Support",
                 "🟡 MED",
                 "Progressive Web App (PWA) packaging planned for V2 sprint"],
                ["Single Language (English)",
                 "🟢 LOW",
                 "Multilingual advisory (Swahili, Hindi, French) planned for V2"],
            ],
            font_size=Pt(11.5))

add_rect(slide, 0.35, 5.88, 12.63, 0.42, fill_color=GREEN_DARK)
add_textbox(slide, 0.45, 5.91, 12.4, 0.32,
            "KEY MESSAGE:  When uncertain, the system asks for a better photo "
            "rather than guessing wrong — safety-first design for field use.",
            font_size=Pt(12), bold=True, color=WHITE,
            font_name="Calibri", align=PP_ALIGN.CENTER)

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 15 — Next Steps & Path to MVP
# ════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(BLANK)
slide_bg(slide)
header_bar(slide, "Next Steps & Path to Production MVP", "From Prototype to Field-Ready Product")
footer_bar(slide, 15)

phases = [
    ("Phase 1\n0 – 3 Months", GREEN_MID, [
        "Collect & label real-field photos",
        "Fine-tune MobileNetV2 on field data",
        "Replace classical with deep segmentation (SAM)",
        "Extension officer usability pilot",
    ]),
    ("Phase 2\n3 – 6 Months", GREEN_DARK, [
        "Package as PWA for offline mobile use",
        "Multilingual support (Swahili, Hindi, French)",
        "Treatment recommendation engine",
        "Cloud deployment with autoscaling",
    ]),
    ("Phase 3\n6 – 12 Months", RGBColor(0x1A, 0x23, 0x7E), [
        "Expand to additional crop/disease classes",
        "On-device model optimisation (TFLite/ONNX)",
        "Integration with agricultural advisory systems",
        "Monitored production rollout",
    ]),
]
for i, (title, col, items) in enumerate(phases):
    x = 0.35 + i * 4.3
    add_rect(slide, x, 1.5, 3.95, 0.7, fill_color=col)
    add_textbox(slide, x + 0.1, 1.55, 3.75, 0.6, title,
                font_size=Pt(13), bold=True, color=WHITE,
                font_name="Calibri", align=PP_ALIGN.CENTER)
    y = 2.3
    for item in items:
        add_textbox(slide, x + 0.15, y, 3.65, 0.35,
                    "→  " + item, font_size=Pt(11.5),
                    color=CHARCOAL, font_name="Calibri")
        y += 0.42

add_textbox(slide, 0.35, 4.9, 12.6, 0.32,
            "RESOURCE REQUIREMENTS",
            font_size=Pt(12), bold=True, color=GREEN_DARK, font_name="Calibri")

resources = [
    "Field data collection: partnership with agricultural NGOs or government extension services for labelled real-world images.",
    "Cloud deployment: minimal infrastructure cost for MVP scale — standard cloud VM or container service sufficient.",
    "Team: 1–2 ML engineers for fine-tuning and domain adaptation sprint; 1 mobile developer for PWA packaging.",
]
bullet_block(slide, 0.35, 5.28, 12.6, 1.9, resources, font_size=Pt(12))

# ════════════════════════════════════════════════════════════════════════════
# SLIDE 16 — Appendix A: Technical Backup
# ════════════════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(BLANK)
slide_bg(slide)
header_bar(slide, "Appendix A — Technical Backup", "Hyperparameters, Architecture & Config")
footer_bar(slide, 16)

table_block(slide, 0.35, 1.5,
            [3.3, 2.5, 2.5, 2.5, 2.08],
            ["Hyperparameter", "CNN", "MobileNetV2", "Hybrid", "Notes"],
            [
                ["Image Size",        "384×384",  "384×384",  "384×384",  "Configurable"],
                ["Batch Size",        "16",        "16",        "12",        "GPU memory fit"],
                ["Learning Rate",     "1e-3",      "1e-4",      "1e-4",      "Adam optimiser"],
                ["Max Epochs",        "50",        "50",        "50",        "Early stop applies"],
                ["Early Stop Patience","8",         "8",         "8",         "Val loss monitor"],
                ["Model Size (MB)",   "1.79",      "8.90",      "18.35",     "Deployment weight"],
                ["Inference (ms)",    "127.8",     "201.4",     "206.4",     "CPU benchmark"],
            ],
            font_size=Pt(11.5))

add_textbox(slide, 0.35, 5.25, 12.6, 0.28,
            "Confidence Threshold: 0.70 (configurable via .env)   |   "
            "Max Leaves per Image: 5   |   "
            "Leaf Validation: HSV green-ratio heuristic",
            font_size=Pt(11.5), bold=False, color=GREEN_DARK,
            font_name="Calibri", italic=True)

add_textbox(slide, 0.35, 5.65, 12.6, 0.28,
            "Grad-CAM Target: Final conv block of MobileNetV2   |   "
            "Segmentation: OpenCV threshold + contour detection   |   "
            "Deployment: FastAPI + uvicorn",
            font_size=Pt(11.5), bold=False, color=GREEN_DARK,
            font_name="Calibri", italic=True)

table_block(slide, 0.35, 6.05,
            [3.3, 9.68],
            ["Endpoint", "Description"],
            [
                ["GET  /health",            "Service health check — returns model loaded status"],
                ["POST /infer",             "Multipart image upload → full CV pipeline → per-leaf JSON response"],
                ["GET  /analytics/summary", "Aggregated inference stats: count, avg confidence, class frequency"],
            ],
            font_size=Pt(11.5))

# ════════════════════════════════════════════════════════════════════════════
# SAVE
# ════════════════════════════════════════════════════════════════════════════
output_path = "management_presentation.pptx"
prs.save(output_path)
print(f"✅  Presentation saved → {output_path}")
print(f"    Slides: {len(prs.slides)}")
