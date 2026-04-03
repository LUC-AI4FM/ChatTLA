"""Build the 3-minute version of the talk (6 slides, ~30s each)."""
from pptx import Presentation
from pptx.util import Pt, Emu
from pptx.dml.color import RGBColor

MAROON = RGBColor(90, 7, 34)
GOLD = RGBColor(234, 170, 0)
GRAY_DARK = RGBColor(84, 83, 91)
WHITE = RGBColor(255, 255, 255)

TEMPLATE = "eric-talk/loy-powerpoint-16x9-websafe.pptx"
OUTPUT = "eric-talk/talk_3min.pptx"

prs = Presentation(TEMPLATE)

# Delete all example slides (iterate in reverse to preserve indices)
while len(prs.slides) > 0:
    rId = prs.slides._sldIdLst[0].get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')
    prs.part.drop_rel(rId)
    prs.slides._sldIdLst.remove(prs.slides._sldIdLst[0])


def get_layout(name):
    for layout in prs.slide_layouts:
        if layout.name == name:
            return layout
    raise ValueError(f"Layout '{name}' not found")


def set_placeholder(slide, idx, text, font_size=None, bold=None, color=None):
    """Set text on a placeholder by index."""
    ph = slide.placeholders[idx]
    ph.text = text
    if font_size or bold is not None or color:
        for para in ph.text_frame.paragraphs:
            for run in para.runs:
                if font_size:
                    run.font.size = Pt(font_size)
                if bold is not None:
                    run.font.bold = bold
                if color:
                    run.font.color.rgb = color


def add_bullet_text(slide, idx, items, font_size=14, color=None):
    """Replace placeholder text with bullet points."""
    ph = slide.placeholders[idx]
    tf = ph.text_frame
    tf.clear()
    for i, item in enumerate(items):
        if i == 0:
            para = tf.paragraphs[0]
        else:
            para = tf.add_paragraph()
        para.text = item
        para.level = 0
        para.space_after = Pt(6)
        for run in para.runs:
            run.font.size = Pt(font_size)
            if color:
                run.font.color.rgb = color


# ── Slide 1: Title (1a_title-text) ──
slide = prs.slides.add_slide(get_layout("1a_title-text"))
set_placeholder(slide, 15, "LOYOLA UNIVERSITY CHICAGO", font_size=11)
set_placeholder(slide, 11, "Can LLMs Write Correct TLA+ Specifications?", font_size=28, bold=True)
set_placeholder(slide, 12, "Evaluating Natural-Language-to-TLA+ Generation", font_size=14)
set_placeholder(slide, 14, "AI4FM Research Group  |  Department of Computer Science", font_size=9)

# ── Slide 2: What is TLA+? (5a_text-basic) ──
slide = prs.slides.add_slide(get_layout("5a_text-basic"))
set_placeholder(slide, 11, "What Is TLA+?", font_size=28, bold=True)
set_placeholder(slide, 12, "A mathematical language for proving systems correct", font_size=13)
add_bullet_text(slide, 10, [
    "A formal specification language -- describe system rules precisely enough for a computer to check them",
    "Used at Amazon (DynamoDB, S3) and Microsoft Azure to catch bugs that testing misses",
    "Two tools validate specs:  SANY checks syntax,  TLC checks correctness",
    "Problem: writing TLA+ requires rare expertise, limiting adoption",
    "Our question: can AI bridge the gap between English and TLA+?",
], font_size=14)

# ── Slide 3: Types of AI Models (5c_text-3col) ──
slide = prs.slides.add_slide(get_layout("5c_text-3col"))
set_placeholder(slide, 11, "Three Kinds of AI Models", font_size=28, bold=True)

set_placeholder(slide, 22, "Basic Models", font_size=13, bold=True)
add_bullet_text(slide, 10, [
    "LLaMA, Mistral, Gemma",
    "Trained on internet text & code",
    "Good at familiar languages",
    "Struggle with TLA+ syntax",
], font_size=12)

set_placeholder(slide, 23, "Reasoning Models", font_size=13, bold=True)
add_bullet_text(slide, 20, [
    "DeepSeek R1, QwQ",
    "Think step-by-step",
    "Best at logical structure",
    "Top performers overall",
], font_size=12)

set_placeholder(slide, 24, "Code-Specialized", font_size=13, bold=True)
add_bullet_text(slide, 18, [
    "Qwen-Coder, CodeLLaMA",
    "Trained on Python, JS, etc.",
    "Surprisingly worst at TLA+",
    "Negative transfer effect",
], font_size=12)

# ── Slide 4: Key Results (7a_emphasis-maroon) ──
slide = prs.slides.add_slide(get_layout("7a_emphasis-maroon"))
set_placeholder(slide, 11, "Key Results", font_size=28, bold=True)
set_placeholder(slide, 22, "30 models  |  205 specifications  |  2,730 evaluation runs", font_size=12)
add_bullet_text(slide, 10, [
    "Best syntactic correctness (SANY):  26.6%",
    "Best semantic correctness (TLC):  8.6%",
    "Semantic success only under progressive prompting -- no other strategy worked",
    "Model size does not predict quality -- 8B model beat its 70B variant",
    "Code-specialized training hurts TLA+ performance (negative transfer)",
], font_size=15, color=WHITE)

# ── Slide 5: Fine-Tuning ChatTLA (5a_text-basic) ──
slide = prs.slides.add_slide(get_layout("5a_text-basic"))
set_placeholder(slide, 11, "Building a Better Model: ChatTLA", font_size=28, bold=True)
set_placeholder(slide, 12, "Fine-tuning + self-improvement with model-checker feedback", font_size=13)
add_bullet_text(slide, 10, [
    "Start with 205 expert-verified TLA+ specifications as training data",
    "Fine-tune a 20B-parameter model specifically for TLA+ generation",
    "Key insight: use TLC (the model checker) as the training signal -- a spec is only \"correct\" if TLC says so",
    "Autonomous loop: generate specs -> validate with TLC -> retrain on successes",
    "The model checker provides a perfect, automatic grading signal -- unlike most AI tasks",
], font_size=14)

# ── Slide 6: Closing (8_closing) ──
slide = prs.slides.add_slide(get_layout("8_closing"))
set_placeholder(slide, 11, "Thank You", font_size=36, bold=True)
set_placeholder(slide, 12, "ai4fm.cs.luc.edu", font_size=14)

prs.save(OUTPUT)
print(f"Saved {OUTPUT} ({len(prs.slides)} slides)")
