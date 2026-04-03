"""Build the 10-minute version of the talk (6 slides, ~100s each, more detail)."""
from pptx import Presentation
from pptx.util import Pt, Emu
from pptx.dml.color import RGBColor

MAROON = RGBColor(90, 7, 34)
GOLD = RGBColor(234, 170, 0)
GRAY_DARK = RGBColor(84, 83, 91)
WHITE = RGBColor(255, 255, 255)

TEMPLATE = "eric-talk/loy-powerpoint-16x9-websafe.pptx"
OUTPUT = "eric-talk/talk_10min.pptx"

prs = Presentation(TEMPLATE)

# Delete all example slides
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


def add_bullet_text(slide, idx, items, font_size=14, color=None, bold_first_word=False):
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
        para.space_after = Pt(4)
        for run in para.runs:
            run.font.size = Pt(font_size)
            if color:
                run.font.color.rgb = color


def add_rich_bullets(slide, idx, items, font_size=14, color=None):
    """Items are (bold_prefix, rest) tuples."""
    from pptx.util import Pt as _Pt
    ph = slide.placeholders[idx]
    tf = ph.text_frame
    tf.clear()
    for i, (prefix, rest) in enumerate(items):
        if i == 0:
            para = tf.paragraphs[0]
            para.clear()
        else:
            para = tf.add_paragraph()
        para.space_after = _Pt(4)
        if prefix:
            run_b = para.add_run()
            run_b.text = prefix
            run_b.font.size = _Pt(font_size)
            run_b.font.bold = True
            if color:
                run_b.font.color.rgb = color
        run_r = para.add_run()
        run_r.text = rest
        run_r.font.size = _Pt(font_size)
        if color:
            run_r.font.color.rgb = color


# ═══════════════════════════════════════════════════════════════════
# SLIDE 1 -- Title
# ═══════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(get_layout("1a_title-text"))
set_placeholder(slide, 15, "LOYOLA UNIVERSITY CHICAGO", font_size=11)
set_placeholder(slide, 11, "Can LLMs Write Correct TLA+ Specifications?", font_size=28, bold=True)
set_placeholder(slide, 12, "Evaluating Natural-Language-to-TLA+ Generation", font_size=14)
set_placeholder(slide, 14, "AI4FM Research Group  |  Department of Computer Science  |  ai4fm.cs.luc.edu", font_size=9)

# ═══════════════════════════════════════════════════════════════════
# SLIDE 2 -- What is TLA+? (two-column)
# ═══════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(get_layout("5b_text-2col"))
set_placeholder(slide, 11, "What Is TLA+?", font_size=28, bold=True)

set_placeholder(slide, 18, "The Language", font_size=14, bold=True)
add_rich_bullets(slide, 10, [
    ("Formal specification language: ", "describes system behavior precisely enough for a computer to verify"),
    ("State: ", "a snapshot of the system at one moment (e.g., \"the door is locked\")"),
    ("Init / Next: ", "the starting state and the rules for moving to the next state"),
    ("Invariant: ", "a safety rule that must always hold (e.g., \"two trains never occupy the same track\")"),
    ("SANY: ", "syntax checker -- \"Is this valid TLA+?\""),
    ("TLC: ", "model checker -- \"Does this spec actually satisfy its safety rules?\""),
], font_size=12)

set_placeholder(slide, 19, "Why It Matters", font_size=14, bold=True)
add_rich_bullets(slide, 15, [
    ("Amazon: ", "used TLA+ to find critical bugs in DynamoDB and S3 that years of testing missed"),
    ("Microsoft Azure: ", "verifies distributed protocols before deployment"),
    ("The barrier: ", "writing TLA+ requires rare, specialized expertise -- most engineers cannot use it"),
    ("The opportunity: ", "if AI could translate English to TLA+, formal verification becomes accessible to every engineer"),
    ("", "That is the question driving this research."),
], font_size=12)

# ═══════════════════════════════════════════════════════════════════
# SLIDE 3 -- Three kinds of AI models (3-col)
# ═══════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(get_layout("5c_text-3col"))
set_placeholder(slide, 11, "Three Kinds of AI Models We Tested", font_size=28, bold=True)

set_placeholder(slide, 22, "Basic Models", font_size=13, bold=True)
add_rich_bullets(slide, 10, [
    ("Examples: ", "LLaMA, Mistral, Gemma"),
    ("", "Trained on large amounts of internet text and code"),
    ("", "Good at pattern matching and completing familiar-looking code"),
    ("Limitation: ", "TLA+ looks nothing like Python or Java -- these models frequently produce broken output"),
    ("", ""),
    ("Analogy: ", "A native English speaker trying to write Mandarin by guessing characters"),
], font_size=11)

set_placeholder(slide, 23, "Reasoning Models", font_size=13, bold=True)
add_rich_bullets(slide, 20, [
    ("Examples: ", "DeepSeek R1, QwQ"),
    ("", "Designed to \"think step-by-step\" before answering"),
    ("", "Better at logical structure and multi-step problems"),
    ("Result: ", "Best performers overall -- how a model reasons matters more than its size"),
    ("", ""),
    ("Analogy: ", "Someone who reads the grammar book before writing a sentence"),
], font_size=11)

set_placeholder(slide, 24, "Code-Specialized", font_size=13, bold=True)
add_rich_bullets(slide, 18, [
    ("Examples: ", "Qwen-Coder, CodeLLaMA"),
    ("", "Specifically trained on programming languages (Python, JS, etc.)"),
    ("", "Should be good at code generation, right?"),
    ("Surprise: ", "Actually performed WORST on TLA+ -- negative transfer from mainstream code"),
    ("", ""),
    ("Analogy: ", "A French chef trying to make sushi -- expertise in one domain can hurt in another"),
], font_size=11)

# ═══════════════════════════════════════════════════════════════════
# SLIDE 4 -- Key Results (emphasis-maroon)
# ═══════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(get_layout("7a_emphasis-maroon"))
set_placeholder(slide, 11, "What We Found", font_size=28, bold=True)
set_placeholder(slide, 22, "30 models  |  8 families  |  205 specifications  |  2,730 evaluation runs", font_size=12)
add_rich_bullets(slide, 10, [
    ("Best syntactic correctness (SANY): ", "26.6% -- only 1 in 4 outputs is even valid TLA+"),
    ("Best semantic correctness (TLC): ", "8.6% -- fewer than 1 in 10 actually pass the model checker"),
    ("", ""),
    ("Surprising finding #1: ", "Model size does not predict quality. DeepSeek R1 at 8 billion parameters outperformed its 70 billion parameter variant across every prompting strategy."),
    ("Surprising finding #2: ", "Code-specialized training hurts. Models trained on Python and JavaScript consistently performed worse than general-purpose models."),
    ("Surprising finding #3: ", "Semantic success requires progressive prompting. Building specs piece-by-piece (Init, then Next, then Invariant) was the ONLY strategy that produced correct output. Direct and few-shot prompting never worked."),
], font_size=13, color=WHITE)

# ═══════════════════════════════════════════════════════════════════
# SLIDE 5 -- Fine-Tuning ChatTLA (two-column)
# ═══════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(get_layout("5b_text-2col"))
set_placeholder(slide, 11, "Building a Better Model: ChatTLA", font_size=28, bold=True)

set_placeholder(slide, 18, "Our Approach", font_size=14, bold=True)
add_rich_bullets(slide, 10, [
    ("1. Curate training data: ", "start with 205 expert-verified TLA+ specs, then scrape and validate more from GitHub. Every spec must pass both SANY and TLC."),
    ("2. Fine-tune with LoRA: ", "take an existing 20-billion-parameter model and train a small set of extra parameters on top. This is fast and efficient compared to training from scratch."),
    ("3. Model checker as training signal: ", "traditional AI training asks \"does the output look right?\" -- we ask \"does TLC say it has zero violations?\" This is a perfect, automatic grading signal."),
    ("4. Self-improvement loop: ", "the model generates new specs, TLC validates them, correct specs become new training data. The model retrains on its own successes, continuously improving."),
], font_size=11)

set_placeholder(slide, 19, "Why This Should Work", font_size=14, bold=True)
add_rich_bullets(slide, 15, [
    ("Perfect grading: ", "unlike most AI tasks where \"correct\" is subjective, TLC provides a binary, reliable answer -- pass or fail"),
    ("Growing dataset: ", "each training cycle produces new correct specs that feed the next cycle. The training set grows automatically."),
    ("Targeted expertise: ", "fine-tuning on TLA+ specifically addresses the negative transfer problem we observed in code-specialized models"),
    ("Accessible hardware: ", "runs continuously on two GPUs (96 GB total VRAM) -- no supercomputer required"),
    ("", ""),
    ("Goal: ", "make formal verification accessible to engineers who understand their systems but lack TLA+ expertise"),
], font_size=11)

# ═══════════════════════════════════════════════════════════════════
# SLIDE 6 -- Closing
# ═══════════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(get_layout("8_closing"))
set_placeholder(slide, 11, "Thank You", font_size=36, bold=True)
set_placeholder(slide, 12, "ai4fm.cs.luc.edu", font_size=14)

prs.save(OUTPUT)
print(f"Saved {OUTPUT} ({len(prs.slides)} slides)")
