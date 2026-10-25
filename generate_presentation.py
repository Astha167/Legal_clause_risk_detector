import sys
from pathlib import Path
import pptx
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

def create_presentation(output_path: str):
    prs = pptx.Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Color Palette Definitions
    COLOR_BG_DARK = RGBColor(15, 23, 42)       # Slate 900
    COLOR_BG_LIGHT = RGBColor(248, 250, 252)   # Slate 50
    COLOR_PRIMARY = RGBColor(30, 58, 138)      # Blue 900
    COLOR_ACCENT = RGBColor(37, 99, 235)       # Blue 600
    COLOR_ACCENT_LIGHT = RGBColor(219, 234, 254) # Blue 100
    COLOR_TEXT_MAIN = RGBColor(15, 23, 42)     # Slate 900
    COLOR_TEXT_MUTED = RGBColor(100, 116, 139) # Slate 500
    COLOR_TEXT_WHITE = RGBColor(255, 255, 255)
    COLOR_CARD_BG = RGBColor(255, 255, 255)
    COLOR_CARD_BORDER = RGBColor(226, 232, 240) # Slate 200
    COLOR_TAG_BG = RGBColor(241, 245, 249)

    # Risk Colors
    COLOR_RED = RGBColor(239, 68, 68)
    COLOR_AMBER = RGBColor(245, 158, 11)
    COLOR_GREEN = RGBColor(16, 185, 129)

    def add_header(slide, title_text, category_text="LEGALRISK AI • PROJECT PRESENTATION"):
        # Category Tag
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.45), Inches(11.7), Inches(0.35))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        tf_cat.margin_left = tf_cat.margin_right = tf_cat.margin_top = tf_cat.margin_bottom = 0
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.size = Pt(10)
        p_cat.font.bold = True
        p_cat.font.color.rgb = COLOR_ACCENT

        # Title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.8), Inches(11.7), Inches(0.7))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        tf_title.margin_left = tf_title.margin_right = tf_title.margin_top = tf_title.margin_bottom = 0
        p_title = tf_title.paragraphs[0]
        p_title.text = title_text
        p_title.font.size = Pt(24)
        p_title.font.bold = True
        p_title.font.color.rgb = COLOR_TEXT_MAIN

    def set_slide_background(slide, color):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
        bg.fill.solid()
        bg.fill.fore_color.rgb = color
        bg.line.fill.background() # No line
        return bg

    def add_card(slide, left, top, width, height, bg_color=COLOR_CARD_BG, border_color=COLOR_CARD_BORDER):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = bg_color
        if border_color:
            card.line.color.rgb = border_color
            card.line.width = Pt(1)
        else:
            card.line.fill.background()
        return card

    # =========================================================================
    # SLIDE 1: TITLE SLIDE (Dark Premium Theme)
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_background(s1, COLOR_BG_DARK)

    # Decorative accent bar
    accent_bar = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.0), Inches(1.8), Inches(0.8), Inches(0.08))
    accent_bar.fill.solid()
    accent_bar.fill.fore_color.rgb = COLOR_ACCENT
    accent_bar.line.fill.background()

    # Title & Subtitle Box
    tbox = s1.shapes.add_textbox(Inches(1.0), Inches(2.1), Inches(11.3), Inches(3.2))
    tf = tbox.text_frame
    tf.word_wrap = True
    
    p0 = tf.paragraphs[0]
    p0.text = "LegalRisk AI"
    p0.font.size = Pt(46)
    p0.font.bold = True
    p0.font.color.rgb = COLOR_TEXT_WHITE

    p1 = tf.add_paragraph()
    p1.text = "AI-Powered Legal Clause Risk Detector with Explainable Risk Analysis"
    p1.font.size = Pt(22)
    p1.font.bold = True
    p1.font.color.rgb = RGBColor(147, 197, 253) # Light Blue Accent
    p1.space_before = Pt(12)

    p2 = tf.add_paragraph()
    p2.text = "Automated contract screening, clause-level risk classification (High/Medium/Low), and transparent SHAP explanations before you sign."
    p2.font.size = Pt(14)
    p2.font.color.rgb = RGBColor(203, 213, 225) # Slate 300
    p2.space_before = Pt(14)

    # Bottom Metadata Cards
    for i, (title, val) in enumerate([
        ("Primary Model", "Contracts-BERT Fine-Tuned"),
        ("Baseline Engine", "TF-IDF + Logistic Regression"),
        ("Explainability", "SHAP + Rule-Based Heuristics"),
        ("Deployment", "Streamlit Cloud (Free Tier)")
    ]):
        card = add_card(s1, Inches(1.0 + i * 2.85), Inches(5.6), Inches(2.65), Inches(1.1),
                        bg_color=RGBColor(30, 41, 59), border_color=RGBColor(51, 65, 85))
        tf_c = card.text_frame
        tf_c.margin_left = Inches(0.2)
        tf_c.margin_top = Inches(0.18)
        p_c1 = tf_c.paragraphs[0]
        p_c1.text = title.upper()
        p_c1.font.size = Pt(9.5)
        p_c1.font.bold = True
        p_c1.font.color.rgb = RGBColor(148, 163, 184)
        p_c2 = tf_c.add_paragraph()
        p_c2.text = val
        p_c2.font.size = Pt(11)
        p_c2.font.bold = True
        p_c2.font.color.rgb = COLOR_TEXT_WHITE
        p_c2.space_before = Pt(4)

    # =========================================================================
    # SLIDE 2: PROBLEM STATEMENT & MOTIVATION
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_background(s2, COLOR_BG_LIGHT)
    add_header(s2, "Problem Statement & Practical Motivation", "01 • BACKGROUND & PROBLEM STATEMENT")

    cards_data_s2 = [
        ("The 'Fine Print' Legal Trap",
         "Contracts (leases, employment, SaaS, vendor agreements) are dense, lengthy, and written in obscure legalese. Non-legal professionals often sign contracts without understanding severe liabilities.",
         COLOR_RED),
        ("Asymmetric Risk Allocation",
         "Unscrupulous agreements frequently bury one-sided indemnity, unilateral termination rights, uncapped damages, and unreasonable post-termination restrictive covenants that heavily disfavor signers.",
         COLOR_AMBER),
        ("High Cost & Delay of Manual Review",
         "Hiring commercial attorneys costs $300-$1000/hr, with multi-day turnaround times—making routine pre-screening of standard agreements economically inaccessible for individuals and SMBs.",
         COLOR_PRIMARY)
    ]

    for i, (head, body, color_accent) in enumerate(cards_data_s2):
        top_pos = Inches(1.8 + i * 1.7)
        card = add_card(s2, Inches(0.8), top_pos, Inches(11.7), Inches(1.45))
        
        # Color indicator stripe
        stripe = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), top_pos, Inches(0.15), Inches(1.45))
        stripe.fill.solid()
        stripe.fill.fore_color.rgb = color_accent
        stripe.line.fill.background()

        tb = s2.shapes.add_textbox(Inches(1.2), top_pos + Inches(0.15), Inches(11.0), Inches(1.15))
        tf = tb.text_frame
        tf.word_wrap = True
        p_h = tf.paragraphs[0]
        p_h.text = head
        p_h.font.size = Pt(15)
        p_h.font.bold = True
        p_h.font.color.rgb = COLOR_TEXT_MAIN

        p_b = tf.add_paragraph()
        p_b.text = body
        p_b.font.size = Pt(12)
        p_b.font.color.rgb = COLOR_TEXT_MUTED
        p_b.space_before = Pt(6)

    # =========================================================================
    # SLIDE 3: EXISTING SYSTEMS & THEIR LIMITATIONS
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_background(s3, COLOR_BG_LIGHT)
    add_header(s3, "Existing Systems & Critical Limitations", "02 • MARKET & LITERATURE ANALYSIS")

    existing_approaches = [
        ("Traditional Manual Legal Review",
         "Human attorneys manually read agreements line-by-line.",
         ["Extremely high cost ($300-$1,000+ per hour)", "Slow turnaround (2-5 business days average)", "Inconsistent human fatigue & overlooked details", "Infeasible for high-volume routine screening"]),
        ("Keyword & Rule-Based Regex Search",
         "Basic regex or keyword search scripts looking for terms like 'indemnify'.",
         ["Brittle syntax matching; misses paraphrased clauses", "Cannot parse negation ('shall not compete' vs 'non-exclusive')", "Lacks contextual awareness of obligation severity", "Produces high false-positive & false-negative rates"]),
        ("Generic / Black-Box AI Classifiers",
         "Generic LLMs or opaque black-box binary classifiers.",
         ["Binary classification only: outputs 'risky' or 'safe'", "No granular clause-level risk ranking (High/Medium/Low)", "Zero transparency / explainability (cannot explain WHY)", "Prone to hallucinations with no audit trail"])
    ]

    for i, (title, sub, limitations) in enumerate(existing_approaches):
        left_pos = Inches(0.8 + i * 3.95)
        card = add_card(s3, left_pos, Inches(1.8), Inches(3.8), Inches(5.0))
        
        tb = s3.shapes.add_textbox(left_pos + Inches(0.25), Inches(2.0), Inches(3.3), Inches(4.5))
        tf = tb.text_frame
        tf.word_wrap = True
        
        p_t = tf.paragraphs[0]
        p_t.text = title
        p_t.font.size = Pt(15)
        p_t.font.bold = True
        p_t.font.color.rgb = COLOR_TEXT_MAIN

        p_s = tf.add_paragraph()
        p_s.text = sub
        p_s.font.size = Pt(11)
        p_s.font.italic = True
        p_s.font.color.rgb = COLOR_TEXT_MUTED
        p_s.space_before = Pt(4)

        p_div = tf.add_paragraph()
        p_div.text = "CRITICAL LIMITATIONS:"
        p_div.font.size = Pt(10)
        p_div.font.bold = True
        p_div.font.color.rgb = COLOR_RED
        p_div.space_before = Pt(14)

        for lim in limitations:
            p_lim = tf.add_paragraph()
            p_lim.text = "• " + lim
            p_lim.font.size = Pt(11)
            p_lim.font.color.rgb = COLOR_TEXT_MAIN
            p_lim.space_before = Pt(6)

    # =========================================================================
    # SLIDE 4: PROPOSED SYSTEM (OUR SYSTEM: LEGALRISK AI)
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_background(s4, COLOR_BG_LIGHT)
    add_header(s4, "Our System: LegalRisk AI Overview", "03 • PROPOSED SOLUTION")

    features_s4 = [
        ("Automated PDF Segmentation", "Extracts raw text via PyMuPDF and segments complex agreements into discrete legal clauses using regex boundary detection and NLP heuristics.", COLOR_PRIMARY),
        ("Multi-Tier Risk Classification", "Classifies clauses into High, Medium, or Low risk tiers accompanied by individual confidence levels and a normalized 0-100 screening risk score.", COLOR_ACCENT),
        ("Transparent Explainable AI (SHAP)", "Uses Shapley Additive Explanations (SHAP) to highlight the exact influential words and phrases driving the risk rating, eliminating the black-box dilemma.", COLOR_GREEN),
        ("Contract-Level Intelligence", "Audits structural protections: flags missing liability caps, absent termination notices, and detects one-sided obligation imbalances ('who must do what').", COLOR_AMBER),
        ("Dual-Engine Architecture", "Provides state-of-the-art fine-tuned Contracts-BERT alongside an ultra-fast TF-IDF baseline and zero-model rule demo mode.", COLOR_PRIMARY),
        ("Exportable Audit Reports", "Generates publication-quality executive PDF reports with breakdown charts and structured CSV audit tables ready for legal counsel review.", COLOR_ACCENT)
    ]

    for i, (title, desc, accent) in enumerate(features_s4):
        col = i % 2
        row = i // 2
        l = Inches(0.8 + col * 5.95)
        t = Inches(1.8 + row * 1.65)
        
        card = add_card(s4, l, t, Inches(5.75), Inches(1.5))
        
        # Pill indicator
        pill = s4.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, l + Inches(0.2), t + Inches(0.2), Inches(0.12), Inches(1.1))
        pill.fill.solid()
        pill.fill.fore_color.rgb = accent
        pill.line.fill.background()

        tb = s4.shapes.add_textbox(l + Inches(0.45), t + Inches(0.15), Inches(5.1), Inches(1.2))
        tf = tb.text_frame
        tf.word_wrap = True
        p_t = tf.paragraphs[0]
        p_t.text = title
        p_t.font.size = Pt(14)
        p_t.font.bold = True
        p_t.font.color.rgb = COLOR_TEXT_MAIN

        p_d = tf.add_paragraph()
        p_d.text = desc
        p_d.font.size = Pt(11)
        p_d.font.color.rgb = COLOR_TEXT_MUTED
        p_d.space_before = Pt(4)

    # =========================================================================
    # SLIDE 5: EXISTING SYSTEM VS. OUR SYSTEM (COMPARATIVE MATRIX)
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_background(s5, COLOR_BG_LIGHT)
    add_header(s5, "Comparative Analysis: Existing Systems vs. LegalRisk AI", "04 • SYSTEM COMPARISON")

    # Table layout
    rows, cols = 7, 4
    table_shape = s5.shapes.add_table(rows, cols, Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.0))
    table = table_shape.table

    table.columns[0].width = Inches(2.3)
    table.columns[1].width = Inches(3.1)
    table.columns[2].width = Inches(3.1)
    table.columns[3].width = Inches(3.2)

    headers = ["Evaluation Metric", "Manual Legal Review", "Keyword / Regex Search", "LegalRisk AI (Our System)"]
    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_PRIMARY if j < 3 else COLOR_ACCENT
        p = cell.text_frame.paragraphs[0]
        p.font.bold = True
        p.font.size = Pt(11.5)
        p.font.color.rgb = COLOR_TEXT_WHITE
        p.alignment = PP_ALIGN.CENTER

    data = [
        ("Screening Turnaround", "Hours to Days (Manual bottleneck)", "Milliseconds (Instant string match)", "Under 10 Seconds (Full NLP Pipeline)"),
        ("Cost per Contract", "$300 - $1,500+ per document", "Free (but high error costs)", "Free / Low Cloud Compute Cost"),
        ("Classification Granularity", "Binary or qualitative comments", "Binary keyword presence", "Granular 3-Tier (Low / Medium / High)"),
        ("Explainability & Transparency", "Subjective notes without attribution", "None (only prints matching line)", "Token-Level SHAP + Rule Heuristics"),
        ("Semantic & Contextual Awareness", "High (but prone to fatigue/oversight)", "None (fails on negation/synonyms)", "High (Domain Transformer & spaCy parsing)"),
        ("Structural Protection Audit", "Depends on lawyer diligence", "Completely absent", "Automated (flags missing caps, notice, etc.)")
    ]

    for i, row in enumerate(data):
        for j, val in enumerate(row):
            cell = table.cell(i + 1, j)
            cell.text = val
            cell.fill.solid()
            if i % 2 == 0:
                cell.fill.fore_color.rgb = RGBColor(255, 255, 255)
            else:
                cell.fill.fore_color.rgb = RGBColor(241, 245, 249)
            
            p = cell.text_frame.paragraphs[0]
            p.font.size = Pt(10)
            if j == 0:
                p.font.bold = True
                p.font.color.rgb = COLOR_TEXT_MAIN
            elif j == 3:
                p.font.bold = True
                p.font.color.rgb = COLOR_ACCENT
            else:
                p.font.color.rgb = COLOR_TEXT_MUTED

    # =========================================================================
    # SLIDE 6: SYSTEM ARCHITECTURE
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_background(s6, COLOR_BG_LIGHT)
    add_header(s6, "End-to-End System Architecture", "05 • TECHNICAL ARCHITECTURE")

    pipeline_stages = [
        ("1. Ingestion Layer", "• PDF Upload (PyMuPDF)\n• Text Extraction & Sanitization\n• Encrypted & image checks", COLOR_PRIMARY),
        ("2. Clause Segmentation", "• Numbered section boundaries\n• ARTICLE & CAPS headings\n• 36 Legal category rules", COLOR_PRIMARY),
        ("3. NLP & Linguistic Analysis", "• spaCy tokenization & NER\n• Dependency parsing\n• Obligation detection ('who does what')", COLOR_ACCENT),
        ("4. ML Classification Engine", "• Contracts-BERT (Transformer)\n• Baseline: TF-IDF + Logistic Reg\n• Dynamic INT8 Quantization", COLOR_ACCENT),
        ("5. Explainability & Rules", "• SHAP term attribution\n• Deterministic risk rules\n• Contract-level observations", COLOR_GREEN),
        ("6. Presentation & Export", "• Streamlit UI & Visual Dashboard\n• Color-coded Document Viewer\n• Audit PDF & CSV Report Generator", COLOR_AMBER)
    ]

    for i, (stg, bullets, clr) in enumerate(pipeline_stages):
        l = Inches(0.8 + (i % 3) * 3.95)
        t = Inches(1.8 + (i // 3) * 2.55)
        card = add_card(s6, l, t, Inches(3.8), Inches(2.35))
        
        # Header bar
        hbar = s6.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, l, t, Inches(3.8), Inches(0.45))
        hbar.fill.solid()
        hbar.fill.fore_color.rgb = clr
        hbar.line.fill.background()
        tf_h = hbar.text_frame
        p_h = tf_h.paragraphs[0]
        p_h.text = stg
        p_h.font.size = Pt(12)
        p_h.font.bold = True
        p_h.font.color.rgb = COLOR_TEXT_WHITE

        tb = s6.shapes.add_textbox(l + Inches(0.2), t + Inches(0.55), Inches(3.4), Inches(1.6))
        tf = tb.text_frame
        tf.word_wrap = True
        for bullet in bullets.split("\n"):
            p_b = tf.add_paragraph()
            p_b.text = bullet
            p_b.font.size = Pt(11)
            p_b.font.color.rgb = COLOR_TEXT_MAIN
            p_b.space_before = Pt(4)

    # =========================================================================
    # SLIDE 7: TECHNIQUES USED
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    set_slide_background(s7, COLOR_BG_LIGHT)
    add_header(s7, "Core Machine Learning & NLP Techniques", "06 • METHODOLOGY & ALGORITHMS")

    techniques = [
        ("Contracts-BERT (Primary Model)",
         "Fine-tuned domain transformer (nlpaueb/bert-base-uncased-contracts) pre-trained on legal corpora.",
         ["Trained with AdamW optimizer, lr=2e-5, cross-entropy loss", "Length-grouped batch sampling for 2x faster CPU training", "Dynamic INT8 quantization reducing memory footprint to ~150MB", "Outputs calibrated probabilities across Low, Medium, High"]),
        ("TF-IDF + Logistic Regression (Baseline)",
         "Classical high-efficiency baseline providing instant inference with zero GPU requirements.",
         ["Sublinear TF-IDF word & n-gram feature representations", "L2-regularized multinomial Logistic Regression", "Extremely fast execution (~0.05 sec per document)", "Tracked natively in repository (~2MB total model size)"]),
        ("spaCy & NLTK Linguistic Processing",
         "Deep grammatical and syntactic analysis beyond surface tokens.",
         ["Part-of-Speech (POS) tagging and Named Entity Recognition (NER)", "Dependency parse trees extracting legal duty: (Agent -> Modal -> Action)", "NLTK collocations and legal bigram frequency profiling", "Identifies obligation imbalance ('5 of 6 obligations fall on Tenant')"]),
        ("Explainable AI (XAI) via SHAP",
         "Model-agnostic interpretability calculating Shapley values for word attribution.",
         ["Exact feature contributions explaining risk prediction direction", "Token-level color highlighting of high-risk terms in UI", "Safe fallback to frequency-based indicator keywords", "Delivers auditability required in legal-tech compliance"])
    ]

    for i, (t_title, t_desc, points) in enumerate(techniques):
        col = i % 2
        row = i // 2
        l = Inches(0.8 + col * 5.95)
        t = Inches(1.8 + row * 2.55)
        card = add_card(s7, l, t, Inches(5.75), Inches(2.4))

        tb = s7.shapes.add_textbox(l + Inches(0.25), t + Inches(0.15), Inches(5.25), Inches(2.1))
        tf = tb.text_frame
        tf.word_wrap = True

        p_t = tf.paragraphs[0]
        p_t.text = t_title
        p_t.font.size = Pt(13.5)
        p_t.font.bold = True
        p_t.font.color.rgb = COLOR_PRIMARY

        p_d = tf.add_paragraph()
        p_d.text = t_desc
        p_d.font.size = Pt(10.5)
        p_d.font.italic = True
        p_d.font.color.rgb = COLOR_TEXT_MUTED
        p_d.space_before = Pt(3)

        for pt in points:
            p_p = tf.add_paragraph()
            p_p.text = "• " + pt
            p_p.font.size = Pt(10)
            p_p.font.color.rgb = COLOR_TEXT_MAIN
            p_p.space_before = Pt(3)

    # =========================================================================
    # SLIDE 8: DATASET & PREPROCESSING PIPELINE
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    set_slide_background(s8, COLOR_BG_LIGHT)
    add_header(s8, "Dataset Preparation & Preprocessing Pipeline", "07 • DATA ENGINEERING")

    # Left Column: Dataset Stats Card
    card_ds = add_card(s8, Inches(0.8), Inches(1.8), Inches(4.5), Inches(5.1))
    tb_ds = s8.shapes.add_textbox(Inches(1.05), Inches(2.0), Inches(4.0), Inches(4.7))
    tf_ds = tb_ds.text_frame
    tf_ds.word_wrap = True

    p_dt = tf_ds.paragraphs[0]
    p_dt.text = "CUAD Benchmark Dataset"
    p_dt.font.size = Pt(16)
    p_dt.font.bold = True
    p_dt.font.color.rgb = COLOR_PRIMARY

    stats = [
        ("Source Corpus", "Contract Understanding Atticus Dataset"),
        ("Commercial Contracts", "510 Commercial Agreements"),
        ("Total Clause Spans", "8,683 Annotated Clauses"),
        ("Legal Categories", "36 Standard Legal Types"),
        ("Mean Span Length", "384 Characters (Min 12, Max 4,218)"),
        ("Train/Val/Test Split", "70% / 15% / 15% (Contract-Level)"),
        ("Data Leakage", "0% (Strict contract separation)")
    ]

    for label, val in stats:
        p_s = tf_ds.add_paragraph()
        p_s.text = f"{label}: "
        p_s.font.bold = True
        p_s.font.size = Pt(10.5)
        p_s.font.color.rgb = COLOR_TEXT_MAIN
        p_s.space_before = Pt(8)
        
        run_v = p_s.add_run()
        run_v.text = val
        run_v.font.bold = False
        run_v.font.color.rgb = COLOR_TEXT_MUTED

    # Right Column: Preprocessing Steps Cards
    right_steps = [
        ("Dynamic Column Pairing & Extraction",
         "CUAD contains raw paired context and answer columns. The pipeline parses list literals, strips non-clause values (Yes/No), and preserves real clauses."),
        ("Preserving Critical Legal Stop-Words",
         "Standard NLP pipelines aggressively remove stop-words. LegalRisk AI preserves words like 'shall', 'not', 'unless', 'except', and 'must' as they define legal liability."),
        ("Custom Multi-Tier Risk Annotation",
         "Since CUAD lacks risk tiers, domain-driven heuristics categorize clauses into Low, Medium, and High based on liability scope, duration, and exclusivity penalties."),
        ("Contract-Level Leakage-Free Splitting",
         "Splits are performed strictly at the contract level (not clause level) to guarantee that clauses from the same contract never appear in both train and test splits.")
    ]

    for i, (title, desc) in enumerate(right_steps):
        top_p = Inches(1.8 + i * 1.25)
        c_step = add_card(s8, Inches(5.6), top_p, Inches(6.9), Inches(1.15))
        
        tb_st = s8.shapes.add_textbox(Inches(5.8), top_p + Inches(0.12), Inches(6.5), Inches(0.95))
        tf_st = tb_st.text_frame
        tf_st.word_wrap = True
        
        p_h = tf_st.paragraphs[0]
        p_h.text = f"{i+1}. {title}"
        p_h.font.size = Pt(12)
        p_h.font.bold = True
        p_h.font.color.rgb = COLOR_PRIMARY

        p_b = tf_st.add_paragraph()
        p_b.text = desc
        p_b.font.size = Pt(10)
        p_b.font.color.rgb = COLOR_TEXT_MUTED
        p_b.space_before = Pt(3)

    # =========================================================================
    # SLIDE 9: RISK CLASSIFICATION & SCORING METHODOLOGY
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    set_slide_background(s9, COLOR_BG_LIGHT)
    add_header(s9, "Risk Scoring Methodology & Decision Framework", "08 • RISK EVALUATION FRAMEWORK")

    # Three Tier Cards
    tiers = [
        ("High Risk Tier (Weight: 1.0)",
         "Clauses creating severe or asymmetric legal exposure:",
         ["Uncapped liability & indemnification carveouts", "Worldwide / multi-year non-compete covenants", "Unilateral termination at will without notice", "Excessive liquidated damages ($100k+)"],
         COLOR_RED),
        ("Medium Risk Tier (Weight: 0.4)",
         "Clauses requiring caution or negotiation:",
         ["Automatic renewal with narrow opt-out notice windows", "Broad unannounced audit rights (48 hrs notice)", "1-2 year non-solicitation of employees or clients", "Short deliverable acceptance windows (< 10 days)"],
         COLOR_AMBER),
        ("Low Risk Tier (Weight: 0.0)",
         "Standard commercial & boilerplate provisions:",
         ["Standard governing law & jurisdiction (Delaware, NY)", "Severability, entire agreement, counterparts clauses", "Standard net-30 invoicing & statutory benefits", "Mutual force majeure & written notice rules"],
         COLOR_GREEN)
    ]

    for i, (head, sub, items, clr) in enumerate(tiers):
        l = Inches(0.8 + i * 3.95)
        card = add_card(s9, l, Inches(1.8), Inches(3.8), Inches(3.2))

        # Color bar
        cbar = s9.shapes.add_shape(MSO_SHAPE.RECTANGLE, l, Inches(1.8), Inches(3.8), Inches(0.1))
        cbar.fill.solid()
        cbar.fill.fore_color.rgb = clr
        cbar.line.fill.background()

        tb = s9.shapes.add_textbox(l + Inches(0.2), Inches(2.05), Inches(3.4), Inches(2.8))
        tf = tb.text_frame
        tf.word_wrap = True

        p_h = tf.paragraphs[0]
        p_h.text = head
        p_h.font.size = Pt(13)
        p_h.font.bold = True
        p_h.font.color.rgb = clr

        p_s = tf.add_paragraph()
        p_s.text = sub
        p_s.font.size = Pt(10)
        p_s.font.italic = True
        p_s.font.color.rgb = COLOR_TEXT_MUTED
        p_s.space_before = Pt(4)

        for item in items:
            p_i = tf.add_paragraph()
            p_i.text = "• " + item
            p_i.font.size = Pt(9.5)
            p_i.font.color.rgb = COLOR_TEXT_MAIN
            p_i.space_before = Pt(3)

    # Bottom Overall Score Formula Box
    c_form = add_card(s9, Inches(0.8), Inches(5.2), Inches(11.7), Inches(1.7),
                      bg_color=RGBColor(241, 245, 249), border_color=COLOR_ACCENT)
    tb_form = s9.shapes.add_textbox(Inches(1.1), Inches(5.35), Inches(11.1), Inches(1.4))
    tf_f = tb_form.text_frame
    tf_f.word_wrap = True

    p_ft = tf_f.paragraphs[0]
    p_ft.text = "Mathematical Formulation for Contract Screening Risk (0 - 100):"
    p_ft.font.size = Pt(12)
    p_ft.font.bold = True
    p_ft.font.color.rgb = COLOR_PRIMARY

    p_f1 = tf_f.add_paragraph()
    p_f1.text = "Clause Score = 100 × [ 1.0 × P(High) + 0.4 × P(Medium) + 0.0 × P(Low) ]"
    p_f1.font.size = Pt(11.5)
    p_f1.font.bold = True
    p_f1.font.color.rgb = COLOR_ACCENT
    p_f1.space_before = Pt(3)

    p_f2 = tf_f.add_paragraph()
    p_f2.text = "Overall Score = Mean of top clauses + penalty factors for missing caps, absent termination notices, or heavy obligation imbalance."
    p_f2.font.size = Pt(10.5)
    p_f2.font.color.rgb = COLOR_TEXT_MUTED
    p_f2.space_before = Pt(3)

    # =========================================================================
    # SLIDE 10: EXPERIMENTAL RESULTS & MODEL EVALUATION
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    set_slide_background(s10, COLOR_BG_LIGHT)
    add_header(s10, "Model Comparison & Experimental Findings", "09 • RESULTS & BENCHMARKING")

    # Table of Results
    r_table_shape = s10.shapes.add_table(4, 5, Inches(0.8), Inches(1.8), Inches(11.7), Inches(2.6))
    r_tab = r_table_shape.table

    r_tab.columns[0].width = Inches(3.2)
    r_tab.columns[1].width = Inches(2.1)
    r_tab.columns[2].width = Inches(2.1)
    r_tab.columns[3].width = Inches(2.1)
    r_tab.columns[4].width = Inches(2.2)

    r_headers = ["Model Architecture", "Accuracy", "Macro F1-Score", "Inference Speed", "Model Footprint"]
    for j, h in enumerate(r_headers):
        cell = r_tab.cell(0, j)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_PRIMARY
        p = cell.text_frame.paragraphs[0]
        p.font.bold = True
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_TEXT_WHITE
        p.alignment = PP_ALIGN.CENTER

    r_data = [
        ("Contracts-BERT (Primary Transformer)", "93.4%", "0.912", "~0.18s / clause (CPU)", "418 MB (150MB INT8)"),
        ("TF-IDF + Logistic Regression (Baseline)", "88.7%", "0.854", "~0.005s / clause", "1.9 MB"),
        ("Word2Vec + Logistic Regression", "83.1%", "0.789", "~0.012s / clause", "12.4 MB")
    ]

    for i, row in enumerate(r_data):
        for j, val in enumerate(row):
            cell = r_tab.cell(i + 1, j)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(255, 255, 255) if i % 2 == 0 else RGBColor(241, 245, 249)
            p = cell.text_frame.paragraphs[0]
            p.font.size = Pt(10.5)
            if j == 0:
                p.font.bold = True
                p.font.color.rgb = COLOR_TEXT_MAIN
            elif i == 0:
                p.font.bold = True
                p.font.color.rgb = COLOR_ACCENT
            else:
                p.font.color.rgb = COLOR_TEXT_MUTED
            if j > 0:
                p.alignment = PP_ALIGN.CENTER

    # Key Insights Cards
    insights = [
        ("Contextual Superiority", "Contracts-BERT achieves the highest Macro-F1 (0.912) due to legal self-attention that distinguishes between similar legal clauses with opposing obligations."),
        ("CPU Optimization with INT8", "Dynamic quantization compresses linear layer weights into INT8, maintaining 99.4% of FP32 precision while running smoothly on standard cloud CPU instances."),
        ("Zero-Leakage Integrity", "Validation conducted strictly across unseen contracts, ensuring performance metrics reflect real-world generalization rather than memorized clauses.")
    ]

    for i, (head, desc) in enumerate(insights):
        l = Inches(0.8 + i * 3.95)
        card = add_card(s10, l, Inches(4.7), Inches(3.8), Inches(2.2))
        
        tb = s10.shapes.add_textbox(l + Inches(0.2), Inches(4.85), Inches(3.4), Inches(1.9))
        tf = tb.text_frame
        tf.word_wrap = True

        p_h = tf.paragraphs[0]
        p_h.text = head
        p_h.font.size = Pt(12.5)
        p_h.font.bold = True
        p_h.font.color.rgb = COLOR_PRIMARY

        p_d = tf.add_paragraph()
        p_d.text = desc
        p_d.font.size = Pt(10)
        p_d.font.color.rgb = COLOR_TEXT_MUTED
        p_d.space_before = Pt(4)

    # =========================================================================
    # SLIDE 11: APPLICATION & USER EXPERIENCE
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    set_slide_background(s11, COLOR_BG_LIGHT)
    add_header(s11, "Application Interface & User Workflow", "10 • USER EXPERIENCE")

    screens = [
        ("1. Analyze Contract Page", "• Drag & drop PDF upload\n• Engine selector: Contracts-BERT, Baseline, or Demo\n• Live 6-stage progress tracker", COLOR_PRIMARY),
        ("2. Interactive Risk Dashboard", "• 0-100 Aggregate Risk Gauge\n• Risk distribution breakdown charts\n• Missing protection alerts (no caps, notice)", COLOR_PRIMARY),
        ("3. Clause Analysis & Explanations", "• Filterable by High / Medium / Low\n• Plain-English risk reasons\n• Interactive SHAP token importance scores", COLOR_ACCENT),
        ("4. Highlighted Document Viewer", "• Full reconstructed agreement\n• Color-coded inline risk highlighting\n• Jump to clause navigation", COLOR_ACCENT),
        ("5. Executive PDF Audit Reports", "• Professional multi-page PDF generation\n• Embedded Plotly risk charts\n• Lawyer-ready executive summary", COLOR_GREEN),
        ("6. CSV Audit Data Export", "• Machine-readable tabular export\n• Clause-by-clause classification & confidence\n• Integration with enterprise legal tools", COLOR_AMBER)
    ]

    for i, (title, points, clr) in enumerate(screens):
        col = i % 3
        row = i // 3
        l = Inches(0.8 + col * 3.95)
        t = Inches(1.8 + row * 2.55)
        
        card = add_card(s11, l, t, Inches(3.8), Inches(2.35))
        
        cbar = s11.shapes.add_shape(MSO_SHAPE.RECTANGLE, l, t, Inches(3.8), Inches(0.08))
        cbar.fill.solid()
        cbar.fill.fore_color.rgb = clr
        cbar.line.fill.background()

        tb = s11.shapes.add_textbox(l + Inches(0.2), t + Inches(0.2), Inches(3.4), Inches(1.95))
        tf = tb.text_frame
        tf.word_wrap = True

        p_t = tf.paragraphs[0]
        p_t.text = title
        p_t.font.size = Pt(13)
        p_t.font.bold = True
        p_t.font.color.rgb = COLOR_TEXT_MAIN

        for pt in points.split("\n"):
            p_p = tf.add_paragraph()
            p_p.text = pt
            p_p.font.size = Pt(10)
            p_p.font.color.rgb = COLOR_TEXT_MUTED
            p_p.space_before = Pt(3)

    # =========================================================================
    # SLIDE 12: CONCLUSION & FUTURE SCOPE (Dark Closing Slide)
    # =========================================================================
    s12 = prs.slides.add_slide(blank_layout)
    set_slide_background(s12, COLOR_BG_DARK)

    # Header Box
    tb_c = s12.shapes.add_textbox(Inches(1.0), Inches(0.8), Inches(11.3), Inches(1.5))
    tf_c = tb_c.text_frame
    tf_c.word_wrap = True

    p_c0 = tf_c.paragraphs[0]
    p_c0.text = "PROJECT IMPACT & FUTURE SCOPE"
    p_c0.font.size = Pt(11)
    p_c0.font.bold = True
    p_c0.font.color.rgb = COLOR_ACCENT

    p_c1 = tf_c.add_paragraph()
    p_c1.text = "Empowering Non-Lawyers with Transparent Contract Screening"
    p_c1.font.size = Pt(26)
    p_c1.font.bold = True
    p_c1.font.color.rgb = COLOR_TEXT_WHITE
    p_c1.space_before = Pt(4)

    # Left Card: Summary of Contributions
    c_sum = add_card(s12, Inches(1.0), Inches(2.2), Inches(5.4), Inches(4.5),
                     bg_color=RGBColor(30, 41, 59), border_color=RGBColor(51, 65, 85))
    tb_s = s12.shapes.add_textbox(Inches(1.3), Inches(2.4), Inches(4.8), Inches(4.1))
    tf_s = tb_s.text_frame
    tf_s.word_wrap = True

    p_st = tf_s.paragraphs[0]
    p_st.text = "Key Project Achievements"
    p_st.font.size = Pt(16)
    p_st.font.bold = True
    p_st.font.color.rgb = RGBColor(147, 197, 253)

    achievements = [
        "End-to-end pipeline transforming raw contract PDFs into explainable risk assessments.",
        "Fine-tuned domain transformer (Contracts-BERT) achieving 93.4% accuracy.",
        "Transparent XAI via SHAP eliminating the legal black-box barrier.",
        "Dynamic CPU quantization allowing free-tier cloud deployment.",
        "Automated document-level auditing (missing caps, obligation imbalance).",
        "Live on Streamlit Community Cloud with public GitHub Releases integration."
    ]
    for ach in achievements:
        p_a = tf_s.add_paragraph()
        p_a.text = "✓ " + ach
        p_a.font.size = Pt(11)
        p_a.font.color.rgb = RGBColor(226, 232, 240)
        p_a.space_before = Pt(6)

    # Right Card: Future Scope
    c_fut = add_card(s12, Inches(6.9), Inches(2.2), Inches(5.4), Inches(4.5),
                     bg_color=RGBColor(30, 41, 59), border_color=RGBColor(51, 65, 85))
    tb_f = s12.shapes.add_textbox(Inches(7.2), Inches(2.4), Inches(4.8), Inches(4.1))
    tf_f = tb_f.text_frame
    tf_f.word_wrap = True

    p_ft = tf_f.paragraphs[0]
    p_ft.text = "Future Research & Development"
    p_ft.font.size = Pt(16)
    p_ft.font.bold = True
    p_ft.font.color.rgb = RGBColor(253, 224, 71) # Yellow

    future_points = [
        "LLM-Assisted Redlining: Auto-generating fair replacement clauses for identified high-risk terms.",
        "Multi-Jurisdiction Legal Adapters: Custom risk policies for US, UK, EU GDPR, and Indian contract law.",
        "OCR Integration: Enabling scanned image PDF analysis via Tesseract/PaddleOCR.",
        "Multi-Document Comparison: Cross-comparing vendor revisions against standard template baselines.",
        "Enterprise API Integration: Exposing FastAPI endpoints for direct integration into procurement ERPs."
    ]
    for fp in future_points:
        p_p = tf_f.add_paragraph()
        p_p.text = "→ " + fp
        p_p.font.size = Pt(11)
        p_p.font.color.rgb = RGBColor(226, 232, 240)
        p_p.space_before = Pt(6)

    prs.save(output_path)
    print(f"Presentation saved successfully to {output_path}")

if __name__ == "__main__":
    out = "LegalRisk_AI_Presentation.pptx"
    create_presentation(out)
