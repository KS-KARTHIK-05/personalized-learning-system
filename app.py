import sys
import types
import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
import asyncio
import streamlit as st

import json
from dotenv import load_dotenv

load_dotenv()
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

from pipeline import PersonalizedLearningPipeline
from gap_analyzer import SkillGapAnalyzer
from pypdf import PdfReader
from fpdf import FPDF

try:
    from evaluator import run_dynamic_ragas_evaluation
except Exception as e:
    run_dynamic_ragas_evaluation = None
    evaluator_import_error = e
else:
    evaluator_import_error = None

# --- STREAMLIT DASHBOARD CONFIGURATION ---
st.set_page_config(
    page_title="Personalized Learning Pathway", 
    page_icon="🎓", 
    layout="wide"
)

# --- PRODUCTION STYLING ---
st.markdown("""
    <style>
    .main { background-color: #0f1116; }
    .stTextArea textarea, .stTextInput input { background-color: #1b1e24 !important; color: #e0e6ed !important; }
    div.stFormSubmitButton > button {
        background-color: #2563eb !important;
        color: white !important;
        border: 1px solid #3b82f6 !important;
        border-radius: 6px !important;
        transition: background-color 0.3s ease;
    }
    div.stFormSubmitButton > button:hover { background-color: #1d4ed8 !important; border-color: #2563eb !important; }
    .badge-have { background-color: #10b98120; color: #10b981; padding: 6px 12px; border-radius: 20px; border: 1px solid #10b98140; font-weight: bold; margin: 4px; display: inline-block; }
    .badge-gap { background-color: #f59e0b20; color: #f59e0b; padding: 6px 12px; border-radius: 20px; border: 1px solid #f59e0b40; font-weight: bold; margin: 4px; display: inline-block; }
    .time-badge { background-color: #3b82f620; color: #3b82f6; font-size: 0.85rem; padding: 4px 10px; border-radius: 4px; border: 1px solid #3b82f640; font-weight: bold; margin-left: 15px; display: inline-block; }
    .timeline-container { padding: 15px 30px; position: relative; }
    .timeline-block { position: relative; padding-left: 35px; margin-bottom: 25px; border-left: 3px solid #3b82f6; }
    .timeline-dot { position: absolute; left: -10px; top: 2px; background-color: #3b82f6; width: 18px; height: 18px; border-radius: 50%; border: 3px solid #0f1116; }
    .timeline-title { font-size: 1.25rem; font-weight: bold; color: #3b82f6; margin-bottom: 5px; display: flex; align-items: center; }
    .timeline-card { background-color: #161920; padding: 20px; border-radius: 8px; border: 1px solid #2d3139; color: #e0e6ed; }
    .course-item { background-color: #1f242e; padding: 10px 15px; border-radius: 6px; margin: 8px 0px; border-left: 3px solid #10b981; font-weight: 500; }
    .empty-item { color: #94a3b8; font-style: italic; }
    [data-testid="stForm"] { border: none !important; padding: 0 !important; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_cached_pipeline():
    return PersonalizedLearningPipeline()

pipeline = None
try:
    pipeline = get_cached_pipeline()
except Exception as e:
    st.error(f"⚠️ Model system linkage failed: {e}")

def parse_pdf_resume(uploaded_file):
    try:
        reader = PdfReader(uploaded_file)
        text_content = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_content += text + "\n"
        return text_content.strip()
    except Exception as e:
        st.error(f"PDF extraction issue: {e}")
        return None

class CleanPDFDocument(FPDF):
    def header(self):
        self.set_font("Helvetica", style="B", size=14)
        self.set_text_color(30, 41, 59)
        self.cell(0, 10, "PERSONALIZED LEARNING ROADMAP REPORT", align="C")
        self.ln(10)
        self.set_draw_color(59, 130, 246)
        self.line(10, 22, 200, 22)
        self.ln(6)

    def write_clean_block(self, title, text_body):
        self.set_font("Helvetica", style="B", size=11)
        self.set_text_color(59, 130, 246)
        self.cell(0, 8, title.upper())
        self.ln(8)
        self.set_font("Helvetica", style="", size=10)
        self.set_text_color(50, 50, 50)
        
        if isinstance(text_body, list) and len(text_body) > 0:
            for block in text_body:
                if isinstance(block, dict):
                    line_text = block.get("title", block.get("name", str(block)))
                else:
                    line_text = str(block)
                cleaned_line = f"- {line_text}"
                encoded_line = cleaned_line.encode('latin-1', 'ignore').decode('latin-1')
                self.multi_cell(0, 6, encoded_line)
                self.ln(1)
        else:
            msg = "- Candidate already has proficient mastery in these foundations."
            self.multi_cell(0, 6, msg)
            self.ln(1)
        self.ln(4)

def format_timeline_card(phase_content):
    if isinstance(phase_content, list) and len(phase_content) > 0:
        html_out = ""
        for item in phase_content:
            # Handle both raw dicts and plain string outputs
            if isinstance(item, dict):
                title = item.get("title", item.get("name", str(item)))
                item_type = item.get("type", "course")
                icon = "📘" if item_type == "course" else "🎯"
                html_out += f"<div class='course-item'>{icon} {title}</div>"
            else:
                html_out += f"<div class='course-item'>📘 {str(item)}</div>"
        return html_out
    return "<div class='empty-item'>✨ Core competencies for this phase are already verified from your profile!</div>"

if "gap_results" not in st.session_state:
    st.session_state.gap_results = None
if "phases" not in st.session_state:
    st.session_state.phases = None
if "last_target_role" not in st.session_state:
    st.session_state.last_target_role = ""
if "retrieved_docs" not in st.session_state:
    st.session_state.retrieved_docs = []
if "eval_df" not in st.session_state:
    st.session_state.eval_df = None

st.title("🎓 Personalized Learning Pathway")
st.markdown("---")

with st.form(key="user_input_form"):
    left_panel, right_panel = st.columns([1, 1])

    with left_panel:
        st.markdown("### 📂 Step 1: Share Your Profile")
        uploaded_file = st.file_uploader("Upload your resume (PDF format):", type=["pdf"])
        st.markdown("**OR** split entries manually below:")
        manual_bio = st.text_area("Type your current technical skills or experience here:", height=110, placeholder="e.g., Python, basic SQL, HTML...")

    with right_panel:
        st.markdown("### 🎯 Step 2: Set Your Career Goal")
        target_role = st.text_input("What technical role or skill track are you aiming for?", placeholder="e.g., Full stack developer, ML engineer...")
        st.markdown("<br><br>", unsafe_allow_html=True)
        generate_btn = st.form_submit_button(label="⚡ Generate My Learning Path", use_container_width=True)

if generate_btn:
    if pipeline is None:
        st.error("❌ Pipeline is uninitialized.")
    else:
        profile_data = parse_pdf_resume(uploaded_file) if uploaded_file else manual_bio
        
        if not profile_data or not target_role:
            st.error("⚠️ Please provide profile details and enter your target goal first.")
        else:
            with st.spinner("🧠 Analyzing profile parameters and mapping relevant database courses..."):
                st.session_state.last_target_role = target_role
                st.session_state.gap_results = pipeline.analyzer.analyze_profile_and_gap(profile_data, target_role)
                st.session_state.eval_df = None
                
                if st.session_state.gap_results:
                    missing_gaps = st.session_state.gap_results.get('calculated_gaps', [])
                    
                    rag_context = ""
                    retrieved_chunks = []
                    
                    for gap in missing_gaps:
                        matched = pipeline.search_engine.search_courses(skill_gap_query=gap, top_k=2)
                        for c in matched:
                            chunk_str = f"Gap: {gap} -> Course Title: {c['title']} (Teaches: {c['skills']})"
                            rag_context += chunk_str + "\n"
                            retrieved_chunks.append(chunk_str)
                    
                    st.session_state.retrieved_docs = retrieved_chunks
                    
                    synthesis_prompt = f"""
                    You are an expert technical curriculum architect.
                    Organize the provided courses into a 4-phase sequential roadmap for the target role: "{target_role}".

                    Retrieved Course Recommendations:
                    {rag_context}

                    OUTPUT REQUIREMENTS:
                    Return ONLY a flat, valid JSON object with EXACTLY these four keys:
                    {{
                        "phase1": ["Course Title 1", "Course Title 2"],
                        "phase2": ["Course Title 3", "Course Title 4"],
                        "phase3": ["Course Title 5", "Course Title 6"],
                        "phase4": ["Course Title 7", "Course Title 8"]
                    }}

                    - Every phase MUST contain at least 2 course title strings from the retrieved recommendations.
                    - Do NOT nest the object inside another dictionary (e.g., do not wrap inside "roadmap" or "phases").
                    - Do NOT return dictionaries inside the lists; return only plain string titles.
                    - Do NOT include markdown blocks, backticks, or any conversational text.
                    """
                    
                    try:
                        clean_json = pipeline.analyzer.generate_text(synthesis_prompt).strip()
                        
                        # Clean potential markdown wrappers
                        if clean_json.startswith("```"):
                            clean_json = clean_json.split("\n", 1)[1].rsplit("\n", 1)[0].strip()
                        if clean_json.lower().startswith("json"):
                            clean_json = clean_json.split("\n", 1)[1].strip()
                        clean_json = clean_json.strip()

                        raw_parsed = json.loads(clean_json)

                        # Handle nested wrappers (e.g., {"roadmap": {...}} or {"curriculum": {...}})
                        if isinstance(raw_parsed, dict):
                            for possible_wrapper in ["roadmap", "curriculum", "phases", "data", "schedule"]:
                                if possible_wrapper in raw_parsed and isinstance(raw_parsed[possible_wrapper], dict):
                                    raw_parsed = raw_parsed[possible_wrapper]
                                    break

                        # Normalize keys defensively (e.g., "Phase 1", "phase_1" -> "phase1")
                        normalized_phases = {"phase1": [], "phase2": [], "phase3": [], "phase4": []}
                        for k, v in raw_parsed.items():
                            k_clean = str(k).lower().replace(" ", "").replace("_", "").replace("-", "")
                            if k_clean in normalized_phases:
                                # Ensure entries are strings, not dicts
                                if isinstance(v, list):
                                    items = []
                                    for item in v:
                                        if isinstance(item, dict):
                                            items.append(item.get("title", item.get("name", str(item))))
                                        else:
                                            items.append(str(item))
                                    normalized_phases[k_clean] = items
                                else:
                                    normalized_phases[k_clean] = [str(v)]

                        # Fallback if any phase is empty: distribute retrieved courses evenly
                        if not any(normalized_phases.values()) and retrieved_chunks:
                            chunk_titles = [c.split("-> Course Title: ")[-1].split(" (Teaches:")[0] for c in retrieved_chunks]
                            n = max(1, len(chunk_titles) // 4)
                            normalized_phases["phase1"] = chunk_titles[:n] or chunk_titles[:1]
                            normalized_phases["phase2"] = chunk_titles[n:2*n] or chunk_titles[1:2]
                            normalized_phases["phase3"] = chunk_titles[2*n:3*n] or chunk_titles[2:3]
                            normalized_phases["phase4"] = chunk_titles[3*n:] or chunk_titles[3:]

                        st.session_state.phases = normalized_phases

                    except Exception as err:
                        st.error(f"Compilation layout sync exception: {err}")

if st.session_state.gap_results and st.session_state.phases:
    gap_results = st.session_state.gap_results
    phases = st.session_state.phases
    current_target = st.session_state.last_target_role
    
    st.markdown("---")
    st.markdown("### 📊 Capabilities Assessment Overview")
    
    st.markdown("**Skills Discovered on Profile:**")
    have_html = "".join([f'<span class="badge-have">{s}</span>' for s in gap_results.get('extracted_skills', [])])
    st.markdown(have_html, unsafe_allow_html=True)
    
    st.markdown("**Identified Target Gaps to Bridge:**")
    gap_html = "".join([f'<span class="badge-gap">{s}</span>' for s in gap_results.get('calculated_gaps', [])])
    st.markdown(gap_html, unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown(f"### 🗺️ Visual Learning Roadmap: {current_target}")
    st.markdown("<div class='timeline-container'>", unsafe_allow_html=True)
    
    for phase_idx, (phase_key, time_tag, label_title) in enumerate([
        ("phase1", "Weeks 1 - 4", "Phase 1: Foundations"),
        ("phase2", "Weeks 5 - 8", "Phase 2: Core Architecture"),
        ("phase3", "Weeks 9 - 12", "Phase 3: Advanced Specialization"),
        ("phase4", "Weeks 13 - 16", "Phase 4: Production & Cloud Deployment")
    ], start=1):
        st.markdown(f"""
            <div class='timeline-block'>
                <div class='timeline-dot'></div>
                <div class='timeline-title'>📌 {label_title} <span class='time-badge'>⏱️ {time_tag}</span></div>
                <div class='timeline-card'>{format_timeline_card(phases.get(phase_key, []))}</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    try:
        pdf = CleanPDFDocument()
        pdf.add_page()
        pdf.write_clean_block("Phase 1 Foundations (Weeks 1-4)", phases.get('phase1', []))
        pdf.write_clean_block("Phase 2 Core Operations (Weeks 5-8)", phases.get('phase2', []))
        pdf.write_clean_block("Phase 3 Advanced Development (Weeks 9-12)", phases.get('phase3', []))
        pdf.write_clean_block("Phase 4 Deployment & Automation (Weeks 13-16)", phases.get('phase4', []))
        
        st.download_button(
            label="📥 Download Roadmap Report",
            data=bytes(pdf.output()),
            file_name=f"{current_target.replace(' ', '_')}_Roadmap.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    except Exception as pdf_error:
        st.error(f"PDF generation stalled: {pdf_error}")

    st.markdown("---")
    st.markdown("### 🧪 Real-Time System Evaluation (RAG Triad Metrics)")
    
    if st.button("⚡ Evaluation", use_container_width=True):
        if run_dynamic_ragas_evaluation is None:
            st.error(f"Evaluation feature disabled: {evaluator_import_error}")
        else:
            with st.spinner("Calculating Faithfulness, Answer Relevancy, and Context Precision..."):
                try:
                    user_query_str = f"Target Role: {current_target} | Skill Gaps: {', '.join(gap_results.get('calculated_gaps', []))}"
                    generated_roadmap_str = json.dumps(phases)
                    
                    _, st.session_state.eval_df = run_dynamic_ragas_evaluation(
                        user_query=user_query_str,
                        retrieved_docs=st.session_state.retrieved_docs,
                        generated_roadmap=generated_roadmap_str
                    )
                except Exception as eval_err:
                    st.error(f"Evaluation error: {eval_err}")
    
    if st.session_state.eval_df is not None:
        eval_df = st.session_state.eval_df
        c1, c2, c3 = st.columns(3)
        c1.metric("Faithfulness (Zero-Hallucination)", f"{eval_df['faithfulness'].iloc[0]:.2f}")
        c2.metric("Answer Relevancy", f"{eval_df['answer_relevancy'].iloc[0]:.2f}")
        c3.metric("Context Precision", f"{eval_df['context_precision'].iloc[0]:.2f}")