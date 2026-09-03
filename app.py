import os
import io
import json
import streamlit as st
from pydantic import BaseModel, Field
from typing import List, Optional
from pypdf import PdfReader
from docx import Document
from google import genai
from google.genai import types

# ==============================================================================
# CONFIGURATION
# ==============================================================================
MODEL_NAME = "gemini-3.6-flash"

st.set_page_config(
    page_title="Resume & Job Analyzer",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# PYDANTIC SCHEMAS FOR STRUCTURED GEMINI OUTPUT
# ==============================================================================
class MatchingSkill(BaseModel):
    skill: str
    evidence: str

class MissingSkill(BaseModel):
    skill: str
    priority: str  # Critical, Important, Nice to have
    status: str = "Not found in resume"

class WeakSkill(BaseModel):
    skill: str
    reason: str

class ExperienceAlignment(BaseModel):
    strong_matches: List[str]
    partial_matches: List[str]
    gaps: List[str]

class Keywords(BaseModel):
    present: List[str]
    missing: List[str]
    recommended: List[str]

class ATSAnalysis(BaseModel):
    score: int
    issues: List[str]
    recommendations: List[str]

class ResumeImprovement(BaseModel):
    priority: str  # High, Medium, Low
    change: str
    reason: str
    suggestion: str

class LearningPriority(BaseModel):
    skill: str
    priority: str  # High, Medium, Low
    reason: str

class AnalysisResult(BaseModel):
    match_score: int
    match_level: str  # Strongly aligned, Moderately aligned, Needs improvement, Poorly aligned
    summary: str
    top_strengths: List[str]
    top_gaps: List[str]
    matching_skills: List[MatchingSkill]
    missing_skills: List[MissingSkill]
    weak_skills: List[WeakSkill]
    experience_alignment: ExperienceAlignment
    keywords: Keywords
    ats_analysis: ATSAnalysis
    resume_improvements: List[ResumeImprovement]
    learning_priorities: List[LearningPriority]
    action_plan: List[str]


# ==============================================================================
# RESUME EXTRACTION HELPERS
# ==============================================================================
def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF file bytes in memory."""
    reader = PdfReader(io.BytesIO(file_bytes))
    extracted = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            extracted.append(text)
    return "\n".join(extracted)

def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text from DOCX file bytes in memory."""
    doc = Document(io.BytesIO(file_bytes))
    return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

def extract_text_from_txt(file_bytes: bytes) -> str:
    """Extract text from plain TXT file bytes."""
    return file_bytes.decode("utf-8", errors="ignore")

def process_uploaded_file(uploaded_file) -> Optional[str]:
    """Route uploaded file to appropriate text extractor."""
    try:
        file_bytes = uploaded_file.read()
        filename = uploaded_file.name.lower()
        
        if filename.endswith(".pdf"):
            text = extract_text_from_pdf(file_bytes)
        elif filename.endswith(".docx"):
            text = extract_text_from_docx(file_bytes)
        elif filename.endswith(".txt"):
            text = extract_text_from_txt(file_bytes)
        else:
            st.error("Unsupported file format. Please upload PDF, DOCX, or TXT.")
            return None
            
        if not text or len(text.strip()) < 50:
            st.error("Unable to extract sufficient text from this file. Please ensure it is not scanned/image-only.")
            return None
            
        return text.strip()
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None


# ==============================================================================
# VALIDATION HELPERS
# ==============================================================================
def validate_inputs(resume_text: str, job_description: str) -> bool:
    """Validate user inputs before making API requests."""
    if not resume_text or len(resume_text.strip()) < 50:
        st.error("Please upload a valid resume with extractable content.")
        return False
    if not job_description or len(job_description.strip()) < 50:
        st.error("Please provide a complete job description (at least 50 characters).")
        return False
    return True


# ==============================================================================
# GEMINI ANALYSIS ENGINE
# ==============================================================================
def get_gemini_client() -> Optional[genai.Client]:
    """Initialize Google GenAI Client securely using env var or st.secrets."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key and "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
        
    if not api_key:
        st.error("⚠️ GEMINI_API_KEY not found. Please configure it in environment variables or Streamlit secrets.")
        return None
        
    return genai.Client(api_key=api_key)

def analyze_resume_and_job(resume_text: str, job_description: str) -> Optional[AnalysisResult]:
    """Send structured prompt to Gemini Flash and parse JSON result."""
    client = get_gemini_client()
    if not client:
        return None

    prompt = f"""
You are an expert career consultant, ATS specialist, and senior hiring manager.
Perform a rigorous, objective comparison between the provided Candidate Resume and Job Description.

EVIDENCE & STRICT TRUTHFULNESS RULES:
1. Base your evaluation strictly on explicit evidence found in the Resume and Job Description.
2. NEVER invent candidate achievements, technologies, certifications, or metrics.
3. NEVER fabricate percentage improvements, revenue numbers, or team sizes if not present.
4. DO NOT assume missing information means the candidate lacks the skill. Use phrases like "Not found in resume" or "Not explicitly mentioned" rather than "The candidate does not know X".
5. Distinguish clearly between what is missing from the resume vs. what is recommended for the candidate to learn.

ANALYSIS GUIDELINES:
- Overall Score (0-100): Reflect evidence-backed alignment, not an absolute guarantee of hire.
- Skills: Group into Matching, Missing (Critical/Important/Nice to have), and Weakly Represented.
- Bullet Improvements: Provide concrete rewrite directions. If metrics are missing, use clear instructions like "[Add team size if applicable]".
- ATS Analysis: Provide realistic feedback on structure, keywords, and formatting standards.
- Learning Priorities: Highlight high-impact skill gaps that would strengthen alignment if acquired.

=== CANDIDATE RESUME ===
{resume_text}

=== JOB DESCRIPTION ===
{job_description}
"""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=AnalysisResult,
                temperature=0.2,
            ),
        )
        # Parse JSON response into Pydantic model
        result_data = json.loads(response.text)
        return AnalysisResult(**result_data)
    except Exception as e:
        st.error(f"An error occurred while communicating with Gemini API: {str(e)}")
        return None


# ==============================================================================
# UI HELPERS & RENDERING
# ==============================================================================
def render_score_badge(score: int, level: str):
    """Display color-coded score metric."""
    if score >= 80:
        color = "#2e7d32"  # Green
    elif score >= 60:
        color = "#f57c00"  # Orange/Amber
    elif score >= 40:
        color = "#d32f2f"  # Red-Orange
    else:
        color = "#c62828"  # Dark Red

    st.markdown(
        f"""
        <div style="background-color: {color}15; border-left: 6px solid {color}; padding: 16px; border-radius: 6px; margin-bottom: 20px;">
            <h2 style="margin:0; color: {color};">Resume–Job Match: {score} / 100</h2>
            <h4 style="margin:4px 0 0 0; color: #424242;">Alignment Status: <strong>{level}</strong></h4>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# MAIN STREAMLIT APPLICATION
# ==============================================================================
def main():
    st.title("💼 Resume & Job Analyzer")
    st.caption("Analyze your CV against job postings, fix ATS issues, and optimize your application with AI.")

    # Sidebar Privacy Notice
    with st.sidebar:
        st.header("🔒 Privacy & Security")
        st.info(
            "Your uploaded resume and job descriptions are processed entirely in-memory and sent directly to Google Gemini for processing. "
            "No files or personal data are stored on disk or saved to a database."
        )
        st.markdown("---")
        st.markdown("**Engine**: Google Gemini 2.5 Flash")

    # Layout Setup
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Candidate Resume")
        uploaded_file = st.file_uploader("Upload CV (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])
        
        resume_text = ""
        if uploaded_file is not None:
            resume_text = process_uploaded_file(uploaded_file) or ""
            if resume_text:
                st.success(f"Successfully loaded `{uploaded_file.name}`")
                with st.expander("📄 View Extracted Text Preview"):
                    st.text(resume_text[:600] + ("..." if len(resume_text) > 600 else ""))

    with col2:
        st.subheader("2. Target Job Description")
        job_description = st.text_area(
            "Paste the complete job description here",
            height=280,
            placeholder="Paste role responsibilities, requirements, and qualifications...",
        )

    st.markdown("---")

    # Trigger Action
    if st.button("🚀 Analyze Alignment", type="primary", use_container_width=True):
        if validate_inputs(resume_text, job_description):
            with st.spinner("Analyzing resume against job description using Gemini Flash..."):
                analysis: Optional[AnalysisResult] = analyze_resume_and_job(resume_text, job_description)

            if analysis:
                st.markdown("## 📊 Executive Summary")
                render_score_badge(analysis.match_score, analysis.match_level)
                st.write(analysis.summary)

                # Key Strengths & Gaps Snapshot
                k1, k2 = st.columns(2)
                with k1:
                    st.markdown("### ✅ Key Strengths")
                    for s in analysis.top_strengths:
                        st.markdown(f"- {s}")
                with k2:
                    st.markdown("### ⚠️ Primary Gaps")
                    for g in analysis.top_gaps:
                        st.markdown(f"- {g}")

                st.markdown("---")

                # Detailed Tabs
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "🛠️ Skills Analysis",
                    "🎯 Experience & Keywords",
                    "🤖 ATS Readiness",
                    "📝 Improvement Plan",
                    "🚀 Learning & Action Plan"
                ])

                # TAB 1: SKILLS ANALYSIS
                with tab1:
                    st.subheader("Matching Skills")
                    if analysis.matching_skills:
                        for item in analysis.matching_skills:
                            st.markdown(f"**• {item.skill}**: {item.evidence}")
                    else:
                        st.write("No direct explicit skill matches identified.")

                    st.markdown("---")
                    st.subheader("Missing Skills (Not Found in Resume)")
                    if analysis.missing_skills:
                        for item in analysis.missing_skills:
                            st.markdown(f"**• {item.skill}** `[{item.priority} Priority]` — *{item.status}*")
                    else:
                        st.write("No major skill gaps detected.")

                    st.markdown("---")
                    st.subheader("Weakly Represented Skills")
                    if analysis.weak_skills:
                        for item in analysis.weak_skills:
                            st.markdown(f"**• {item.skill}**: {item.reason}")
                    else:
                        st.write("No weakly represented skills noted.")

                # TAB 2: EXPERIENCE & KEYWORDS
                with tab2:
                    st.subheader("Experience Alignment")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Highly Relevant Experience**")
                        for m in analysis.experience_alignment.strong_matches:
                            st.markdown(f"- {m}")
                        st.markdown("**Partially Relevant Experience**")
                        for p in analysis.experience_alignment.partial_matches:
                            st.markdown(f"- {p}")
                    with c2:
                        st.markdown("**Unaddressed Responsibilities / Experience Gaps**")
                        for g in analysis.experience_alignment.gaps:
                            st.markdown(f"- {g}")

                    st.markdown("---")
                    st.subheader("Keyword Optimization")
                    st.markdown(f"**Present Keywords:** {', '.join(analysis.keywords.present) if analysis.keywords.present else 'None'}")
                    st.markdown(f"**Missing Keywords:** {', '.join(analysis.keywords.missing) if analysis.keywords.missing else 'None'}")
                    
                    if analysis.keywords.recommended:
                        st.info("💡 **Incorporation Advice:** " + ", ".join(analysis.keywords.recommended))

                # TAB 3: ATS READINESS
                with tab3:
                    st.subheader("ATS Compatibility Review")
                    st.metric("ATS Readiness Score", f"{analysis.ats_analysis.score} / 100")
                    
                    a1, a2 = st.columns(2)
                    with a1:
                        st.markdown("**Potential ATS Formatting/Content Issues**")
                        for issue in analysis.ats_analysis.issues:
                            st.markdown(f"- {issue}")
                    with a2:
                        st.markdown("**Recommended ATS Fixes**")
                        for rec in analysis.ats_analysis.recommendations:
                            st.markdown(f"- {rec}")

                # TAB 4: IMPROVEMENT PLAN
                with tab4:
                    st.subheader("Prioritized Resume Enhancements")
                    for imp in sorted(analysis.resume_improvements, key=lambda x: x.priority):
                        with st.expander(f"[{imp.priority} Priority] {imp.change}"):
                            st.markdown(f"**Why change this:** {imp.reason}")
                            st.markdown(f"**How to improve:** {imp.suggestion}")

                # TAB 5: LEARNING & ACTION PLAN
                with tab5:
                    st.subheader("What Should I Learn First?")
                    st.caption("Missing target role skills prioritized by role requirements:")
                    for lp in analysis.learning_priorities:
                        st.markdown(f"**1. {lp.skill}** (`{lp.priority} Priority`) — {lp.reason}")

                    st.markdown("---")
                    st.subheader("Top Actions Before Applying")
                    for i, step in enumerate(analysis.action_plan, 1):
                        st.markdown(f"**{i}.** {step}")


if __name__ == "__main__":
    main()
