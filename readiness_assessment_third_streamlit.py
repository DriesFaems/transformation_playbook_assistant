"""Streamlit Generative AI Readiness Assessment (Groq-only, no CrewAI)

This app replicates the multi-dimension readiness analysis but uses direct
Groq chat completion calls instead of CrewAI agents. It gathers structured
inputs, sends focused prompts per dimension, then synthesizes a consolidated
summary and concise recommendations.

Dimensions:
- Strategic Readiness
- Use Case Readiness
- Architecture Readiness
- Human (Talent & Culture) Readiness

Prerequisites:
- `groq`, `streamlit`, `langchain-groq` (optional) installed (already in requirements.txt)
- Set or enter your Groq API key.

Run:
    streamlit run readiness_assessment_third_streamlit.py
"""
from __future__ import annotations
import os
import streamlit as st
from groq import Groq
from typing import Dict, List

# ----------------------- UI SETUP -----------------------
st.set_page_config(page_title="Generative AI Readiness (Groq)", layout="wide")
st.title("Generative AI Readiness Assessment (Groq)")
st.write("""Provide information about your company and answer the questions.
The app will call Groq for structured assessments per dimension and a final summary.""")

# API Key input
api_key = st.text_input(
    "Groq API Key (https://console.groq.com/playground)", type="password"
)
model_choice = st.selectbox(
    "Model", ["openai/gpt-oss-20b", "llama-3.1-8b-instant", "llama3-70b-8192"], index=0
)
temperature = st.slider("Temperature", min_value=0.0, max_value=1.5, value=0.8, step=0.1)

st.markdown("### Company Context")
company_description = st.text_area(
    "Company Description (products, market, processes, customers)", height=150
)
existing_initiatives = st.text_area(
    "Existing Generative AI Initiatives (projects, pilots, vendors)", height=120
)
current_readiness_notes = st.text_area(
    "Current Internal Readiness Notes (optional)", height=100
)

st.markdown("### Multiple Choice Readiness Questions")
LIKERT = ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"]

question_sets = {
    "Strategic": [
        "Leadership understands the potential impact of generative AI.",
        "Clear strategy for implementing generative AI across departments.",
        "Generative AI initiatives align with overall business objectives.",
        "Clear vision for transformation via generative AI.",
    ],
    "Use Case": [
        "Specific generative AI use cases identified.",
        "Pilots / proofs of concept executed.",
        "ROI metrics defined for generative AI projects.",
        "Process exists to assess & prioritize new use cases regularly.",
    ],
    "Architecture": [
        "IT infrastructure supports generative AI workloads.",
        "Data storage/management suits large AI datasets.",
        "Data security & privacy measures are strong for AI data.",
        "Infrastructure is flexible/integrable for AI tools.",
    ],
    "Human": [
        "Workforce aware of generative AI and applications.",
        "In-house skills exist to build/implement solutions.",
        "Positive attitude toward adopting generative AI.",
        "Culture encourages experimentation & innovation with AI.",
    ],
}

responses: Dict[str, List[str]] = {}
cols = st.columns(2)
for i, (dimension, qs) in enumerate(question_sets.items()):
    with cols[i % 2]:
        st.markdown(f"#### {dimension} Readiness")
        answers = []
        for q in qs:
            answers.append(st.radio(q, LIKERT, key=f"{dimension}-{q}"))
        responses[dimension] = answers

# ----------------------- Helper Functions -----------------------

def likert_to_score(answer: str) -> int:
    mapping = {v: i + 1 for i, v in enumerate(LIKERT)}  # 1..5
    return mapping.get(answer, 3)


def build_client() -> Groq:
    if not api_key:
        raise ValueError("Please enter your Groq API key.")
    return Groq(api_key=api_key)


def build_dimension_prompt(dimension: str, qs: List[str], answers: List[str]) -> str:
    scored = [likert_to_score(a) for a in answers]
    lines = [
        f"Q{i+1}: {q}\n - Answer: {a}\n - Score(1-5): {scored[i]}" for i, (q, a) in enumerate(zip(qs, answers))
    ]
    return (
        f"You are an expert assessing the {dimension.lower()} readiness for generative AI.\n"
        "Context: Provide a concise, insight-focused assessment. Avoid generic statements.\n"
        f"Company Description: {company_description or 'N/A'}\n"
        f"Existing Initiatives: {existing_initiatives or 'N/A'}\n"
        f"Internal Notes: {current_readiness_notes or 'N/A'}\n"
        f"Dimension: {dimension}\n"
        "Questions & Answers with scores:\n" + "\n".join(lines) + "\n"
        "Instructions:\n"
        "1. Summarize current readiness level (1-5) with justification.\n"
        "2. List 2-3 specific strengths (if any).\n"
        "3. List 2-3 specific gaps or risks.\n"
        "4. DO NOT give recommendations here.\n"
        "Return JSON with keys: readiness_score, summary, strengths, gaps."
    )


def build_recommendation_prompt(dimension: str, assessment_json: str) -> str:
    return (
        f"You are an advisor generating improvement recommendations for {dimension.lower()} readiness in generative AI.\n"
        f"Assessment JSON: {assessment_json}\n"
        "Provide EXACTLY 2 high-impact, actionable recommendations.\n"
        "Each recommendation must include: action, rationale, success_metric.\n"
        "Return JSON list under key 'recommendations'."
    )


def call_groq(prompt: str, max_tokens: int = 1200) -> str:
    client = build_client()
    completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model_choice,
        temperature=temperature,
        max_completion_tokens=max_tokens,
        top_p=1,
        stream=False,
    )
    return completion.choices[0].message.content


def summarize_all(assessments: Dict[str, str], recommendations: Dict[str, str]) -> str:
    return (
        "You are a transformation strategist. Synthesize cross-dimensional readiness and recommendations.\n"
        f"Assessments RAW: {assessments}\nRecommendations RAW: {recommendations}\n"
        "Instructions:\n"
        "1. Provide an integrated narrative (<= 180 words).\n"
        "2. Highlight interplay across dimensions; avoid repetition.\n"
        "3. No new recommendations—only synthesis.\n"
        "4. End with one overall readiness risk statement."
    )


def build_readable_report_prompt(
    assessments: Dict[str, str],
    recommendations: Dict[str, str],
    synthesis: str,
) -> str:
    """Build a prompt that asks Groq to produce a polished, human-readable report."""
    return (
        "You are a professional business report writer. "
        "Convert the following raw assessment data into a polished, well-structured "
        "plain-text report that is easy and pleasant to read.\n\n"
        "RAW DIMENSION ASSESSMENTS (JSON):\n"
        + "\n".join(f"--- {dim} ---\n{data}" for dim, data in assessments.items())
        + "\n\nRAW DIMENSION RECOMMENDATIONS (JSON):\n"
        + "\n".join(f"--- {dim} ---\n{data}" for dim, data in recommendations.items())
        + f"\n\nINTEGRATED SYNTHESIS:\n{synthesis}\n\n"
        "INSTRUCTIONS:\n"
        "1. Produce a professional, reader-friendly report in plain text (no JSON, no code blocks, no markdown).\n"
        "2. Use this structure:\n"
        "   a. Title: 'GENERATIVE AI READINESS ASSESSMENT REPORT'\n"
        "   b. For EACH of the four dimensions (Strategic, Use Case, Architecture, Human):\n"
        "      - Readiness Score (out of 5)\n"
        "      - Brief narrative assessment (2-4 sentences)\n"
        "      - Key Strengths (bullet list)\n"
        "      - Key Gaps / Risks (bullet list)\n"
        "      - Recommendations with rationale and success metrics (numbered list)\n"
        "   c. Integrated Synthesis section (the cross-dimensional narrative)\n"
        "   d. Overall Readiness Risk Statement (1-2 sentences)\n"
        "3. Use clear section dividers (e.g. lines of dashes or equals signs).\n"
        "4. Write in professional, concise business language.\n"
        "5. Do NOT include any JSON, code, or markdown formatting.\n"
        "6. The report should be self-contained and ready to share with executives."
    )

# ----------------------- Execution -----------------------

if st.button("Run Readiness Analysis"):
    if not api_key:
        st.error("Please provide a Groq API key.")
    else:
        with st.spinner("Contacting Groq and generating assessments..."):
            dimension_assessments: Dict[str, str] = {}
            dimension_recommendations: Dict[str, str] = {}
            for dim, qs in question_sets.items():
                prompt = build_dimension_prompt(dim, qs, responses[dim])
                assessment_json = call_groq(prompt)
                dimension_assessments[dim] = assessment_json
                rec_prompt = build_recommendation_prompt(dim, assessment_json)
                rec_json = call_groq(rec_prompt, max_tokens=800)
                dimension_recommendations[dim] = rec_json

            synthesis_prompt = summarize_all(dimension_assessments, dimension_recommendations)
            synthesis_text = call_groq(synthesis_prompt, max_tokens=600)

        with st.spinner("Generating downloadable report..."):
            report_prompt = build_readable_report_prompt(
                dimension_assessments, dimension_recommendations, synthesis_text
            )
            readable_report = call_groq(report_prompt, max_tokens=3000)

        # Persist results in session state so they survive reruns
        st.session_state["dim_assessments"] = dimension_assessments
        st.session_state["dim_recommendations"] = dimension_recommendations
        st.session_state["synthesis"] = synthesis_text
        st.session_state["readable_report"] = readable_report

# Display results if available in session state
if "dim_assessments" in st.session_state:
    dimension_assessments = st.session_state["dim_assessments"]
    dimension_recommendations = st.session_state["dim_recommendations"]
    synthesis_text = st.session_state["synthesis"]

    readable_report = st.session_state["readable_report"]

    st.success("Analysis completed.")

    st.markdown("## Full Readiness Report")
    st.text(readable_report)

    with st.expander("Raw Dimension Data (JSON)"):
        for dim in question_sets.keys():
            st.markdown(f"**{dim} Assessment**")
            st.code(dimension_assessments[dim], language="json")
            st.markdown(f"**{dim} Recommendations**")
            st.code(dimension_recommendations[dim], language="json")

    st.markdown("## Integrated Synthesis")
    st.write(synthesis_text)

    # Download button
    st.download_button(
        label="\U0001F4E5 Download Report as TXT",
        data=readable_report,
        file_name="generative_ai_readiness_report.txt",
        mime="text/plain",
    )
else:
    st.info("Fill inputs and click 'Run Readiness Analysis'.")

# ----------------------- Footer -----------------------
st.markdown("---")
st.caption("Powered by Groq. For more info contact Dries Faems.")
