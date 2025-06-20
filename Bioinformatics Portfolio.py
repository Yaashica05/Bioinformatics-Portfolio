import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import time
from streamlit_option_menu import option_menu # For the professional menu

# --- Page Configuration ---
st.set_page_config(
    page_title="Bioinformatics Portfolio",
    page_icon=" B",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Styling ---
def local_css():
    st.markdown("""
    <style>
    /* --- FONT, COLOR & GLOBAL STYLES --- */
    html, body, [class*="st-"], .stApp {
        font-family: 'Times New Roman', Times, serif !important;
        font-weight: bold !important;
        color: #000000 !important; /* Black text */
    }

    /* --- BACKGROUNDS & LAYOUT --- */
    .stApp {
        background-color: #E3F2FD; /* Light Blue Background */
    }

    /* Main content area card */
    .main .block-container {
        background-color: #FFFFFF;
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border: 1px solid #B0BEC5;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #FFFFFF !important; /* White sidebar background */
        border-right: 2px solid #B0BEC5;
    }

    /* --- TYPOGRAPHY & TITLES --- */
    h1, h2, h3 {
        color: #0d47a1 !important; /* Darker professional blue for titles */
    }

    h1 {
        font-size: 38px !important;
        border-bottom: 3px solid #007BFF;
        padding-bottom: 10px;
    }
    h2, h3 { font-size: 30px !important; }

    /* --- WIDGETS & COMPONENTS --- */
    .stInfo { /* Targeting st.info boxes */
        background-color: #e3f2fd;
        border-left: 8px solid #007BFF;
        border-radius: 8px;
    }
    
    .stButton>button {
        background-color: #007BFF;
        color: white !important;
        border-radius: 8px;
        border: none;
        padding: 12px 24px;
        transition: background-color 0.2s ease;
    }
    .stButton>button:hover { background-color: #0056b3; }
    
    .stMetric {
        background-color: #f5f5f5;
        border-left: 8px solid #007BFF;
        padding: 20px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Mock Data and Backend Functions (Unchanged) ---
def mock_predict_interaction(pathogen, protein):
    time.sleep(2)
    affinity_score = np.random.uniform(0.7, 0.98)
    compounds = pd.DataFrame({
        'Compound': ['Curcumin', 'Quercetin', 'Ginsenoside', 'Resveratrol'],
        'Source': ['Turmeric', 'Onion', 'Ginseng', 'Grapes'],
        'Predicted Affinity': np.random.uniform(0.6, 0.95, 4).round(2),
        'NPASS ID': [f'NPA00{np.random.randint(1000, 9999)}' for _ in range(4)]
    })
    return affinity_score, compounds.sort_values(by='Predicted Affinity', ascending=False)

def mock_analyze_ethics(justification):
    time.sleep(1)
    if any(word in justification.lower() for word in ['benefit', 'proactive', 'greater good']):
        return "Utilitarian", np.random.uniform(0.6, 0.9)
    elif any(word in justification.lower() for word in ['rights', 'consent', 'duty']):
        return "Deontological", np.random.uniform(0.6, 0.9)
    else:
        return "Undetermined", 0.5

def mock_epigenetic_simulation(generations, stress_factor):
    n_genes = 50
    initial_state = np.random.choice([0, 1], size=n_genes, p=[0.8, 0.2])
    history = [initial_state]
    for _ in range(generations - 1):
        last_gen = history[-1].copy()
        change_prob = 0.05 + (stress_factor / 100) * 0.2
        flips = np.random.random(size=n_genes) < change_prob
        last_gen[flips] = 1 - last_gen[flips]
        history.append(last_gen)
    return np.array(history)

def mock_fetch_pubmed(disease):
    time.sleep(2.5)
    papers = [
        {"title": f"The role of GeneA in {disease}", "journal": "Nature", "genes": ["GeneA", "GeneB"]},
        {"title": f"A new pathway involving GeneB and GeneC in {disease}", "journal": "Cell", "genes": ["GeneB", "GeneC"]},
        {"title": f"Therapeutic potential of targeting GeneA in {disease}", "journal": "Science", "genes": ["GeneA", "GeneD"]},
        {"title": f"Metabolomic profiling of {disease} patients reveals GeneC importance", "journal": "JBC", "genes": ["GeneC", "GeneE"]},
        {"title": f"Co-expression of GeneA and GeneB in severe {disease}", "journal": "Nature", "genes": ["GeneA", "GeneB"]},
    ]
    summary = f"Recent literature on **{disease}** highlights a strong focus on *GeneA* and *GeneB*. A key emerging theme is the interaction between these genes, potentially forming a new therapeutic target pathway."
    return papers, summary

def mock_parse_lab_notes(notes):
    time.sleep(1.5)
    entries = []
    lines = notes.split('\n')
    current_date = "Unknown"
    for line in lines:
        if line.startswith("Date:"):
            current_date = line.split("Date:")[1].strip()
        if "exp" in line.lower() and ":" in line:
            try:
                exp_id, procedure = line.split(":", 1)
                tags = []
                if 'pcr' in procedure.lower(): tags.append('PCR')
                if 'incubate' in procedure.lower(): tags.append('Incubation')
                if 'ng/ul' in procedure.lower(): tags.append('Quantification')
                entries.append({
                    "Date": current_date, "Experiment ID": exp_id.strip(),
                    "Procedure": procedure.strip(), "Tags": ", ".join(tags)
                })
            except:
                continue
    return pd.DataFrame(entries)

# --- UI for Each Page (Functions are now properly named) ---

def page_home():
    st.title("Welcome to the Bioinformatics Portfolio")
    st.info("""
    This interactive portfolio showcases innovative projects at the intersection of bioinformatics, data science, and development.
    Navigate through the demos using the professional menu on the left.
    """)
    st.markdown("""
    **Project Highlights:**
    - **Predictive Modeling:** See how we can predict protein interactions.
    - **Ethical Simulation:** Explore complex bioethical dilemmas interactively.
    - **Gamified Learning:** Experience drug discovery in a fun, game-like format.
    - **Automated Lab Tools:** Discover how AI can streamline lab work and analysis.
    """)

def page_protein_predictor():
    st.title("Protein–Pathogen Interaction Predictor")
    st.info("Predicts protein targets in pathogens and suggests natural compounds with potential binding affinity.")
    col1, col2 = st.columns(2)
    with col1:
        pathogen = st.text_input("Enter Pathogen Name", "SARS-CoV-2")
    with col2:
        protein = st.selectbox("Select Target Protein (mocked)", ["Spike Glycoprotein", "Nsp12", "Mpro", "Nsp5"])
    if st.button("Predict Interaction & Find Compounds »"):
        with st.spinner(f"Analyzing {protein} from {pathogen}..."):
            affinity_score, compounds_df = mock_predict_interaction(pathogen, protein)
        st.success(f"Prediction Complete for **{protein}**!")
        st.metric(label="Predicted Host-Protein Binding Affinity", value=f"{affinity_score:.2f}")
        st.subheader("🌿 Suggested Natural Compounds")
        st.dataframe(compounds_df, use_container_width=True)
        fig = px.bar(compounds_df, x='Compound', y='Predicted Affinity', color='Source', title="Binding Affinity of Natural Compounds")
        st.plotly_chart(fig, use_container_width=True)

def page_dilemma_simulator():
    st.title("Bioethical Dilemma Simulator")
    st.info("An interactive simulator for bioethical dilemmas, using mock NLP to gauge your reasoning.")
    dilemmas = {
        "Gene Editing": {"text": "A new CRISPR therapy can cure a fatal genetic disorder, but costs $2M per patient with unknown long-term effects. Should it be approved?", "options": ["Approve (focus on saving lives now)", "Deny (focus on potential harm)"]},
        "AI Diagnosis": {"text": "An AI model is 99% accurate in diagnosing a rare cancer, far better than human doctors, but it cannot explain its reasoning. Should hospitals rely on this 'black box' AI?", "options": ["Rely on the AI (prioritize accuracy)", "Do not rely on it (prioritize transparency)"]}
    }
    if 'dilemma_idx' not in st.session_state:
        st.session_state.dilemma_idx = 0; st.session_state.scores = []
    dilemma_keys = list(dilemmas.keys())
    if st.session_state.dilemma_idx < len(dilemma_keys):
        key = dilemma_keys[st.session_state.dilemma_idx]
        dilemma = dilemmas[key]
        st.subheader(f"Dilemma: {key}")
        st.write(dilemma["text"])
        choice = st.radio("Your Decision:", dilemma["options"], key=f"choice_{key}")
        justification = st.text_area("Justify your reasoning:", key=f"justify_{key}")
        if st.button("Submit Decision »"):
            framework, score = mock_analyze_ethics(justification)
            st.session_state.scores.append({"Dilemma": key, "Framework": framework, "Score": score})
            st.session_state.dilemma_idx += 1
            st.success(f"Reasoning leans towards a **{framework}** framework. Next dilemma...")
            time.sleep(2); st.rerun()
    else:
        st.success("All dilemmas completed!")
        st.subheader("Your Ethical Journey")
        scores_df = pd.DataFrame(st.session_state.scores)
        st.dataframe(scores_df)
        fig = px.line(scores_df, x='Dilemma', y='Score', color='Framework', markers=True, title="Your Ethical Framework Trend", range_y=[0,1])
        st.plotly_chart(fig, use_container_width=True)
        if st.button("Restart Simulator »"):
            st.session_state.dilemma_idx = 0; st.session_state.scores = []; st.rerun()

def page_epigenetic_tracker():
    st.title("Epigenetic Memory Tracker")
    st.info("Simulates how environmental factors affect epigenetic modifications across generations.")
    col1, col2 = st.columns(2)
    with col1:
        generations = st.slider("Number of Generations", 5, 100, 20)
    with col2:
        stress_factor = st.slider("Environmental Stress Factor (%)", 0, 100, 10)
    if st.button("Run Simulation »"):
        with st.spinner("Simulating epigenetic drift..."):
            methylation_history = mock_epigenetic_simulation(generations, stress_factor)
        st.success("Simulation complete!")
        fig = go.Figure(data=go.Heatmap(z=methylation_history, colorscale='Viridis'))
        fig.update_layout(title=f'Epigenetic Memory Over {generations} Generations', xaxis_title="Genomic Loci", yaxis_title="Generation", yaxis=dict(autorange='reversed'))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Interpretation:** The heatmap shows methylation status (dark=methylated) for genes across generations. Watch how patterns emerge or disappear based on the stress factor.")

def page_ai_assistant():
    st.title("AI-Powered Research Assistant")
    st.info("Scrapes PubMed abstracts for a topic and builds a real-time co-citation network.")
    disease = st.text_input("Enter a Disease or Topic", "Tularemia")
    if st.button("Analyze Literature »"):
        with st.spinner(f"Mining PubMed for '{disease}'..."):
            papers, summary = mock_fetch_pubmed(disease)
        st.success("Analysis Complete!")
        st.subheader("AI-Generated Summary")
        st.markdown(summary)
        gene_counts = pd.Series([gene for paper in papers for gene in paper['genes']]).value_counts()
        st.subheader("Trending Genes/Proteins")
        st.dataframe(gene_counts.reset_index().rename(columns={'index': 'Gene', 0: 'Mentions'}))
        st.subheader("Co-citation Network")
        G = nx.Graph()
        for paper in papers:
            for i in range(len(paper['genes'])):
                for j in range(i + 1, len(paper['genes'])):
                    G.add_edge(paper['genes'][i], paper['genes'][j])
        pos = nx.spring_layout(G, k=0.9)
        fig = go.Figure()
        # Drawing logic is preserved
        edge_x, edge_y = [], [];
        for edge in G.edges():
            x0, y0 = pos[edge[0]]; x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines'))
        node_x, node_y, node_text, node_size = [], [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x); node_y.append(y)
            node_text.append(f'{node} ({gene_counts[node]} mentions)'); node_size.append(15 + gene_counts[node] * 5)
        fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='text', text=node_text, marker=dict(showscale=True, colorscale='YlGnBu', size=node_size, color=[gene_counts[node] for node in G.nodes()], colorbar=dict(thickness=15, title='Gene Mentions'), line_width=2)))
        fig.update_layout(title=f'Gene Co-citation Network for {disease}', showlegend=False, xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
        st.plotly_chart(fig, use_container_width=True)

def page_repurposing_game():
    st.title("Drug Repurposing Game")
    st.info("A gamified approach to drug repurposing. Match drugs to protein targets to score points.")
    game_data = {
        "EGFR (Lung Cancer)": {"drugs": ["Gefitinib", "Aspirin", "Metformin"], "correct": "Gefitinib"},
        "COX-2 (Inflammation)": {"drugs": ["Celecoxib", "Sildenafil", "Lisinopril"], "correct": "Celecoxib"},
        "ACE (Hypertension)": {"drugs": ["Lisinopril", "Insulin", "Atorvastatin"], "correct": "Lisinopril"}
    }
    if 'score' not in st.session_state:
        st.session_state.score = 0; st.session_state.target_idx = 0
    targets = list(game_data.keys())
    if st.session_state.target_idx < len(targets):
        target = targets[st.session_state.target_idx]
        data = game_data[target]
        st.subheader(f"Target: {target}")
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_drug = st.selectbox("Choose the best drug for this target:", data["drugs"])
        with col2:
            st.metric("Your Score", st.session_state.score)
        if st.button("Submit Answer »"):
            if selected_drug == data["correct"]:
                st.success(f"Correct! {selected_drug} is a known inhibitor."); st.session_state.score += 10
            else:
                st.error(f"Not quite. The best choice was {data['correct']}."); st.session_state.score -= 5
            st.session_state.target_idx += 1; time.sleep(1.5); st.rerun()
    else:
        st.balloons(); st.success(f"Game Over! Your final score is: {st.session_state.score}")
        if st.button("Play Again? »"):
            st.session_state.score = 0; st.session_state.target_idx = 0; st.rerun()

def page_lab_notebook():
    st.title("Lab Notebook Intelligence Tool")
    st.info("Takes unstructured lab notes and automatically structures them into searchable entries.")
    default_notes = "Date: 2023-10-26\nExp_045: Ran PCR on samples A-D. Used taq poly. 35 cycles.\nResults look good. Quantified samples using NanoDrop.\nSample A: 50.2 ng/ul\n\nDate: 2023-10-27\nExp_046: Cell culture work. Incubated plate #3 for 48h at 37C."
    notes = st.text_area("Paste your unstructured lab notes below:", default_notes, height=250)
    if st.button("Structure My Notes »"):
        if not notes.strip(): st.warning("Please enter some notes to structure.")
        else:
            with st.spinner("Applying NLP to parse your notes..."):
                structured_df = mock_parse_lab_notes(notes)
            st.success("Notes structured successfully!")
            st.dataframe(structured_df, use_container_width=True)
            csv = structured_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download as CSV", csv, 'structured_lab_notes.csv', 'text/csv')

def page_risk_estimator():
    st.title("Contamination Risk Estimator")
    st.info("Uses Bayesian-inspired logic to assess contamination risk in a multi-step lab protocol.")
    st.subheader("Select the steps in your experimental protocol:")
    protocol_steps = st.multiselect("Protocol Steps", ["Sample Collection", "DNA/RNA Extraction", "PCR Setup (Pre-Amp)", "PCR Amplification", "Post-PCR Handling", "Sequencing Prep"], default=["DNA/RNA Extraction", "PCR Setup (Pre-Amp)", "PCR Amplification"])
    st.subheader("Assess Your Lab Environment:")
    col1, col2 = st.columns(2)
    with col1:
        cleanliness = st.slider("Lab Cleanliness (1=Poor, 10=Sterile)", 1, 10, 7)
    with col2:
        experience = st.slider("Operator Experience (1=Novice, 10=Expert)", 1, 10, 8)
    if st.button("Calculate Contamination Risk »"):
        base_risk = {"Sample Collection": 0.15, "DNA/RNA Extraction": 0.10, "PCR Setup (Pre-Amp)": 0.20, "PCR Amplification": 0.05, "Post-PCR Handling": 0.25, "Sequencing Prep": 0.15}
        cleanliness_modifier = 1 + (10 - cleanliness) * 0.1
        experience_modifier = 1 + (10 - experience) * 0.05
        total_modifier = cleanliness_modifier * experience_modifier
        step_risks = {step: min(base_risk.get(step, 0) * total_modifier, 0.99) for step in protocol_steps}
        final_risk = 1 - np.prod([1 - risk for risk in step_risks.values()])
        st.subheader("Risk Assessment Results")
        fig = go.Figure(go.Indicator(mode="gauge+number", value=final_risk * 100, title={'text': "Overall Contamination Risk"}, gauge={'axis': {'range': [None, 100]}, 'steps': [{'range': [0, 25], 'color': 'lightgreen'}, {'range': [25, 50], 'color': 'yellow'}, {'range': [50, 100], 'color': 'red'}]}))
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Risk Breakdown & Suggestions")
        if final_risk > 0.5: st.error("High Risk Detected!")
        elif final_risk > 0.25: st.warning("Moderate Risk Detected. Consider improvements.")
        else: st.success("Low Risk. Good protocol and environment.")
        for step, risk in sorted(step_risks.items(), key=lambda item: item[1], reverse=True):
            if risk > 0.3: st.markdown(f"- **{step}**: High risk step ({risk:.1%}). **Suggestion:** Use dedicated pre/post PCR areas, filter tips.")
            elif risk > 0.15: st.markdown(f"- **{step}**: Moderate risk step ({risk:.1%}). **Suggestion:** Ensure gloves are changed frequently.")
            else: st.markdown(f"- **{step}**: Low risk step ({risk:.1%}).")

# --- Footer Function ---
def render_footer():
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; font-family: 'Times New Roman', serif;">
        <p style="font-size: 20px !important; font-weight: bold;">Thank You For Visiting!</p>
        <p style="font-size: 18px !important; font-weight: normal; color: #333;">© 2024 Bioinformatics Portfolio. All Rights Reserved.</p>
    </div>
    """, unsafe_allow_html=True)

# --- Main App ---
def main():
    local_css()

    project_options = {
        "Home": page_home,
        "Protein Predictor": page_protein_predictor,
        "Ethical Simulator": page_dilemma_simulator,
        "Epigenetic Tracker": page_epigenetic_tracker,
        "AI Research Assistant": page_ai_assistant,
        "Drug Repurposing Game": page_repurposing_game,
        "Lab Notebook Tool": page_lab_notebook,
        "Contamination Estimator": page_risk_estimator,
    }

    with st.sidebar:
        st.markdown("<h1 style='text-align: center; color: #0d47a1;'>Portfolio Menu</h1>", unsafe_allow_html=True)
        st.markdown("---")
        
        choice = option_menu(
            menu_title=None,
            options=list(project_options.keys()),
            icons=['house-door-fill', 'bullseye', 'clipboard2-heart-fill', 'dna', 'robot', 'joystick', 'journal-richtext', 'shield-shaded'],
            menu_icon="cast", default_index=0, orientation="vertical",
            styles={
                "container": {"padding": "0!important", "background-color": "#FFFFFF"},
                "icon": {"color": "#007BFF", "font-size": "24px"},
                "nav-link": {
                    "font-size": "20px", "font-family": "'Times New Roman', serif", "font-weight": "bold",
                    "text-align": "left", "margin": "5px", "padding": "10px", "--hover-color": "#E3F2FD"
                },
                "nav-link-selected": {"background-color": "#007BFF", "color": "white !important"},
            }
        )
        st.markdown("---")
        st.info("This portfolio uses mocked data for fast, interactive demonstrations.")

    project_options[choice]()
    render_footer()

if __name__ == "__main__":
    main()
