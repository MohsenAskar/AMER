
import streamlit as st
import spacy
from spacy import displacy
import pandas as pd
from faker import Faker
import networkx as nx
import matplotlib.pyplot as plt
from translate import Translator
from summarizer import Summarizer
import itertools
import base64
import requests
import openpyxl
import textwrap 


st.set_page_config(layout='wide')

# Function to convert image to base64
def image_to_base64(image_path):
    with open(image_path, 'rb') as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# Display your image and name in the top right corner
image_path = "cartoon.JPG"

image_base64 = image_to_base64(image_path)
st.markdown(
    f"""
    <style>
    .header {{
        display: flex;
        justify-content: center; /* Center horizontally */
        align-items: center; /* Center vertically */
        padding: 10px;
        flex-direction: column; /* Stack items vertically */
        position: absolute;
        top: 15%;  /* Adjust vertical position to be slightly higher */
        right: -10%; /* Horizontal center */
        transform: translate(-50%, -40%); /* Adjust to keep the element centered with the new top value */
    }}
    .header img {{
        border-radius: 50%;
        width: 50px;
        height: 50px;
        margin-bottom: 5px; /* Space between image and text */
    }}
    .header-text {{
        font-size: 16px;
        font-weight: normal; /* Regular weight for text */
        text-align: center;
    }}
    </style>
    <div class="header">
        <img src="data:image/jpeg;base64,{image_base64}" alt="Mohsen Askar">
        <div class="header-text">Developed by: Mohsen Askar</div>
    </div>
    """,
    unsafe_allow_html=True
)

# Webapp description
# In the sidebar
st.sidebar.title("Modules Description")

with st.sidebar.expander("1. 💊 Recognize Entities"):
    st.write("""
    - Recognizes medical entities in the text such as drugs, diagnoses, etc.
    - You can specify which type of entities to show or to show them all.
    """)

with st.sidebar.expander("2. 📋 Correct ICD Codes"):
    st.write("""
    - Suggests the correct ICD codes for the diagnoses mentioned in the text.
    """)

with st.sidebar.expander("3. 💉 Inject ATC Codes"):
    st.write("""
    - Inserts the correct ATC codes to the substances and drug names in the text.
    """)

with st.sidebar.expander("4. 🕵️ Deidentify Patient's Information"):
    st.write("""
    - Identifies patient's sensitive info.
    - Inserts fake names, dates, etc. in the patient records.
    """)

with st.sidebar.expander("5. ⚠️ Check DDIs"):
    st.write("""
    - Checks for possible drug-drug interactions between the drugs mentioned in the text.
    - Red: severe, Orange: moderate, Green: possible DDIs.
    """)

with st.sidebar.expander("6. 📈 Visualize Comorbidity Progression"):
    st.write("""
    - Draws a directed network of the comorbidities progression in chronological order from the text.
    """)

with st.sidebar.expander("7. 🧪 Dose Checker"):
    st.write("""
    - Returns the correct drug dosage for the identified drugs in the text. Source of doses: Renal Drug Handbook, fifth edition.
    """)

with st.sidebar.expander("8. 💊⚠️ Dose Checker in Renal Impairment"):
    st.write("""
    - Returns the correct drug dosage in case of renal impairment. Source of doses: Renal Drug Handbook, fifth edition.
    """)

with st.sidebar.expander("9. 📝 EHR Summarizer"):
    st.write("""
    - Summarizes the patient records to 50% of the original text.
    """)

with st.sidebar.expander("10. 🏥 Structure EHR"):
    st.write("""
    - Divides the EHR into episodes of hospital admissions.
    - Extracts the most relevant information from each episode.
    """)

with st.sidebar.expander("11. 🚑 Find Side Effects (soon)"):
    st.write("""
    - Identifies potential side effects in the text.
    """)

# Cached model loading functions
@st.cache_resource
def load_nlp_model():
    return spacy.load(r'C:\Users\mas082\OneDrive - UiT Office 365\Desktop\VS_Code\NLP\Train_Norsk_Medical_NER\NER_Model')

nlp = load_nlp_model()

@st.cache_resource
def load_deidentify_model():
    return spacy.load('nb_core_news_sm')

nlp_deidentify = load_deidentify_model()

@st.cache_resource
def load_summarizer_model():
    return Summarizer('distilbert-base-uncased')

summarizer_model = load_summarizer_model()

# Cached data loading functions
@st.cache_data
def load_icd10_data():
    icd10_data= pd.read_csv(r"C:\Users\mas082\OneDrive - UiT Office 365\Desktop\ICD codes\ICD_Names.csv")
    # Ensure 'diagnosis' column is lowercase
    icd10_data['diagnosis'] = icd10_data['diagnosis'].str.lower()
    return icd10_data

@st.cache_data
def load_atc_data():
    return pd.read_csv(r'C:\Users\mas082\OneDrive - UiT Office 365\Desktop\VS_Code\Extract_From_XML\ATC_Injector.csv')  

@st.cache_data
def load_ddi_data():
    return pd.read_csv(r'C:\Users\mas082\OneDrive - UiT Office 365\Desktop\VS_Code\NLP\NER_Applications\Datasets\DDIs_2_Columns.csv')  

@st.cache_data
def load_renal_data():
    return pd.read_csv(r"C:\Users\mas082\OneDrive - UiT Office 365\Desktop\VS_Code\NLP\NER_Applications\Datasets\Drug_Dose_In_Renal_Impairment_To_Use.csv")  

# Initialize Faker instance
fake = Faker()

# Load translators
@st.cache_resource
def load_translators():
    translator_to_english = Translator(from_lang="no", to_lang="en")
    translator_to_norwegian = Translator(from_lang="en", to_lang="no")
    return translator_to_english, translator_to_norwegian

translator_to_english, translator_to_norwegian = load_translators()

# Define functions
def recognize_entities(text, entity_types):
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        if ent.label_ in entity_types:
            entities.append((ent.text, ent.label_))
    return entities, doc

def extract_diagnoses(text):
    doc = nlp(text)
    diagnoses = [ent.text for ent in doc.ents if ent.label_ == 'CONDITION']
    return diagnoses

def get_icd_codes_and_diagnoses(diagnosis, icd10_data):
    diagnosis = diagnosis.lower()
    matching_rows = icd10_data[icd10_data['diagnosis'].str.contains(diagnosis)]
    if not matching_rows.empty:
        icd_codes_and_diagnoses = list(matching_rows.itertuples(index=False, name=None))
        return icd_codes_and_diagnoses
    else:
        return None

def assign_icd_codes(text, icd10_data):
    diagnoses = extract_diagnoses(text)
    icd_codes_and_diagnoses = {diagnosis: get_icd_codes_and_diagnoses(diagnosis, icd10_data) for diagnosis in diagnoses}
    return icd_codes_and_diagnoses

def correct_icd_codes(text, icd10_data):
    icd_codes_and_diagnoses = assign_icd_codes(text, icd10_data)
    return icd_codes_and_diagnoses

def summarize_text(text, summary_proportion=0.5):
    if text:
        try:
            # Translate Norwegian text to English in chunks
            translated_text_parts = []
            for piece in textwrap.wrap(text, 500):
                translated_piece = translator_to_english.translate(piece)
                translated_text_parts.append(translated_piece)
            translated_text = " ".join(translated_text_parts)
        except Exception as e:
            st.error(f"Translation to English failed: {e}")
            return

        try:
            # Perform summarization on the English text
            summary_english = summarizer_model(translated_text, min_length=60, max_length=500)
            if isinstance(summary_english, list):
                summary_english = " ".join(summary_english)
        except Exception as e:
            st.error(f"Summarization failed: {e}")
            return

        try:
            # Translate English summary back to Norwegian in chunks
            summary_norwegian_parts = []
            for piece in textwrap.wrap(summary_english, 500):
                translated_piece = translator_to_norwegian.translate(piece)
                summary_norwegian_parts.append(translated_piece)
            summary_norwegian = " ".join(summary_norwegian_parts)
        except Exception as e:
            st.error(f"Translation to Norwegian failed: {e}")
            return

        return f'Summary: {summary_norwegian}'

def dose_check(text, renal_dataset):
    doc = nlp(text)
    drugs_normal = [ent.text for ent in doc.ents if ent.label_ == 'SUBSTANCE']
    drug_dose_map_normal = {}
    for drug in drugs_normal:
        matching_rows = renal_dataset[renal_dataset['Short_Drug'].str.lower() == drug.lower()]
        if not matching_rows.empty:
            dose_text = matching_rows["Normal dose"].iloc[0]
            dose_parts = dose_text.split(". ")
            formatted_dose = "\n".join([f"  - {part.strip()}." for part in dose_parts if part.strip()])
            drug_dose_map_normal[drug] = formatted_dose
    return drug_dose_map_normal

def get_renal_doses(text, renal_dataset):
    doc = nlp(text)
    drugs_renal = [ent.text for ent in doc.ents if ent.label_ == 'SUBSTANCE']
    drug_dose_map = {}
    for drug in drugs_renal:
        matching_rows = renal_dataset[renal_dataset['Short_Drug'].str.lower() == drug.lower()]
        if not matching_rows.empty:
            dose_text = matching_rows["Dose in renal impairment GFR (mL/min)"].iloc[0]
            dose_parts = dose_text.split(". ")
            formatted_dose = "\n".join([f"  - {part.strip()}." for part in dose_parts if part.strip()])
            drug_dose_map[drug] = formatted_dose
    return drug_dose_map

def atc_code_injection(text, atc_df):
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents if ent.label_ in ['SUBSTANCE', 'CONDITION']]
    entity_atc_dict = {}
    for entity in entities:
        entity_lower = entity.lower()
        atc_code = get_atc_code(entity_lower, atc_df)
        if atc_code is not None and entity_lower not in entity_atc_dict:
            entity_atc_dict[entity_lower] = atc_code
    for entity, atc_code in entity_atc_dict.items():
        styled_atc_code = f'<mark style="background-color: green; color: white;">(ATC code: {atc_code})</mark>'
        text = text.replace(entity, f'{entity} {styled_atc_code}')
    return text

def get_atc_code(entity, atc_df):
    matching_rows = atc_df[(atc_df['virkestoff'] == entity) | (atc_df['varenavn'] == entity)]
    if not matching_rows.empty:
        return matching_rows['atckode'].iloc[0]
    else:
        return None

def patient_identification(text):
    doc = nlp_deidentify(text)
    identified_names = [ent.text for ent in doc.ents if ent.label_ == 'PER']
    identified_addresses = [ent.text for ent in doc.ents if ent.label_ == 'LOC']
    name_map = {name: fake.name() for name in identified_names}
    address_map = {address: fake.address() for address in identified_addresses}
    anonymized_data = text
    for real_name, fake_name in name_map.items():
        anonymized_data = anonymized_data.replace(real_name, f'<mark style="background-color: red; color: white;">{fake_name}</mark>')
    for real_address, fake_address in address_map.items():
        anonymized_data = anonymized_data.replace(real_address, f'<mark style="background-color: blue; color: white;">{fake_address}</mark>')
    return anonymized_data

def check_and_visualize_ddi(text, ddi_data):
    doc = nlp(text)
    substances = [ent.text.lower() for ent in doc.ents if ent.label_ == 'SUBSTANCE']
    if len(substances) < 2:
        st.write("Not enough substances found to check for interactions.")
        return
    results = []
    interactions = set()
    for substance1, substance2 in itertools.combinations(substances, 2):
        interaction = ddi_data[((ddi_data['lm1'] == substance1) & 
                                (ddi_data['lm2'] == substance2)) |
                               ((ddi_data['lm1'] == substance2) & 
                                (ddi_data['lm2'] == substance1))]
        for row in interaction.itertuples():
            interaction_key = (substance1, substance2, row.grad)
            if interaction_key not in interactions:
                interactions.add(interaction_key)
                results.append({"Substance 1": substance1, 
                                "Substance 2": substance2, 
                                "Grade": row.grad})
    color_dict = {1: 'red', 2: 'orange', 3: 'green'}
    ddi_df = pd.DataFrame(results)
    if ddi_df.empty:
        st.write("No drug-drug interactions found.")
        return
    ddi_df['Grade'] = ddi_df['Grade'].map(lambda x: f'<span style="color: {color_dict.get(x, "black")}; font-weight: bold;">{x}</span>')
    st.markdown(ddi_df.to_html(escape=False), unsafe_allow_html=True)

def visualize_comorbidity(text):
    doc = nlp(text)
    conditions_with_positions = [(ent.text, ent.start_char) for ent in doc.ents if ent.label_ == 'CONDITION']
    conditions = []
    seen = set()
    for condition, _ in sorted(conditions_with_positions, key=lambda x: x[1]):
        if condition not in seen:
            seen.add(condition)
            conditions.append(condition)
    if not conditions:
        st.write("No conditions found to visualize.")
        return None
    graph = nx.DiGraph()
    for i in range(len(conditions)-1):
        graph.add_edge(conditions[i], conditions[i+1])
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, edge_color='black', width=1, linewidths=1,
            node_size=500, node_color='pink', alpha=0.9, with_labels=True)
    plt.axis('off')
    plt.tight_layout()
    return plt.gcf()

def find_sideeffects(text):
    st.write("This function is not yet implemented.")
    return

def structure_ehr(text):
    st.write("This function is not yet implemented.")
    return

# Main app logic
st.title("EHR Text Processing 📋 (Demo)")
user_input = st.text_area("Paste the EHR text here")

# Create a checkbox group for the user to select which entity types to recognize
st.write("Select the entity type:")
all_checkbox_col, disease_checkbox_col, substance_checkbox_col, physiology_checkbox_col = st.columns(4)
procedure_checkbox_col, anatomi_checkbox_col, microorganism_checkbox_col = st.columns(3)

with all_checkbox_col:
    all_checkbox = st.checkbox("All")

with anatomi_checkbox_col:
    anatomi_checkbox = st.checkbox("ANAT_LOC")

with disease_checkbox_col:
    disease_checkbox = st.checkbox("CONDITION")

with microorganism_checkbox_col:
    microorganism_checkbox = st.checkbox("MICROORGANISM")

with physiology_checkbox_col:
    physiology_checkbox = st.checkbox("PHYSIOLOGY")

with procedure_checkbox_col:
    procedure_checkbox = st.checkbox("PROCEDURE")

with substance_checkbox_col:
    substance_checkbox = st.checkbox("SUBSTANCE")

entity_types = []
if all_checkbox:
    entity_types = ["ANAT_LOC", "CONDITION", "MICROORGANISM", "PHYSIOLOGY", "PROCEDURE", "SUBSTANCE"]
else:
    if anatomi_checkbox:
        entity_types.append("ANAT_LOC")
    if disease_checkbox:
        entity_types.append("CONDITION")
    if microorganism_checkbox:
        entity_types.append("MICROORGANISM")
    if physiology_checkbox:
        entity_types.append("PHYSIOLOGY")
    if procedure_checkbox:
        entity_types.append("PROCEDURE")
    if substance_checkbox:
        entity_types.append("SUBSTANCE")

# Buttons for different actions
if st.button('Recognize Entities'):
    entities, doc = recognize_entities(user_input, entity_types)
    st.write("Number of recognized entities:", len(entities))
    if len(entities) > 0:
        st.write("Recognized entities:")
        entity_table = []
        for entity in entities:
            entity_table.append({"Entity": entity[0], "Type": entity[1]})
        st.table(entity_table)
        colors = {"ANAT_LOC": "#FAC748", "CONDITION": "#FF5733", "MICROORGANISM": "#47D1D1", "PHYSIOLOGY": "#2E86C1", "PROCEDURE": "#BB8FCE", "SUBSTANCE": "#27AE60"}
        options = {"ents": entity_types, "colors": colors}
        html = displacy.render(doc, style="ent", options=options)
        html = html.replace("\n", " ")
        st.write(f"{html}", unsafe_allow_html=True)

correct_icd_codes_button, inject_atc_codes_button, deidentify_patients_button = st.columns(3)
check_ddis_button, visualize_comorbidity_button, dose_checker_button = st.columns(3)
correct_renal_dose_button, summarize_ehr_button, structure_ehr_button = st.columns(3)
find_side_effects_button = st.button('Find Side Effects')

if correct_icd_codes_button.button('Correct ICD Codes'):
    icd10_data = load_icd10_data()
    result = correct_icd_codes(user_input, icd10_data)
    st.write(result)

if inject_atc_codes_button.button('Inject ATC Codes'):
    atc_df = load_atc_data()
    result = atc_code_injection(user_input, atc_df)
    st.markdown(result, unsafe_allow_html=True)

if deidentify_patients_button.button('Deidentifier'):
    result = patient_identification(user_input)
    st.markdown(result, unsafe_allow_html=True)

if check_ddis_button.button('Check DDIs'):
    ddi_data = load_ddi_data()
    check_and_visualize_ddi(user_input, ddi_data)

if visualize_comorbidity_button.button('Visualize Comorbidity'):
    fig = visualize_comorbidity(user_input)
    if fig:
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(fig)

if dose_checker_button.button('Dose Checker'):
    renal_dataset = load_renal_data()
    result = dose_check(user_input, renal_dataset)
    if result:
        st.write("***Correct Doses:***")
        for drug, dose in result.items():
            st.markdown(f"**{drug}:**")
            st.write(dose)
    else:
        st.write("No drugs detected in the text.")

if correct_renal_dose_button.button('Correct Dose Renal'):
    renal_dataset = load_renal_data()
    result = get_renal_doses(user_input, renal_dataset)
    if result:
        st.write("***Correct Doses for Renal Impairment according to GFR value:***")
        for drug, dose in result.items():
            st.markdown(f"**{drug}:**")
            st.write(dose)
    else:
        st.write("No drugs detected in the text.")

if summarize_ehr_button.button('Summarize EHR'):
    result = summarize_text(user_input)
    st.write(result)

if structure_ehr_button.button('Structure EHR'):
    result = structure_ehr(user_input)
    st.write(result)

if find_side_effects_button:
    result = find_sideeffects(user_input)
    st.write(result) 
