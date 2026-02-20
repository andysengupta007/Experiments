import streamlit as st
import fitz  # PyMuPDF
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

YOUR_API_KEY="AIzaSyDKyXgGSOl4LlR80giKAcu9g53mhjGDCYs"

# Setup LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=YOUR_API_KEY)

def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

st.title("🌍 Resume-to-LinkedIn Transformer")
st.subheader("Convert any language resume to an English LinkedIn Profile")

uploaded_file = st.file_uploader("Upload your Resume (PDF)", type="pdf")

if uploaded_file:
    with st.spinner("Analyzing and translating..."):
        # 1. Extract
        resume_text = extract_text_from_pdf(uploaded_file)
        
        # 2. Define the Prompt
        template = """
        You are a professional LinkedIn Profile Writer. 
        Take the following resume text, which may be in any language, and:
        1. Translate all content into professional English.
        2. Format it into a perfect LinkedIn Profile.
        
        Provide the output in these sections:
        - **Headline**: Catchy, keyword-optimized (max 220 chars).
        - **About**: A compelling 1st-person summary.
        - **Experience**: List roles with bullet points focused on achievements.
        - **Skills**: A list of the top 10 relevant skills.

        Resume Text: {resume_content}
        """
        
        prompt = PromptTemplate(input_variables=["resume_content"], template=template)
        chain = prompt | llm
        
        # 3. Generate
        response = chain.invoke({"resume_content": resume_text})
        
        st.markdown("---")
        st.markdown("### 🚀 Your New English LinkedIn Profile")
        st.write(response.content)
