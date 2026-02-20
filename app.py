import streamlit as st
import fitz  # PyMuPDF
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Check if API key is available
if not api_key:
    st.error("Please set the GOOGLE_API_KEY in your environment variables or .env file.")
    st.stop()

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)

def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    return "".join([page.get_text() for page in doc])

st.set_page_config(page_title="Resume Transformer", page_icon="📝")
st.title("🌍 Resume-to-LinkedIn Transformer")

uploaded_file = st.file_uploader("Upload your Resume (PDF)", type="pdf")

if uploaded_file:
    with st.spinner("Processing..."):
        resume_text = extract_text_from_pdf(uploaded_file)
        
        template = """
        You are a professional LinkedIn Profile Writer. 
        Take the following resume text, which may be in any language, and:
        1. Translate all content into professional English.
        2. Format it into a perfect LinkedIn Profile.
        
        Sections: Headline, About (1st person), Experience (bullet points), Skills.
        Resume Text: {resume_content}
        """
        
        prompt = PromptTemplate(input_variables=["resume_content"], template=template)
        chain = prompt | llm
        response = chain.invoke({"resume_content": resume_text})
        
        st.success("Conversion Complete!")
        st.markdown(response.content)
