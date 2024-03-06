import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from deep_translator import GoogleTranslator
from langdetect import detect
import speech_recognition as sr
# from google.cloud import language_v1

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

st.set_page_config("Chat PDF")  # Move this line to the beginning

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, you can add some more meaning and do not just tell the response as it is, make sure to provide all the details, if the answer is not in
    provided context just say, "Answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   temperature=0.9)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question, detected_language):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question}
        , return_only_outputs=True)
    print('response',response)
    st.header("Answer")
    translated = GoogleTranslator(source='en', target=detected_language).translate(response["output_text"])
    st.write(translated)

def main():
    st.title("Need Assistance ? Let us know more about your queries 💁")
    st.header("Ask your doubts here ⬇️")
    st.markdown("""
        <style>
            .stButton>button {
                margin-top: 28px;
            }
        </style>
        """, unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])

    with col1:
        user_question = st.text_input("")

    with col2:
        button_clicked = st.button("Speak")  # Display the button

    if button_clicked:
        with st.spinner("Listening..."):
            r = sr.Recognizer()
            with sr.Microphone() as source:
                audio_input = None
                audio = r.listen(source)
                try:
                    audio_input = r.recognize_google(audio)
                    st.write("You said:", audio_input)
                    detected_language = detect(audio_input)
                    print('detected language',detected_language)
                    translated = GoogleTranslator(source=detected_language, target='en').translate(audio_input)
                    user_question = translated  
                    user_input(translated, detected_language)
                except sr.UnknownValueError:
                    st.write("Sorry, could not understand audio.")
                except sr.RequestError as e:
                    st.write("Error occurred; {0}".format(e))
            st.success("Listened")
        # button_clicked = not button_clicked
        # break

    elif user_question:
            detected_language = detect(user_question)
            translated = GoogleTranslator(source='auto', target='en').translate(user_question)
            user_input(translated, detected_language)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF dataset Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()