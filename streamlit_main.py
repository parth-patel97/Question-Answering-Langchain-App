from dotenv import dotenv_values
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader


config = dotenv_values(".env")

OPENAI_API_KEY = config["OPENAI_API_KEY"]

# Set up the Streamlit app
st.title("Question Answering App")
st.write("Enter a long text and convert it to a text file for further processing.")

# Create input box for long text
long_text = st.text_area("Enter long text", height=300)

# Create a placeholder for the qa variable
qa = None

# Button to convert long text to text file
if st.button("Convert to Text File"):
    # Save the long text to a file
    with open("input.txt", "w", encoding="utf-8") as file:
        file.write(long_text)

    loader = TextLoader("input.txt")
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=10, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    docsearch = Chroma.from_documents(texts, embeddings)
    qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=OPENAI_API_KEY),
                                     chain_type="stuff", retriever=docsearch.as_retriever(search_kwargs={"k": 1}))

    st.write("Process is completed successfully!")

# Create input box for asking questions
question = st.text_input("Ask a question")

# Process user input when question is provided
if st.button("Get Answer") and question:
    result = qa.run(question)
    answer = result

    # Display the answer
    st.write("Answer:", answer)
