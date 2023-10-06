import streamlit as st
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import Cohere
from PyPDF2 import PdfReader
from langchain.embeddings import CohereEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv


def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF")

    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    # extract the text
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=0,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        embeddings = CohereEmbeddings(
            cohere_api_key='FoY9OqiB9Zpsm1siHOGOYHrgXn2ExUHwn9YVSmzk')
        doc_search = Chroma.from_texts(chunks, embeddings)
        llm = Cohere(cohere_api_key='FoY9OqiB9Zpsm1siHOGOYHrgXn2ExUHwn9YVSmzk')
        retriever = doc_search.as_retriever(search_kwargs={"k": 1})
        ConversationBufferMemory(
            memory_key="chat_history", return_messages=True)
        qa = ConversationalRetrievalChain.from_llm(llm, retriever)
        if qa:
            question = st.text_input("Ask a question")
            button = st.button("Answer")
            if question and button:
                chat_history = []
                result = qa(
                    {"question": question, "chat_history": chat_history})
                answer = result['answer']
                chat_history = [(question, result["answer"])]
                st.write("Answer:", answer)


if __name__ == '__main__':
    main()
