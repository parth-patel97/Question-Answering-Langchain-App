import streamlit as st
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import Cohere
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings import CohereEmbeddings

def process_long_text(long_text):
    # Save the long text to a file
    with open("input.txt", "w", encoding="utf-8") as file:
        file.write(long_text)

    loader = TextLoader("input.txt")
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=10, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    embeddings = CohereEmbeddings(
        cohere_api_key='FoY9OqiB9Zpsm1siHOGOYHrgXn2ExUHwn9YVSmzk')
    docsearch = Chroma.from_documents(texts, embeddings)
    llm = Cohere(cohere_api_key='FoY9OqiB9Zpsm1siHOGOYHrgXn2ExUHwn9YVSmzk')
    qa = RetrievalQA.from_chain_type(llm=llm,
                                     chain_type="stuff", retriever=docsearch.as_retriever(search_kwargs={"k": 1}))

    return qa


def question_answering_app():
    # Set up the Streamlit app
    st.title("Question Answering App")
    st.write("Enter a long text and convert it to a text file for further processing.")

    # Create input box for long text
    long_text = st.text_area("Enter long text", height=300)

    # Create a placeholder for the qa variable
    session_state = st.session_state
    if 'qa' not in session_state:
        session_state.qa = None

    # Button to convert long text to text file
    if st.button("Convert to Text File"):
        session_state.qa = process_long_text(long_text)
        st.write("Process is completed successfully!")

    # Process user input when question is provided
    if session_state.qa is not None:
        # Create input box for asking questions
        question = st.text_input("Ask a question")

        if question:
            result = session_state.qa.run(question)
            answer = result

            # Display the answer
            st.write("Answer:", answer)
        else:
            st.write("Please enter a question.")
    else:
        st.write("Please convert the long text to a text file first.")


if __name__ == "__main__":
    question_answering_app()
