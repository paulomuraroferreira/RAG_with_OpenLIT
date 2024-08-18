import os
import PyPDF2
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
load_dotenv()
from logger_setup import logger

class PDFProcessor:
    def __init__(self, data_folder='./app/data/pdfs/', chunk_size=1000, chunk_overlap=200):
        self.data_folder = data_folder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = CharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap, separator=".\n")

    def extract_text_from_pdf(self, pdf_path):
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        return text

    def chunk_text(self, text):
        chunks = self.text_splitter.split_text(text)
        return chunks

    def process_all_pdfs(self):
        all_documents = []
        for filename in os.listdir(self.data_folder):
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(self.data_folder, filename)
                pdf_text = self.extract_text_from_pdf(pdf_path)
                chunks = self.chunk_text(pdf_text)

                documents = [Document(page_content=chunk, metadata={"source": filename}) for chunk in chunks]
                all_documents.extend(documents)
        return all_documents


class ChromaStore:
    def __init__(self, collection_name="rag_collection"):
        self.collection_name = collection_name
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        self.pdf_processor = PDFProcessor()
        self.vector_store = Chroma(
            collection_name="rag_collection",
            embedding_function=self.embeddings,
            persist_directory="./app/data/vector_store/",
        )

    def store_documents(self):
        self.documents = self.pdf_processor.process_all_pdfs()
        self.vector_store.add_documents(documents=self.documents)

    def get_vector_store(self):
        return self.vector_store.as_retriever()


if __name__ == "__main__":

    chroma_store = ChromaStore()
    chroma_store.store_documents()
    retriever = chroma_store.get_vector_store()
    question = "What is QLoRA"
    answer = retriever.invoke(question)
    logger.info(answer)
