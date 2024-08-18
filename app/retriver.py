from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
from chunking import ChromaStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from logger_setup import logger

class RAGSystem:
    def __init__(self, model_name="gpt-4"):
        self.llm = ChatOpenAI(model_name=model_name)
        self.embeddings = OpenAIEmbeddings()
        self.chroma = ChromaStore()

    def define_rag_prompt(self):

        template = """
        You are a knowledgeable assistant. Below is some context information retrieved from relevant documents.

        Context:
        {context}

        Based on the above context, answer the following question:

        Question: {question}

        Answer:
        """

        rag_prompt = ChatPromptTemplate.from_template(template=template)
        
        return rag_prompt
    

    def format_context(self, list_of_documents: list):
        context = ""
        for document in list_of_documents:
            context += document.page_content + "\n"
        return context


    def setup_rag_chain(self, question):
        retriever = self.chroma.get_vector_store()
        retrieval_qa_chat_prompt = self.define_rag_prompt()
        output_parser = StrOutputParser()

        # Step 1: Set up the retrieval and context formatting
        retrieval_chain = RunnableLambda(
            lambda x: {"context": self.format_context(retriever.invoke(x))}
        )

        # Step 2: Set up the full RAG chain
        rag_chain = (
            retrieval_chain
            | RunnableParallel(
                context=RunnablePassthrough() | RunnableLambda(lambda x: x["context"]),
                question=lambda _: question,
            )
            | retrieval_qa_chat_prompt
            | self.llm
            | output_parser
        )

        # Step 3: Combine the RAG chain with the context retrieval
        full_chain = RunnableParallel(
            answer=rag_chain,
            context=retrieval_chain | RunnableLambda(lambda x: x["context"])
        )

        # Step 4: Invoke the chain and return the result
        return full_chain.invoke(question)

if __name__ == "__main__":
    rag_system = RAGSystem()
    result = rag_system.setup_rag_chain("What is QLora?")
    logger.info(f"Answer: {result['answer']}")
    logger.info(f"Context: {result['context']}")


