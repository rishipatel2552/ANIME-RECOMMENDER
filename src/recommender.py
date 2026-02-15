from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from src.prompt_template import get_anime_prompt  # Ensure this returns ChatPromptTemplate

class AnimeRecommender:
    def __init__(self, retriever, api_key: str, model_name: str):
        self.llm = ChatOpenAI(
            model=model_name, 
            api_key=api_key, 
            temperature=0
        )
        self.prompt = get_anime_prompt()  # Must be ChatPromptTemplate
        doc_chain = create_stuff_documents_chain(self.llm, self.prompt)
        self.qa_chain = create_retrieval_chain(retriever, doc_chain)
    
    def get_recommendation(self, query: str):
        result = self.qa_chain.invoke({"input": query})  # Key: "input"
        return result["answer"]  # Output key: "answer"