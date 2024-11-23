import getpass
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
os.environ["TAVILY_API_KEY"] = ""
os.environ["GROQ_API_KEY"] = ""



local_llm = "llama3.1"
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

urls = [
    'https://en.wikipedia.org/wiki/Formula_One', 
    'https://en.wikipedia.org/wiki/Formula_One_racing',
    'https://simple.wikipedia.org/wiki/Formula_1',
    'https://en.wikipedia.org/wiki/List_of_Formula_One_Grands_Prix',
    'https://en.wikipedia.org/wiki/List_of_Formula_One_drivers',
    'https://en.wikipedia.org/wiki/List_of_Formula_One_constructors',
    'https://en.wikipedia.org/wiki/List_of_Formula_One_world_champions',
    'https://en.wikipedia.org/wiki/List_of_Formula_One_grand_prix_winners',
    'https://en.wikipedia.org/wiki/List_of_Formula_One_race_results',
    'https://en.wikipedia.org/wiki/List_of_Formula_One_race_tracks',
    'https://en.wikipedia.org/wiki/List_of_Formula_One_race_winners_by_race_number',
    'https://en.wikipedia.org/wiki/List_of_Formula_One_race_winners_by_season',
    'https://en.wikipedia.org/wiki/List_of_Formula_One_race_winners_by_driver',
    'https://en.wikipedia.org/wiki/List_of_Formula_One_race_winners_by_constructor',
    'https://en.wikipedia.org/wiki/List_of_Formula_One_race_winners_by_grand_prix',
    'https://en.wikipedia.org/wiki/List_of_Formula_One_race_winners_by_race_type',
    'https://en.wikipedia.org/wiki/List_of_Formula_One_race_winners_by_race_distance',
    'https://en.wikipedia.org/wiki/List_of_Formula_One_race_winners_by_race_surface',
    'https://en.wikipedia.org/wiki/List_of_Formula_One_race_winners_by_race_year',
    'https://en.wikipedia.org/wiki/List_of_Formula_One_race_winners_by_race_circuit',
    'https://en.wikipedia.org/wiki/List_of_Formula_One_race_winners_by_race_date',
    'https://en.wikipedia.org/wiki/List_of_Formula_One_race_winners_by_race_location',
    'https://en.wikipedia.org/wiki/List_of_Formula_One_race_winners_by_race_time',
    'https://en.wikipedia.org/wiki/List_of_Formula_One_race_winners_by_race_season_and_driver',
    'https://en.wikipedia.org/wiki/List_of_Formula_One_race_winners_by_race_season_and_constructor',
    'https://en.wikipedia.org/wiki/List_of_Formula_One_race_winners_by_race_season_and_grand_prix',
    'https://en.wikipedia.org/wiki/List_of_Formula_One_race_winners_by_race_season_and_race_type',
    'https://en.wikipedia.org/wiki/List_of_Formula_One_race_winners_by_race_season_and_race_distance',
    'https://en.wikipedia.org/wiki/List_of_Formula_One_race_winners_by_race_season_and_race_surface',
    
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

db2 = Chroma.from_documents(doc_splits, embedding=embedding, persist_directory="./chroma_db/Adaptive")
db3 = Chroma(persist_directory="./chroma_db/Adaptive", embedding_function=embedding)

retriever = db3.as_retriever()


llm = ChatOllama(model=local_llm, format="json", temperature=0)
prompt = PromptTemplate(
    template="""You are a teacher grading a quiz. You will be given: 
    1/ a QUESTION
    2/ A FACT provided by the student
    
    You are grading RELEVANCE RECALL:
    A score of 1 means that ANY of the statements in the FACT are relevant to the QUESTION. 
    A score of 0 means that NONE of the statements in the FACT are relevant to the QUESTION. 
    1 is the highest (best) score. 0 is the lowest score you can give. 
    
    Explain your reasoning in a step-by-step manner. Ensure your reasoning and conclusion are correct. 
    
    Avoid simply stating the correct answer at the outset.
    
    Question: {question} \n
    Fact: \n\n {documents} \n\n
    
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
    """,
    input_variables=["question", "documents"],
)

retrieval_grader = prompt | llm | JsonOutputParser()


from langchain_core.output_parsers import StrOutputParser

prompt = PromptTemplate(
    template="""You are an assistant for question-answering tasks. 
    
    Use the following documents to answer the question. 
    
    If you don't know the answer, just say that you don't know. 
    
    Use three sentences maximum and keep the answer concise:
    Question: {question} 
    Documents: {documents} 
    Answer: 
    """,
    input_variables=["question", "documents"],
)

llm = ChatOllama(model=local_llm, temperature=0)

rag_chain = prompt | llm | StrOutputParser()





from langchain_community.tools.tavily_search import TavilySearchResults

web_search_tool = TavilySearchResults(k=3)

from typing import List
from typing_extensions import TypedDict
from IPython.display import Image, display
from langchain.schema import Document
from langgraph.graph import START, END, StateGraph

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        search: whether to add search
        documents: list of documents
    """

    question: str
    generation: str
    search: str
    documents: List[str]
    steps: List[str]
    
def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    question = state["question"]
    documents = retriever.invoke(question)
    steps = state["steps"]
    steps.append("retrieve_documents")
    return {"documents": documents, "question": question, "steps": steps}

def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """

    question = state["question"]
    documents = state["documents"]
    generation = rag_chain.invoke({"documents": documents, "question": question})
    steps = state["steps"]
    steps.append("generate_answer")
    return {
        "documents": documents,
        "question": question,
        "generation": generation,
        "steps": steps,
    }
    
def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    question = state["question"]
    documents = state["documents"]
    steps = state["steps"]
    steps.append("grade_document_retrieval")
    filtered_docs = []
    search = "No"
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "documents": d.page_content}
        )
        grade = score["score"]
        if grade == "yes":
            filtered_docs.append(d)
        else:
            search = "Yes"
            continue
    return {
        "documents": filtered_docs,
        "question": question,
        "search": search,
        "steps": steps,
    }

def web_search(state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    question = state["question"]
    documents = state.get("documents", [])
    steps = state["steps"]
    steps.append("web_search")
    web_results = web_search_tool.invoke({"query": question})
    documents.extend(
        [
            Document(page_content=d["content"], metadata={"url": d["url"]})
            for d in web_results
        ]
    )
    return {"documents": documents, "question": question, "steps": steps}

def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """
    search = state["search"]
    if search == "Yes":
        return "search"
    else:
        return "generate"

workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)  
workflow.add_node("grade_documents", grade_documents) 
workflow.add_node("generate", generate)  
workflow.add_node("web_search", web_search) 

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "search": "web_search",
        "generate": "generate",
    },
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

custom_graph = workflow.compile(checkpointer=memory)
print("ready")