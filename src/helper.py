# importing libraries
import os
import json
import pandas as pd
import google.generativeai as genai
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from sklearn.feature_extraction.text import CountVectorizer
import warnings
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("Please set GEMINI_API_KEY in a .env file")

genai.configure(api_key=API_KEY)

#  Add error handling for model initialization
try:
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3,
        convert_system_message_to_human=True
    )
except Exception as e:
    raise Exception(f"Failed to initialize Gemini model: {e}")

#  Constants and configurations
VECTOR_DB_PATH = "../faiss_db"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_RESULTS = 5
CHUNK_SIZE = 1000

# embeddings initialization
def initialize_embeddings():
    try:
        return HuggingFaceEmbeddings(model_name=MODEL_NAME)
    except Exception as e:
        raise Exception(f"Failed to initialize embeddings: {e}")

embeddings = initialize_embeddings()

#  vector DB loading with error handling
def load_vector_db():
    """Load FAISS Vector Database with proper error handling."""
    try:
        print("Loading FAISS vector database...")
        if not os.path.exists(VECTOR_DB_PATH):
            raise FileNotFoundError(f"Vector database not found at {VECTOR_DB_PATH}")
        return FAISS.load_local(
            VECTOR_DB_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        raise Exception(f"Failed to load vector database: {e}")
    
# Initialize vector store with error handling
try:
    vector_store = load_vector_db()
except Exception as e:
    raise Exception(f"Failed to initialize vector store: {e}")

#  conversation memory setup
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)


#  QA chain setup
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=model,
    retriever=vector_store.as_retriever(search_kwargs={"k": MAX_RESULTS}),
    memory=memory,
    return_source_documents=True,
    verbose=True,
    chain_type="stuff"
)

# paper summarization with error handling
def summarize_paper(paper_title: str) -> str:
    """Generate a summary of a specific research paper with error handling."""
    try:
        results = vector_store.similarity_search(paper_title, k=1)
        if not results:
            return "Paper not found in the database."
        
        paper_content = results[0].page_content
        summary_prompt = f"""
        Please provide a detailed summary of the following research paper:
        {paper_content}

        Include:
        1. Main objectives
        2. Key findings
        3. Methodology
        4. Conclusions
        """
        response = model.invoke(summary_prompt)
        
        # Use .content instead of .text
        return response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        return f"Error generating summary: {str(e)}"

    
#  chat function with error handling
def chat_with_expert(query: str) -> dict:
    """Chatbot function with proper error handling."""
    try:
        if not query.strip():
            return {
                "answer": "Please provide a valid question.",
                "source_papers": []
            }
        
        result = qa_chain({"question": query})
        return {
            "answer": result["answer"],
            "source_papers": [
                doc.page_content.split('\n\n')[0] 
                for doc in result["source_documents"]
                if '\n\n' in doc.page_content
            ]
        }
    except Exception as e:
        return {
            "answer": f"Error processing question: {str(e)}",
            "source_papers": []
        }
        

# paper search function
def search_papers(query: str, n_results: int = MAX_RESULTS) -> list:
    """Search for relevant papers with error handling."""
    try:
        if not query.strip():
            return []

        papers = []
        fetched_titles = set()  # To track and avoid duplicate papers
        current_results = 0
        fetch_count = n_results  # Start by fetching the number requested
        iteration = 0  # Track the number of iterations to avoid infinite loops
        
        while current_results < n_results and iteration < 20:  # Increased iteration limit for thorough search
            # Fetch results
            results = vector_store.similarity_search(query, k=fetch_count + iteration)
            if not results:
                break  # Stop if no more results are found
            
            for doc in results:
                content = doc.page_content.strip()  # Remove leading/trailing spaces
                lines = content.split("\n")  # Split into lines instead of "\n\n"
                
                title, abstract = None, None
                
                # Extract title and abstract
                for line in lines:
                    if line.lower().startswith("title:"):
                        title = line.replace("Title:", "").strip()
                    elif line.lower().startswith("abstract:"):
                        abstract = line.replace("Abstract:", "").strip()
                        break  # Stop after finding the first abstract
                
                # Only add the paper if both title and abstract are present
                # Also, check for duplicate titles
                if title and abstract and title not in fetched_titles:
                    papers.append({"title": title, "abstract": abstract})
                    fetched_titles.add(title)
                    current_results += 1
                
                # Stop adding more papers once we reach the required number
                if current_results >= n_results:
                    break
            
            # If fewer results than requested, fetch more in the next iteration
            # Incrementally increase the fetch count to get more papers
            fetch_count = n_results - current_results + iteration
            iteration += 1
        
        # Final check: If still not enough papers, indicate that fewer were found
        if current_results < n_results:
            print(f"Only found {current_results} out of {n_results} requested papers.")
        
        return papers
    except Exception as e:
        return [{"title": f"Error searching papers: {str(e)}", "abstract": ""}]


    

#  concept visualization
def visualize_concepts(query: str) -> go.Figure:
    """Generate an improved concept visualization graph."""
    try:
        papers = search_papers(query, n_results=10)
        if not papers:
            raise ValueError("No papers found for visualization")

        vectorizer = CountVectorizer(
            max_features=10,
            stop_words='english',
            min_df=2
        )
        
        abstracts = [p['abstract'] for p in papers]
        doc_term_matrix = vectorizer.fit_transform(abstracts)
        terms = vectorizer.get_feature_names_out()

        # Create network graph
        G = nx.Graph()
        G.add_node(query, size=20, color='red')
        for term in terms:
            G.add_node(term, size=15, color='blue')
            G.add_edge(query, term)

        pos = nx.spring_layout(G)
        
        # Create Plotly figure
        fig = go.Figure()
        
        # Add edges
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        ))

        # Add nodes
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        node_color = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            node_size.append(G.nodes[node]['size'])
            node_color.append(G.nodes[node]['color'])
            
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition='top center',
            marker=dict(
                size=node_size,
                color=node_color,
                line_width=2
            )
        ))
        
        fig.update_layout(
            title=f"Concept Map for: {query}",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40)
        )
        
        return fig
    except Exception as e:
        # Return a basic figure with error message
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error generating visualization: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig