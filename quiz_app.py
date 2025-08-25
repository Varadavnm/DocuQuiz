import hashlib
import io
import json
import operator
import os
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional

# Core dependencies
import chromadb
import PyPDF2
import streamlit as st
import tiktoken
from dotenv import load_dotenv
from groq import Groq
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
# LangGraph imports
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

load_dotenv()

#Groq API Key"
api_key=os.getenv("GROQ_API_KEY")

# Configuration
st.set_page_config(
    page_title="PDF Q&A & Quiz Generator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_TOKENS_PER_REQUEST = 6000
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables"""
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None
    if 'pdf_text' not in st.session_state:
        st.session_state.pdf_text = ""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'quiz_data' not in st.session_state:
        st.session_state.quiz_data = None
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = None

class AppState(TypedDict):
    """State definition for LangGraph workflow"""
    query: str
    context: List[str]
    response: str
    mode: str  # 'qa' or 'quiz'
    quiz_type: str
    num_questions: int
    difficulty: str

class PDFProcessor:
    """Handles PDF processing and text extraction"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_file) -> str:
        """Extract text content from uploaded PDF"""
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return ""

class VectorStore:
    """Manages document vectorization and retrieval"""
    
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.client = chromadb.Client()
        self.collection = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def create_vectorstore(self, text: str, pdf_name: str):
        """Create vector store from PDF text"""
        try:
            # Split text into chunks
            chunks = self.text_splitter.split_text(text)
            
            # Create unique collection name
            collection_name = f"pdf_{hashlib.md5(pdf_name.encode()).hexdigest()[:8]}"
            
            # Delete existing collection if it exists
            try:
                self.client.delete_collection(collection_name)
            except:
                pass
            
            # Create new collection
            self.collection = self.client.create_collection(collection_name)
            
            # Add documents to collection
            for i, chunk in enumerate(chunks):
                embedding = self.embeddings.embed_query(chunk)
                self.collection.add(
                    embeddings=[embedding],
                    documents=[chunk],
                    ids=[f"chunk_{i}"]
                )
            
            return True
            
        except Exception as e:
            st.error(f"Error creating vector store: {str(e)}")
            return False
    
    def similarity_search(self, query: str, k: int = 5) -> List[str]:
        """Perform similarity search and return relevant chunks"""
        try:
            if not self.collection:
                return []
            
            query_embedding = self.embeddings.embed_query(query)
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k
            )
            
            return results['documents'][0] if results['documents'] else []
            
        except Exception as e:
            st.error(f"Error in similarity search: {str(e)}")
            return []
class GroqLLM:
    """Handles Groq API interactions"""

    def __init__(self, api_key: Optional[str] = None):
        # If no key passed, load from environment
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("❌ Groq API key not found. Please set GROQ_API_KEY in your .env file.")
        
        self.client = Groq(api_key=self.api_key)
        self.model = "llama3-8b-8192"  # Fast and efficient model

    def generate_response(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate response using Groq API"""
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant. Provide accurate answers based only on the given context. If insufficient, say so clearly."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                max_tokens=max_tokens,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"⚠️ Error generating response: {str(e)}"


class LangGraphWorkflow:
    """LangGraph workflow for processing queries"""
    
    def __init__(self, llm: GroqLLM, vectorstore: VectorStore):
        self.llm = llm
        self.vectorstore = vectorstore
        self.graph = self._create_workflow()
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow"""
        
        def retrieve_context(state: AppState) -> AppState:
            """Retrieve relevant context from vector store"""
            context = self.vectorstore.similarity_search(state["query"], k=5)
            state["context"] = context
            return state
        
        def generate_qa_response(state: AppState) -> AppState:
            """Generate Q&A response"""
            context_text = "\n\n".join(state["context"])
            
            prompt = f"""Based on the following context from a PDF document, please answer the user's question accurately and comprehensively.

Context:
{context_text}

Question: {state["query"]}

Please provide a detailed answer based only on the information provided in the context. If the context doesn't contain enough information to answer the question, please state that clearly.

Answer:"""
            
            response = self.llm.generate_response(prompt, max_tokens=800)
            state["response"] = response
            return state
        
        def generate_quiz_response(state: AppState) -> AppState:
            """Generate quiz questions"""
            context_text = "\n\n".join(state["context"])
            
            quiz_prompt = f"""Based on the following content from a PDF document, create {state["num_questions"]} {state["quiz_type"]} questions at {state["difficulty"]} difficulty level.

Content:
{context_text}

Requirements:
- Questions should test understanding of key concepts
- For multiple choice: provide 4 options (A, B, C, D) with only one correct answer
- For short answer: provide the expected answer
- Include explanations for correct answers
- Format as JSON with this structure:
{{
    "questions": [
        {{
            "question": "Question text",
            "type": "{state["quiz_type"]}",
            "options": ["A. Option 1", "B. Option 2", "C. Option 3", "D. Option 4"] (for multiple choice only),
            "correct_answer": "Correct answer",
            "explanation": "Why this is correct"
        }}
    ]
}}

Generate the quiz:"""
            
            response = self.llm.generate_response(quiz_prompt, max_tokens=1500)
            state["response"] = response
            return state
        
        def route_query(state: AppState) -> str:
            """Route to appropriate processing node"""
            if state["mode"] == "quiz":
                return "generate_quiz"
            else:
                return "generate_qa"
        
        # Create workflow
        workflow = StateGraph(AppState)
        
        # Add nodes
        workflow.add_node("retrieve_context", retrieve_context)
        workflow.add_node("generate_qa", generate_qa_response)
        workflow.add_node("generate_quiz", generate_quiz_response)
        
        # Add edges
        workflow.set_entry_point("retrieve_context")
        workflow.add_conditional_edges(
            "retrieve_context",
            route_query,
            {
                "generate_qa": "generate_qa",
                "generate_quiz": "generate_quiz"
            }
        )
        workflow.add_edge("generate_qa", END)
        workflow.add_edge("generate_quiz", END)
        
        return workflow.compile()
    
    def process_query(self, query: str, mode: str = "qa", **kwargs) -> str:
        """Process query through the workflow"""
        initial_state = AppState(
            query=query,
            context=[],
            response="",
            mode=mode,
            quiz_type=kwargs.get("quiz_type", "multiple_choice"),
            num_questions=kwargs.get("num_questions", 5),
            difficulty=kwargs.get("difficulty", "medium")
        )
        
        result = self.graph.invoke(initial_state)
        return result["response"]

def display_quiz(quiz_json: str):
    """Display quiz questions with interactive elements"""
    try:
        # Clean the response to extract JSON - handle markdown code blocks
        json_str = quiz_json.strip()
        
        # Remove markdown code block markers if present
        if json_str.startswith('```json'):
            json_str = json_str[7:]  # Remove ```json
        elif json_str.startswith('```'):
            json_str = json_str[3:]   # Remove ```
        
        if json_str.endswith('```'):
            json_str = json_str[:-3]  # Remove closing ```
        
        # Find JSON object within the text
        start_idx = json_str.find('{')
        end_idx = json_str.rfind('}') + 1
        
        if start_idx != -1 and end_idx > start_idx:
            json_str = json_str[start_idx:end_idx]
        
        quiz_data = json.loads(json_str)
        
        st.subheader("📝 Generated Quiz")
        
        user_answers = {}
        
        for i, question in enumerate(quiz_data.get("questions", [])):
            st.write(f"**Question {i+1}:** {question['question']}")
            
            if question["type"] == "multiple_choice":
                options = question.get("options", [])
                user_answer = st.radio(
                    f"Select your answer for Question {i+1}:",
                    options,
                    key=f"q_{i}"
                )
                user_answers[i] = user_answer
            else:
                user_answer = st.text_input(
                    f"Your answer for Question {i+1}:",
                    key=f"q_{i}"
                )
                user_answers[i] = user_answer
            
            # Show answer button
            if st.button(f"Show Answer for Question {i+1}", key=f"show_{i}"):
                st.success(f"**Correct Answer:** {question['correct_answer']}")
                st.info(f"**Explanation:** {question['explanation']}")
            
            st.divider()
            
    except json.JSONDecodeError:
        st.error("Error parsing quiz data. The generated response may not be in the correct format.")
        st.text("Raw response:")
        st.code(quiz_json)
    except Exception as e:
        st.error(f"Error displaying quiz: {str(e)}")
        st.text("Raw response:")
        st.code(quiz_json)

def main():
    """Main Streamlit application"""
    initialize_session_state()
    
    st.title("📚 PDF Q&A & Quiz Generator")
    st.markdown("Upload a PDF document and interact with it through questions or generate quizzes!")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Check if Groq API key is available
        if not os.getenv("GROQ_API_KEY"):
            st.error("❌ GROQ_API_KEY not found in environment variables!")
            st.info("Please add your Groq API key to the .env file:")
            st.code("GROQ_API_KEY=your_api_key_here")
            st.stop()
        else:
            st.success("✅ using Groq LLM")
        
        st.divider()
        
        # PDF Upload
        st.header("📄 Upload PDF")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload the PDF document you want to analyze"
        )
        
        if uploaded_file:
            if st.button("Process PDF", type="primary"):
                with st.spinner("Processing PDF..."):
                    # Extract text
                    pdf_text = PDFProcessor.extract_text_from_pdf(uploaded_file)
                    
                    if pdf_text:
                        # Create vector store
                        vectorstore = VectorStore()
                        if vectorstore.create_vectorstore(pdf_text, uploaded_file.name):
                            st.session_state.vectorstore = vectorstore
                            st.session_state.pdf_text = pdf_text
                            st.session_state.pdf_processed = True
                            st.session_state.embeddings = vectorstore.embeddings
                            st.success("PDF processed successfully!")
                        else:
                            st.error("Failed to process PDF.")
                    else:
                        st.error("No text could be extracted from the PDF.")
    
    # Main content area
    if not st.session_state.pdf_processed:
        st.info("👆 Please upload and process a PDF file to get started.")
        return
    
    # Initialize LLM and workflow
    llm = GroqLLM()
    workflow = LangGraphWorkflow(llm, st.session_state.vectorstore)
    
    # Tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["💬 Q&A Chat", "🧠 Quiz Generator", "📊 Document Stats"])
    
    with tab1:
        st.header("Ask Questions About Your PDF")
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Chat input
        if question := st.chat_input("Ask a question about your PDF..."):
            # Add user message to history
            st.session_state.chat_history.append({"role": "user", "content": question})
            
            with st.chat_message("user"):
                st.write(question)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = workflow.process_query(question, mode="qa")
                    st.write(response)
                    
                    # Add assistant response to history
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Clear chat history button
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    with tab2:
        st.header("Generate Quiz Questions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            quiz_type = st.selectbox(
                "Quiz Type",
                ["multiple_choice", "short_answer"],
                format_func=lambda x: "Multiple Choice" if x == "multiple_choice" else "Short Answer"
            )
        
        with col2:
            num_questions = st.slider("Number of Questions", 1, 10, 5)
        
        with col3:
            difficulty = st.selectbox(
                "Difficulty Level",
                ["easy", "medium", "hard"]
            )
        
        if st.button("Generate Quiz", type="primary"):
            with st.spinner("Generating quiz questions..."):
                quiz_response = workflow.process_query(
                    "Generate quiz questions based on this document",
                    mode="quiz",
                    quiz_type=quiz_type,
                    num_questions=num_questions,
                    difficulty=difficulty
                )
                
                st.session_state.quiz_data = quiz_response
        
        # Display quiz if available
        if st.session_state.quiz_data:
            display_quiz(st.session_state.quiz_data)
    
    with tab3:
        st.header("Document Statistics")
        
        if st.session_state.pdf_text:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Characters", len(st.session_state.pdf_text))
            
            with col2:
                word_count = len(st.session_state.pdf_text.split())
                st.metric("Total Words", word_count)
            
            with col3:
                # Estimate reading time (average 200 words per minute)
                reading_time = max(1, word_count // 200)
                st.metric("Est. Reading Time", f"{reading_time} min")
            
            st.subheader("Document Preview")
            st.text_area(
                "First 1000 characters:",
                st.session_state.pdf_text[:1000] + "..." if len(st.session_state.pdf_text) > 1000 else st.session_state.pdf_text,
                height=200,
                disabled=True
            )

if __name__ == "__main__":
    st.markdown(
    """
    <div style="padding:10px; background-color:#fff3cd; border:1px solid #ffeeba; border-radius:8px; margin-top:10px;">
        ⚠️ <b>Disclaimer:</b> This application uses AI language models to generate answers and quizzes.  
        While the system is designed to provide accurate information based on your PDF, the responses may occasionally be incomplete, incorrect, or misleading.  
        Please verify critical information independently before relying on it.
    </div>
    """,
    unsafe_allow_html=True
)

    main()