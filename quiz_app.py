import hashlib
import io
import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

# # Core dependencies
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
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SESSIONS_DIR = "sessions"  # Directory to store session files

# Session Management Classes
class SessionManager:
    """Handles session creation, saving, loading, and management"""
    
    def __init__(self):
        self.sessions_dir = SESSIONS_DIR
        self.ensure_sessions_directory()
    
    def ensure_sessions_directory(self):
        """Create sessions directory if it doesn't exist"""
        if not os.path.exists(self.sessions_dir):
            os.makedirs(self.sessions_dir)
    
    def create_new_session(self, session_name: str = None) -> str:
        """Create a new session with unique ID"""
        session_id = str(uuid.uuid4())
        session_data = {
            "session_id": session_id,
            "session_name": session_name or f"Session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "created_at": datetime.now().isoformat(),
            "last_modified": datetime.now().isoformat(),
            "pdf_name": None,
            "pdf_processed": False,
            "chat_history": [],
            "quiz_data": None,
            "pdf_text": "",
            "vectorstore_data": None,
            "model_config": {
                "llm_type": "groq",
                "model_name": "llama-3.1-70b-versatile"
            }
        }
        
        self.save_session(session_data)
        return session_id
    
    def save_session(self, session_data: dict):
        """Save session data to file"""
        try:
            session_id = session_data["session_id"]
            session_data["last_modified"] = datetime.now().isoformat()
            
            filepath = os.path.join(self.sessions_dir, f"{session_id}.json")
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            st.error(f"Error saving session: {str(e)}")
    
    def load_session(self, session_id: str) -> Optional[dict]:
        """Load session data from file"""
        try:
            filepath = os.path.join(self.sessions_dir, f"{session_id}.json")
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return None
        except Exception as e:
            st.error(f"Error loading session: {str(e)}")
            return None
    
    def list_sessions(self) -> List[dict]:
        """Get list of all available sessions"""
        sessions = []
        try:
            for filename in os.listdir(self.sessions_dir):
                if filename.endswith('.json'):
                    session_id = filename[:-5]  # Remove .json extension
                    session_data = self.load_session(session_id)
                    if session_data:
                        sessions.append({
                            "session_id": session_id,
                            "session_name": session_data.get("session_name", "Unknown"),
                            "created_at": session_data.get("created_at", ""),
                            "last_modified": session_data.get("last_modified", ""),
                            "pdf_name": session_data.get("pdf_name", "No PDF"),
                            "chat_count": len(session_data.get("chat_history", []))
                        })
            
            # Sort by last modified (newest first)
            sessions.sort(key=lambda x: x["last_modified"], reverse=True)
            return sessions
        except Exception as e:
            st.error(f"Error listing sessions: {str(e)}")
            return []
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        try:
            filepath = os.path.join(self.sessions_dir, f"{session_id}.json")
            if os.path.exists(filepath):
                os.remove(filepath)
                return True
            return False
        except Exception as e:
            st.error(f"Error deleting session: {str(e)}")
            return False
    
    def export_session(self, session_id: str) -> Optional[str]:
        """Export session as JSON string"""
        session_data = self.load_session(session_id)
        if session_data:
            return json.dumps(session_data, indent=2, ensure_ascii=False)
        return None
    
    def import_session(self, session_json: str) -> Optional[str]:
        """Import session from JSON string"""
        try:
            session_data = json.loads(session_json)
            # Generate new session ID to avoid conflicts
            old_id = session_data.get("session_id", "unknown")
            new_id = str(uuid.uuid4())
            session_data["session_id"] = new_id
            session_data["session_name"] = f"{session_data.get('session_name', 'Imported')} (Copy)"
            session_data["created_at"] = datetime.now().isoformat()
            
            self.save_session(session_data)
            return new_id
        except Exception as e:
            st.error(f"Error importing session: {str(e)}")
            return None

# Initialize session state with session management
def initialize_session_state():
    """Initialize all session state variables"""
    if 'session_manager' not in st.session_state:
        st.session_state.session_manager = SessionManager()
    
    if 'current_session_id' not in st.session_state:
        # Create a default session
        st.session_state.current_session_id = st.session_state.session_manager.create_new_session("Default Session")
    
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
    if 'current_session_name' not in st.session_state:
        st.session_state.current_session_name = "Default Session"

def load_session_to_state(session_id: str):
    """Load session data into Streamlit session state"""
    session_data = st.session_state.session_manager.load_session(session_id)
    if session_data:
        st.session_state.current_session_id = session_id
        st.session_state.current_session_name = session_data.get("session_name", "Unknown")
        st.session_state.pdf_processed = session_data.get("pdf_processed", False)
        st.session_state.pdf_text = session_data.get("pdf_text", "")
        st.session_state.chat_history = session_data.get("chat_history", [])
        st.session_state.quiz_data = session_data.get("quiz_data", None)
        
        # Note: vectorstore needs to be recreated as it can't be serialized
        if st.session_state.pdf_processed and st.session_state.pdf_text:
            # Recreate vectorstore from saved text
            vectorstore = VectorStore()
            pdf_name = session_data.get("pdf_name", "unknown.pdf")
            if vectorstore.create_vectorstore(st.session_state.pdf_text, pdf_name):
                st.session_state.vectorstore = vectorstore
                st.session_state.embeddings = vectorstore.embeddings

def save_current_session():
    """Save current session state to file"""
    session_data = {
        "session_id": st.session_state.current_session_id,
        "session_name": st.session_state.current_session_name,
        "created_at": datetime.now().isoformat(),  # Will be overwritten if session exists
        "last_modified": datetime.now().isoformat(),
        "pdf_name": getattr(st.session_state, 'current_pdf_name', None),
        "pdf_processed": st.session_state.pdf_processed,
        "chat_history": st.session_state.chat_history,
        "quiz_data": st.session_state.quiz_data,
        "pdf_text": st.session_state.pdf_text,
        "vectorstore_data": None,  # Can't serialize vectorstore
        "model_config": getattr(st.session_state, 'model_config', {
            "llm_type": "groq",
            "model_name": "llama-3.1-70b-versatile"
        })
    }
    
    # Preserve original creation date if session already exists
    existing_session = st.session_state.session_manager.load_session(st.session_state.current_session_id)
    if existing_session:
        session_data["created_at"] = existing_session.get("created_at", session_data["created_at"])
    
    st.session_state.session_manager.save_session(session_data)

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

    def __init__(self, api_key: Optional[str] = None, model: str = None):
        # If no key passed, load from environment
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("❌ Groq API key not found. Please set GROQ_API_KEY in your .env file.")
        
        self.client = Groq(api_key=self.api_key)
        # Updated to use the model you want - checking if it's available in Groq
        self.model = model or "llama-3.1-70b-versatile"  # This is a powerful Groq model
        # Note: "openai/gpt-oss-120b" is not available in Groq's model list
        
        # Available Groq models (as of early 2024):
        # - "llama3-8b-8192"
        # - "llama3-70b-8192" 
        # - "llama-3.1-70b-versatile"
        # - "mixtral-8x7b-32768"
        # - "gemma-7b-it"

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

# Alternative: If you want to use HuggingFace models directly
class HuggingFaceLLM:
    """Alternative LLM using HuggingFace transformers"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-large"):
        try:
            from transformers import pipeline
            self.model_name = model_name
            # For text generation, you might want to use:
            # - "microsoft/DialoGPT-large"
            # - "facebook/blenderbot-400M-distill" 
            # - "google/flan-t5-large"
            self.pipeline = pipeline("text-generation", model=model_name)
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers torch")
    
    def generate_response(self, prompt: str, max_tokens: int = 1000) -> str:
        """Generate response using HuggingFace model"""
        try:
            # This is a basic implementation - you might need to adjust based on the specific model
            response = self.pipeline(prompt, max_length=max_tokens, num_return_sequences=1)
            return response[0]['generated_text']
        except Exception as e:
            return f"⚠️ Error generating response: {str(e)}"

class LangGraphWorkflow:
    """LangGraph workflow for processing queries"""
    
    def __init__(self, llm, vectorstore: VectorStore):
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

def render_session_sidebar():
    """Render session management in sidebar"""
    st.sidebar.header("📁 Session Management")
    
    # Current session info
    st.sidebar.info(f"**Current Session:** {st.session_state.current_session_name}")
    
    # Session actions
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("💾 Save Session", help="Save current session"):
            save_current_session()
            st.success("Session saved!")
    
    with col2:
        if st.button("🆕 New Session", help="Create new session"):
            new_session_id = st.session_state.session_manager.create_new_session()
            load_session_to_state(new_session_id)
            st.rerun()
    
    # Session list
    st.sidebar.subheader("Available Sessions")
    sessions = st.session_state.session_manager.list_sessions()
    
    if sessions:
        for session in sessions:
            with st.sidebar.expander(f"📝 {session['session_name'][:20]}..." if len(session['session_name']) > 20 else f"📝 {session['session_name']}"):
                st.write(f"**Created:** {session['created_at'][:16]}")
                st.write(f"**PDF:** {session['pdf_name']}")
                st.write(f"**Messages:** {session['chat_count']}")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("📂", key=f"load_{session['session_id']}", help="Load session"):
                        load_session_to_state(session['session_id'])
                        st.rerun()
                
                with col2:
                    if st.button("📤", key=f"export_{session['session_id']}", help="Export session"):
                        exported = st.session_state.session_manager.export_session(session['session_id'])
                        if exported:
                            st.download_button(
                                "💾 Download",
                                exported,
                                f"{session['session_name']}.json",
                                "application/json",
                                key=f"download_{session['session_id']}"
                            )
                
                with col3:
                    if st.button("🗑️", key=f"delete_{session['session_id']}", help="Delete session"):
                        if st.session_state.session_manager.delete_session(session['session_id']):
                            st.success("Session deleted!")
                            st.rerun()
    else:
        st.sidebar.write("No sessions found")
    
    # Import session
    st.sidebar.subheader("📥 Import Session")
    uploaded_session = st.sidebar.file_uploader(
        "Upload session file",
        type="json",
        key="session_upload"
    )
    
    if uploaded_session:
        try:
            session_content = uploaded_session.read().decode('utf-8')
            new_session_id = st.session_state.session_manager.import_session(session_content)
            if new_session_id:
                st.sidebar.success("Session imported successfully!")
                load_session_to_state(new_session_id)
                st.rerun()
        except Exception as e:
            st.sidebar.error(f"Error importing session: {str(e)}")
    
    # Rename current session
    st.sidebar.subheader("✏️ Rename Session")
    new_name = st.sidebar.text_input("New session name:", value=st.session_state.current_session_name)
    if st.sidebar.button("Rename"):
        st.session_state.current_session_name = new_name
        save_current_session()
        st.success("Session renamed!")
        st.rerun()

def main():
    """Main Streamlit application"""
    initialize_session_state()
    
    st.title("📚 PDF Q&A & Quiz Generator with Sessions")
    st.markdown("Upload PDF documents, chat with them, and manage multiple conversation sessions!")
    
    # Auto-save session on interactions
    if 'last_interaction' not in st.session_state:
        st.session_state.last_interaction = datetime.now()
    
    # Sidebar configuration
    with st.sidebar:
        # Session Management
        render_session_sidebar()
        
        st.divider()
        
        st.header("⚙️ Configuration")
        
        # Model selection
        st.subheader("🤖 Model Selection")
        use_hf_model = st.checkbox("Use HuggingFace Model", value=False)
        
        if use_hf_model:
            hf_model = st.selectbox(
                "HuggingFace Model",
                [
                    "microsoft/DialoGPT-large",
                    "google/flan-t5-large",
                    "facebook/blenderbot-400M-distill"
                ]
            )
            st.info("⚠️ HuggingFace models will download on first use and may be slow.")
            st.session_state.model_config = {"llm_type": "huggingface", "model_name": hf_model}
        else:
            groq_model = st.selectbox(
                "Groq Model",
                [
                    "llama-3.1-70b-versatile",
                    "llama3-70b-8192", 
                    "llama3-8b-8192",
                    "mixtral-8x7b-32768",
                    "gemma-7b-it"
                ]
            )
            st.session_state.model_config = {"llm_type": "groq", "model_name": groq_model}
        
        # Check API key for Groq
        if not use_hf_model and not os.getenv("GROQ_API_KEY"):
            st.error("❌ GROQ_API_KEY not found in environment variables!")
            st.info("Please add your Groq API key to the .env file:")
            st.code("GROQ_API_KEY=your_api_key_here")
            st.stop()
        elif not use_hf_model:
            st.success(f"✅ Using Groq LLM: {groq_model}")
        else:
            st.success(f"✅ Using HuggingFace Model: {hf_model}")
        
        st.divider()
        
        # PDF Upload
        st.header("📄 Upload PDF")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload the PDF document you want to analyze"
        )
        
        if uploaded_file:
            st.session_state.current_pdf_name = uploaded_file.name
            
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
                            
                            # Save session with new PDF
                            save_current_session()
                            
                            st.success("PDF processed successfully!")
                        else:
                            st.error("Failed to process PDF.")
                    else:
                        st.error("No text could be extracted from the PDF.")
    
    # Main content area
    if not st.session_state.pdf_processed:
        st.info("👆 Please upload and process a PDF file to get started.")
        return
    
    # Initialize LLM based on selection
    try:
        if use_hf_model:
            llm = HuggingFaceLLM(hf_model)
        else:
            llm = GroqLLM(model=groq_model)
        
        workflow = LangGraphWorkflow(llm, st.session_state.vectorstore)
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        return
    
    # Tabs for different functionalities
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Q&A Chat", "🧠 Quiz Generator", "📊 Document Stats", "📋 Session Analytics"])
    
    with tab1:
        st.header("Ask Questions About Your PDF")
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if "timestamp" in message:
                    st.caption(f"🕒 {message['timestamp']}")
        
        # Chat input
        if question := st.chat_input("Ask a question about your PDF..."):
            # Add user message to history with timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.chat_history.append({
                "role": "user", 
                "content": question,
                "timestamp": timestamp
            })
            
            with st.chat_message("user"):
                st.write(question)
                st.caption(f"🕒 {timestamp}")
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = workflow.process_query(question, mode="qa")
                    st.write(response)
                    
                    response_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.caption(f"🕒 {response_timestamp}")
                    
                    # Add assistant response to history with timestamp
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": response,
                        "timestamp": response_timestamp
                    })
                    
                    # Auto-save session after each interaction
                    save_current_session()
        
        # Chat management buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                save_current_session()
                st.rerun()
        
        with col2:
            if st.button("💾 Save Chat"):
                save_current_session()
                st.success("Chat saved to session!")
        
        with col3:
            if len(st.session_state.chat_history) > 0:
                # Export chat as text
                chat_text = "\n".join([
                    f"[{msg.get('timestamp', 'N/A')}] {msg['role'].upper()}: {msg['content']}"
                    for msg in st.session_state.chat_history
                ])
                st.download_button(
                    "📤 Export Chat",
                    chat_text,
                    f"chat_{st.session_state.current_session_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    "text/plain"
                )
    
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
                
                # Save quiz to session
                save_current_session()
        
        # Display quiz if available
        if st.session_state.quiz_data:
            display_quiz(st.session_state.quiz_data)
            
            # Export quiz button
            if st.button("📤 Export Quiz"):
                quiz_filename = f"quiz_{st.session_state.current_session_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                st.download_button(
                    "💾 Download Quiz",
                    st.session_state.quiz_data,
                    quiz_filename,
                    "application/json"
                )
    
    with tab3:
        st.header("Document Statistics")
        
        if st.session_state.pdf_text:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Characters", len(st.session_state.pdf_text))
            
            with col2:
                word_count = len(st.session_state.pdf_text.split())
                st.metric("Total Words", word_count)
            
            with col3:
                # Estimate reading time (average 200 words per minute)
                reading_time = max(1, word_count // 200)
                st.metric("Est. Reading Time", f"{reading_time} min")
            
            with col4:
                # Count unique words
                unique_words = len(set(st.session_state.pdf_text.lower().split()))
                st.metric("Unique Words", unique_words)
            
            # Text analysis
            st.subheader("📊 Text Analysis")
            
            # Most common words
            from collections import Counter
            words = [word.lower().strip('.,!?";') for word in st.session_state.pdf_text.split() if len(word) > 3]
            word_freq = Counter(words)
            
            if word_freq:
                st.write("**Most Common Words:**")
                common_words = word_freq.most_common(10)
                
                # Create a simple bar chart representation
                for word, count in common_words:
                    bar_length = int((count / common_words[0][1]) * 20)
                    bar = "█" * bar_length
                    st.write(f"{word}: {bar} ({count})")
            
            st.subheader("Document Preview")
            st.text_area(
                "First 1000 characters:",
                st.session_state.pdf_text[:1000] + "..." if len(st.session_state.pdf_text) > 1000 else st.session_state.pdf_text,
                height=200,
                disabled=True
            )
    
    with tab4:
        st.header("Session Analytics")
        
        # Current session info
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Current Session")
            st.metric("Messages Exchanged", len(st.session_state.chat_history))
            st.metric("Session Name", st.session_state.current_session_name)
            if hasattr(st.session_state, 'current_pdf_name'):
                st.metric("Current PDF", st.session_state.current_pdf_name)
        
        with col2:
            st.subheader("⏱️ Session Timeline")
            if st.session_state.chat_history:
                first_message = st.session_state.chat_history[0].get('timestamp', 'Unknown')
                last_message = st.session_state.chat_history[-1].get('timestamp', 'Unknown')
                st.write(f"**Started:** {first_message}")
                st.write(f"**Last Activity:** {last_message}")
                
                # Calculate session duration if timestamps are available
                try:
                    if first_message != 'Unknown' and last_message != 'Unknown':
                        start_time = datetime.strptime(first_message, "%Y-%m-%d %H:%M:%S")
                        end_time = datetime.strptime(last_message, "%Y-%m-%d %H:%M:%S")
                        duration = end_time - start_time
                        st.write(f"**Duration:** {str(duration).split('.')[0]}")
                except:
                    pass
        
        # All sessions overview
        st.subheader("📈 All Sessions Overview")
        sessions = st.session_state.session_manager.list_sessions()
        
        if sessions:
            total_sessions = len(sessions)
            total_messages = sum(session['chat_count'] for session in sessions)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Sessions", total_sessions)
            with col2:
                st.metric("Total Messages", total_messages)
            with col3:
                avg_messages = total_messages / total_sessions if total_sessions > 0 else 0
                st.metric("Avg Messages/Session", f"{avg_messages:.1f}")
            
            # Sessions table
            st.subheader("📋 Sessions Summary")
            
            session_data = []
            for session in sessions:
                session_data.append({
                    "Session Name": session['session_name'][:30] + "..." if len(session['session_name']) > 30 else session['session_name'],
                    "PDF": session['pdf_name'][:20] + "..." if session['pdf_name'] and len(session['pdf_name']) > 20 else (session['pdf_name'] or "None"),
                    "Messages": session['chat_count'],
                    "Created": session['created_at'][:16] if session['created_at'] else "Unknown",
                    "Last Modified": session['last_modified'][:16] if session['last_modified'] else "Unknown"
                })
            
            # Display as a table
            import pandas as pd
            df = pd.DataFrame(session_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No sessions found.")
        
        # Session management actions
        st.subheader("🔧 Session Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🧹 Clean Empty Sessions", help="Remove sessions with no messages"):
                deleted_count = 0
                for session in sessions:
                    if session['chat_count'] == 0 and session['session_id'] != st.session_state.current_session_id:
                        if st.session_state.session_manager.delete_session(session['session_id']):
                            deleted_count += 1
                
                if deleted_count > 0:
                    st.success(f"Deleted {deleted_count} empty sessions!")
                    st.rerun()
                else:
                    st.info("No empty sessions to clean.")
        
        with col2:
            if st.button("📊 Export All Sessions", help="Export all sessions as a single file"):
                all_sessions_data = {
                    "export_timestamp": datetime.now().isoformat(),
                    "total_sessions": len(sessions),
                    "sessions": []
                }
                
                for session in sessions:
                    full_session = st.session_state.session_manager.load_session(session['session_id'])
                    if full_session:
                        all_sessions_data["sessions"].append(full_session)
                
                export_json = json.dumps(all_sessions_data, indent=2, ensure_ascii=False)
                st.download_button(
                    "💾 Download All Sessions",
                    export_json,
                    f"all_sessions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json"
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
