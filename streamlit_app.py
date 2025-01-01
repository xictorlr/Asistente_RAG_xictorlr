import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import PyPDF2
import sqlite3
import datetime
import torch
import torch.utils.data
from torch.utils.data import DataLoader
import warnings
from typing import Optional

warnings.filterwarnings('ignore')

class AsistenteRAG:
    def __init__(self):
        st.set_page_config(layout="wide", page_title="📚 RAG Pro", page_icon="📚")
        self.init_styles()
        self.init_session_state()
        self.init_db()
        
    def init_styles(self):
        st.markdown("""
        <style>
            .stApp { 
                background: linear-gradient(-45deg, #16222A, #3A6073, #1F1C2C, #2C5364);
                background-size: 400% 400%;
                animation: gradient 15s ease infinite;
                color: #ffffff;
            }
            /* Sidebar */
            .css-1d391kg {
                background: rgba(22, 34, 42, 0.85) !important;
                backdrop-filter: blur(12px);
            }
            .css-1d391kg .block-container {
                padding: 2rem 1rem;
            }
            /* Botones */
            .stButton > button {
                background: linear-gradient(45deg, #FFD700, #FFA500);
                color: #16222A;
                border: none;
                padding: 0.6rem 1.2rem;
                border-radius: 10px;
                font-weight: bold;
                transition: all 0.3s ease;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            .stButton > button:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(255, 215, 0, 0.4);
            }
            /* Inputs */
            .stTextInput > div > div {
                background: rgba(255,255,255,0.08);
                border-radius: 10px;
                border: 1px solid rgba(255,215,0,0.2);
                color: #fff;
                transition: all 0.3s ease;
            }
            .stTextInput > div > div:focus-within {
                border-color: #FFD700;
                box-shadow: 0 0 10px rgba(255,215,0,0.2);
            }
            /* Selectbox */
            .stSelectbox > div > div {
                background: rgba(255,255,255,0.08);
                border-radius: 10px;
                border: 1px solid rgba(255,215,0,0.2);
                color: #fff;
            }
            /* Chat container */
            .chat-container {
                background: rgba(22, 34, 42, 0.6);
                padding: 20px;
                border-radius: 15px;
                border: 1px solid rgba(255,215,0,0.1);
                max-height: 600px;
                overflow-y: auto;
            }
            .chat-question {
                background: rgba(255, 215, 0, 0.1);
                border-left: 4px solid #FFD700;
                margin: 10px 0;
                padding: 15px;
                border-radius: 10px;
                animation: slideIn 0.3s ease;
            }
            .chat-answer {
                background: rgba(255, 165, 0, 0.1);
                border-left: 4px solid #FFA500;
                margin: 10px 0;
                padding: 15px;
                border-radius: 10px;
                animation: slideIn 0.3s ease;
            }
            @keyframes slideIn {
                from { opacity: 0; transform: translateX(-10px); }
                to { opacity: 1; transform: translateX(0); }
            }
            .info-card {
                background: rgba(255,255,255,0.08);
                border-radius: 15px;
                padding: 20px;
                margin: 10px 0;
                border: 1px solid rgba(255,215,0,0.2);
                transition: all 0.3s ease;
            }
            .info-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 5px 15px rgba(255,215,0,0.2);
            }
        </style>
        """, unsafe_allow_html=True)

    def init_session_state(self):
        if 'initialized' not in st.session_state:
            st.session_state.update({
                'api_key_configured': False,
                'vectorstore': None,
                'chat_history': [],
                'session_id': datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                'device': 'cpu',
                'model_name': 'gpt-3.5-turbo',
                'temperature': 0.7,
                'chunk_size': 1000,
                'chunk_overlap': 200,
                'initialized': True
            })

    def init_db(self):
        conn = sqlite3.connect('rag_history.db', timeout=10)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS chat_sessions 
                    (session_id TEXT, timestamp TEXT, api_key TEXT, device TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS chat_history 
                    (session_id TEXT, question TEXT, answer TEXT, timestamp TEXT, 
                     model TEXT, temperature REAL)''')
        c.execute('''CREATE TABLE IF NOT EXISTS documents 
                    (session_id TEXT, filename TEXT, content TEXT, embedding_model TEXT)''')
        conn.commit()
        conn.close()

    def get_device(self) -> str:
        try:
            if torch.cuda.is_available():
                torch.cuda.init()
                return 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'
        except Exception as e:
            print(f"Error de detección de dispositivo: {e}")
        return 'cpu'

    def process_pdf(self, file: PyPDF2.PdfReader, device: str) -> FAISS:
        text = ""
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=st.session_state.chunk_size,
            chunk_overlap=st.session_state.chunk_overlap
        )
        chunks = splitter.split_text(text)
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': device}
        )
        return FAISS.from_texts(chunks, embeddings)

    def run(self):
        st.markdown("""
            <div style='text-align: center; padding: 30px; margin-bottom: 30px; 
                    background: rgba(22, 34, 42, 0.8); border-radius: 15px;'>
                <h1 style='color: #FFD700;'>Asistente RAG Profesional</h1>
                <p style='color: #ffffff; font-size: 1.2em;'>
                    Una potente herramienta de análisis de documentos que combina tecnologías avanzadas:
                </p>
                <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-top: 20px;'>
                    <div class='info-card'>
                        <h3 style='color: #FFD700;'>🤖 Modelos IA</h3>
                        <p>GPT-4 y GPT-3.5-Turbo para respuestas inteligentes</p>
                    </div>
                    <div class='info-card'>
                        <h3 style='color: #FFD700;'>🔍 Tecnología RAG</h3>
                        <p>Análisis avanzado con FAISS y embeddings de HuggingFace</p>
                    </div>
                    <div class='info-card'>
                        <h3 style='color: #FFD700;'>⚡ Optimización Hardware</h3>
                        <p>Soporte para CPU, CUDA y MPS</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with st.sidebar:
            st.markdown("""
                <div style='padding: 20px; background: rgba(22, 34, 42, 0.8); border-radius: 15px; margin-bottom: 20px;'>
                    <h2 style='color: #FFD700; text-align: center;'>⚙️ Configuración</h2>
                    <p style='color: #ffffff; text-align: center; font-size: 0.9em;'>
                        Configura los ajustes de tu asistente IA
                    </p>
                </div>
            """, unsafe_allow_html=True)
            
            api_key = st.text_input("🔑 Clave API OpenAI:", type="password")
            if api_key:
                st.session_state.api_key = api_key
                st.session_state.api_key_configured = True
                st.success("✅ Clave configurada correctamente")

            available_devices = ['cpu']
            if torch.cuda.is_available():
                available_devices.append('cuda')
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                available_devices.append('mps')
            
            st.session_state.device = st.selectbox(
                "🖥️ Dispositivo de procesamiento:",
                options=available_devices,
                index=available_devices.index(st.session_state.device)
            )

            st.session_state.model_name = st.selectbox(
                "🤖 Modelo:",
                ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"]
            )

            with st.expander("🔧 Configuración Avanzada"):
                st.session_state.temperature = st.slider(
                    "Temperatura:", 0.0, 1.0, 0.7, 0.1
                )
                st.session_state.chunk_size = st.number_input(
                    "Tamaño de fragmento:", 100, 2000, 1000, 100
                )
                st.session_state.chunk_overlap = st.number_input(
                    "Superposición:", 0, 500, 200, 50
                )

            if st.button("📝 Nueva Sesión"):
                st.session_state.update({
                    'session_id': datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                    'chat_history': []
                })

        file = st.file_uploader("📄 Sube tu archivo PDF", type="pdf")
        
        if file and st.session_state.api_key_configured:
            with st.spinner("🔍 Procesando PDF..."):
                try:
                    st.session_state.vectorstore = self.process_pdf(
                        file, st.session_state.device
                    )
                    st.success("✅ PDF procesado correctamente")
                except Exception as e:
                    st.error(f"❌ Error al procesar el PDF: {str(e)}")
                    return

        if st.session_state.vectorstore and st.session_state.api_key_configured:
            query = st.text_input("💬 ¿Qué deseas preguntar sobre el documento?")
            
            if query:
                with st.spinner("🧠 Generando respuesta..."):
                    try:
                        docs = st.session_state.vectorstore.similarity_search(query)
                        context = "\n".join(doc.page_content for doc in docs)
                        
                        prompt = ChatPromptTemplate.from_template("""
                            Usando el siguiente contexto, responde a la pregunta.
                            Si no encuentras la respuesta en el contexto, indícalo.
                            
                            Contexto: {context}
                            Pregunta: {question}
                            
                            Respuesta:
                        """)
                        
                        chain = prompt | ChatOpenAI(
                            temperature=st.session_state.temperature,
                            model=st.session_state.model_name,
                            openai_api_key=st.session_state.api_key
                        )
                        
                        response = chain.invoke({
                            "context": context,
                            "question": query
                        })
                        
                        st.session_state.chat_history.append({
                            "q": query,
                            "a": response.content
                        })
                        
                    except Exception as e:
                        st.error(f"❌ Error al generar respuesta: {str(e)}")
                        return

            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            for chat in st.session_state.chat_history:
                st.markdown(f"""
                    <div class="chat-question">👤 {chat['q']}</div>
                    <div class="chat-answer">🤖 {chat['a']}</div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        else:
            st.warning("⚠️ Por favor, configura la clave API y sube un archivo PDF.")

if __name__ == "__main__":
    app = AsistenteRAG()
    app.run()