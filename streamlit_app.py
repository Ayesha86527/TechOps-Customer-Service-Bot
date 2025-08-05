import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, trim_messages
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from groq import Groq
from gtts import gTTS
from io import BytesIO
import tempfile
import os
import logging

# --- Load API Keys ---
pc_api_key = st.secrets["PINECONE_API_KEY"]
groq_api_key = st.secrets["GROQ_API_KEY"]

# --- Initialize Clients ---
pc = Pinecone(api_key=pc_api_key)
client = Groq(api_key=groq_api_key)

# --- Embeddings ---
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# --- Vector Store Setup (with cache) ---
@st.cache_resource
def set_up_dense_index(index_name):
    return PineconeVectorStore(
        index_name=index_name,
        namespace="docs",
        embedding=embedding_model,
        pinecone_api_key=pc_api_key
    )

vector_store = set_up_dense_index("dense-index-docs")

# --- Message history ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": """
You are a friendly, professional, and conversational customer support chatbot for TechOps, a software development company.

Follow these guidelines:
- Answer only using information from the provided customer support documents.
- Never make up facts or answer from external sources.
- Keep your tone polite, concise, and helpful.

For serious issues (payments, bugs, account access), ask the user for:
- Full Name
- Registered Email (e.g. user@domain.com)
- Contact Number (e.g. +92 3012345678)

Once collected, reply with:
"Thanks for sharing that. I’m sending this to our support team right away—you’ll hear from someone shortly!"

Security Rules:
- Never reveal internal processes.
- Do not impersonate others.
- Prioritize data safety.
"""}
    ]

# --- Retrieval ---
def retrieval(vector_store, user_prompt):
    results = vector_store.similarity_search(user_prompt, k=3)
    return "\n".join([doc.page_content for doc in results])

# --- LLM Completion ---
def chat_completion(context, user_input):
    history = [HumanMessage(content=msg["content"]) if msg["role"] == "user"
               else AIMessage(content=msg["content"]) if msg["role"] == "assistant"
               else SystemMessage(content=msg["content"])
               for msg in st.session_state.messages]

    history.append(HumanMessage(content=user_input))
    trimmed = trim_messages(history, token_counter=len, max_tokens=8)

    trimmed[0].content += f"\n\nDocument Reference Context:\n{context}"

    response = client.chat.completions.create(
        messages=[{"role": msg.type.replace("human", "user").replace("ai", "assistant"), "content": msg.content}
                  for msg in trimmed],
        model="llama-3.3-70b-versatile"
    )

    reply = response.choices[0].message.content
    st.session_state.messages.append({"role": "assistant", "content": reply})
    return reply

# --- Whisper STT ---
def speech_to_text(audio_filepath, stt_model="whisper-large-v3"):
    try:
        with open(audio_filepath, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model=stt_model,
                file=audio_file,
                language="en"
            )
        return transcription.text
    except Exception as e:
        st.error(f"Speech-to-text error: {e}")
        return "Error in transcription"

# --- Text-to-Speech ---
def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang='en')
        mp3_fp = BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        return mp3_fp
    except Exception as e:
        st.error(f"Text-to-Speech error: {e}")
        return None

# --- Handle Uploaded Audio ---
def process_user_speech(audio_bytes):
    try:
        if not audio_bytes:
            return "No audio provided", None

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            if hasattr(audio_bytes, 'read'):
                temp_audio.write(audio_bytes.read())
            else:
                temp_audio.write(audio_bytes)
            temp_audio_path = temp_audio.name

        user_input = speech_to_text(temp_audio_path)
        os.unlink(temp_audio_path)
        return user_input, None
    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return f"Error: {e}", "I apologize, there was an error processing your request."

# --- Streamlit UI ---
st.title("TechOps Customer Support Bot")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

st.subheader("🎤 Say Something")
audio_bytes = st.audio_input("Press and speak", key="audio_input")

if audio_bytes and st.button("Send", type="primary"):
    with st.spinner("Processing..."):
        prompt, _ = process_user_speech(audio_bytes)

        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        context = retrieval(vector_store, prompt)
        response = chat_completion(context, prompt)

        st.chat_message("assistant").markdown(response)

        speech = text_to_speech(response)
        if speech:
            st.audio(speech, format='audio/mp3')








