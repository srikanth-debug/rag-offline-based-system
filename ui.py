import streamlit as st
import requests

# Configuration
API_URL = "http://127.0.0.1:8000/ask"
st.set_page_config(page_title="AI History RAG", layout="wide")

# --- UI Elements ---
st.title("📖 AI History RAG System")
st.write("This is a demo of a Retrieval-Augmented Generation system. Ask a question about the history of AI based on the provided documents.")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Main Chat Logic ---
if prompt := st.chat_input("What was the Dartmouth workshop?"):
    # Add user message to history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get and display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        try:
            # Call the FastAPI backend
            payload = {"query": prompt}
            response = requests.post(API_URL, json=payload, timeout=600) # 2 minute timeout
            response.raise_for_status() # Raise an error for bad status codes
            
            data = response.json()
            answer = data.get("answer", "I couldn't find an answer.")
            citiations = data.get("citiations", [])
            #timings = data.get("timings", {})
            # We check if the answer is the specific refusal message.
            if "cannot answer" in answer:
            # Display a friendlier, more conversational message.
                message_placeholder.warning("It looks like that question is outside my knowledge base. I can only answer questions about the history of AI based on my documents.")
            else:  
            # Simulate stream of response with spaces
                message_placeholder.markdown(answer)

            #Display expandable citations
            if citiations :
                with st.expander("Show Sources"):
                    for citation in citations:
                        st.caption(citation) #Display each clean citation
            full_response = answer            
            #st.caption(f"Total time: {timings.get('total', 'N/A')}")

        except requests.exceptions.RequestException as e:
            message_placeholder.error(f"Failed to get a response from the backend. Error: {e}")
            answer = "Error communicating with the backend."

    # Add assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": answer})