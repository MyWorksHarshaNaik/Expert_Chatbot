import streamlit as st
from helper import (
    chat_with_expert,
    search_papers,
    summarize_paper,
    visualize_concepts
)

def initialize_session_state():
    """Initialize session state variables."""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def main():
    st.set_page_config(
        page_title="AI Research Chatbot",
        page_icon="📚",
        layout="wide"
    )
    
    initialize_session_state()
    
    st.title("📚 AI Computer Science Research Chatbot")
    
    # Sidebar
    st.sidebar.title("🔍 Features")
    feature = st.sidebar.radio(
        "Select a feature",
        ["Chat with Expert", "Search Papers", "Summarize Paper", "Concept Visualization"]
    )
    
    # Main content area
    if feature == "Chat with Expert":
        st.header("💬 Chat with AI Expert")
        
        # Display chat history
        for msg in st.session_state.chat_history:
            st.write(f"**You:** {msg['question']}")
            st.write(f"**AI:** {msg['answer']}")
            if msg['sources']:
                st.write("**Sources:**")
                for source in msg['sources']:
                    st.write(f"- {source}")
            st.write("---")
        
        # Input area
        query = st.text_area("Ask a question:")
        if st.button("Get Answer", key="chat"):
            if query:
                with st.spinner("Processing your question..."):
                    response = chat_with_expert(query)
                    st.session_state.chat_history.append({
                        'question': query,
                        'answer': response["answer"],
                        'sources': response["source_papers"]
                    })
                    # Update the display without using experimental_rerun
                    st.write(f"**You:** {query}")
                    st.write(f"**AI:** {response['answer']}")
                    if response['source_papers']:
                        st.write("**Sources:**")
                        for source in response['source_papers']:
                            st.write(f"- {source}")
                    st.write("---")
            else:
                st.warning("Please enter a question.")
    
    elif feature == "Search Papers":
        st.header("🔎 Search Research Papers")
        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input("Enter a topic:")
        with col2:
            n_results = st.number_input("Number of results:", 1, 10, 5)
        
        if st.button("Search", key="search"):
            if query:
                with st.spinner("Searching papers..."):
                    papers = search_papers(query, n_results)
                    if papers:
                        for i, paper in enumerate(papers, 1):
                            with st.expander(f"{i}. {paper['title']}"):
                                st.write(paper["abstract"])
                    else:
                        st.info("No papers found for your query.")
            else:
                st.warning("Please enter a topic.")
    
    elif feature == "Summarize Paper":
        st.header("📄 Summarize a Research Paper")
        paper_title = st.text_input("Enter the paper title:")
        if st.button("Summarize", key="summarize"):
            if paper_title:
                with st.spinner("Generating summary..."):
                    summary = summarize_paper(paper_title)
                    st.write("### Summary:")
                    st.write(summary)
            else:
                st.warning("Please enter a paper title.")
    
    elif feature == "Concept Visualization":
        st.header("📊 Concept Visualization")
        query = st.text_input("Enter a concept:")
        if st.button("Visualize", key="visualize"):
            if query:
                with st.spinner("Generating visualization..."):
                    fig = visualize_concepts(query)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please enter a concept.")

    # Add a clear chat history button for the chat feature
    if feature == "Chat with Expert" and st.session_state.chat_history:
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

if __name__ == "__main__":
    main()