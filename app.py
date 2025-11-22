from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
# from secret_api_keys import hugging_face_api_key
import os
import streamlit as st
import time
from dotenv import load_dotenv

load_dotenv()

hugging_face_api_key = os.getenv("HF_TOKEN")

if not hugging_face_api_key:
    raise ValueError("❌ Missing HF_TOKEN. Please set it in your .env or repo secrets.")


MODEL_OPTIONS = [
    "HuggingFaceH4/zephyr-7b-beta",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "deepseek-ai/DeepSeek-V3.2-Exp",
    "meta-llama/Llama-3.1-8B-Instruct"
]

@st.cache_resource
def initialize_langchain_model(model_id):
    """
    Initialize LangChain model with proper error handling.
    Uses ChatHuggingFace for better chat-based interactions.
    """
    try:
        llm = HuggingFaceEndpoint(
            repo_id=model_id,
            task="text-generation",
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            repetition_penalty=1.1,
            return_full_text=False,
            huggingfacehub_api_token=hugging_face_api_key,
        )
        
        chat_model = ChatHuggingFace(llm=llm)
        
        return chat_model, "success"
    
    except Exception as e:
        return None, str(e)

def safe_invoke_chain(chain, inputs, max_retries=3):
    """
    Safely invoke a LangChain chain with retries and error handling.
    """
    for attempt in range(max_retries):
        try:
            response = chain.invoke(inputs)
            if hasattr(response, 'content'):
                return response.content.strip()
            elif isinstance(response, dict) and 'text' in response:
                return response['text'].strip()
            elif isinstance(response, str):
                return response.strip()
            else:
                return str(response).strip()
        
        except StopIteration:
            if attempt < max_retries - 1:
                st.warning(f"⚠️ StopIteration error (attempt {attempt + 1}/{max_retries}). Retrying...")
                time.sleep(2)
                continue
            else:
                return "Error: Model failed after multiple attempts. Please try a different model or simplify your request."
        
        except Exception as e:
            error_msg = str(e).lower()
            
            if "rate limit" in error_msg or "429" in error_msg:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5
                    st.warning(f"⚠️ Rate limit hit. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    return "Error: Rate limit exceeded. Please wait a minute and try again."
            
            elif "timeout" in error_msg:
                if attempt < max_retries - 1:
                    st.warning(f"⚠️ Timeout (attempt {attempt + 1}/{max_retries}). Retrying...")
                    time.sleep(3)
                    continue
                else:
                    return "Error: Request timed out. Try reducing the word count."
            
            elif "503" in error_msg or "loading" in error_msg:
                if attempt < max_retries - 1:
                    st.warning(f"⚠️ Model loading (attempt {attempt + 1}/{max_retries}). Please wait...")
                    time.sleep(5)
                    continue
                else:
                    return "Error: Model is currently loading. Please try again in a minute or select a different model."
            
            else:
                if attempt < max_retries - 1:
                    st.warning(f"⚠️ Error: {str(e)}. Retrying...")
                    time.sleep(2)
                    continue
                else:
                    return f"Error: {str(e)}"
    
    return "Error: All retry attempts failed."

def create_title_generation_chain(chat_model):
    """Create a LangChain chain for title generation."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a creative blog title generator. Generate exactly 10 numbered blog titles, one per line. Be concise and creative."),
        ("human", "Generate 10 creative blog post titles about: {topic}\n\nTarget audience: beginners and tech enthusiasts.\nFormat: numbered list 1-10, no explanations, just titles.")
    ])
    
    chain = prompt | chat_model
    return chain

def create_blog_generation_chain(chat_model):
    """Create a LangChain chain for blog content generation."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a professional blog writer. Write informative, engaging, and well-structured blog posts. Use clear language suitable for beginners."),
        ("human", """Write a comprehensive blog post with these specifications:

Title: {title}
Keywords to include: {keywords}
Target length: approximately {blog_length} words
Target audience: beginners

Structure your blog post with:
1. An engaging introduction
2. Well-organized main content with clear sections
3. A strong conclusion with key takeaways

Make it informative, easy to understand, and engaging.""")
    ])
    
    chain = prompt | chat_model
    return chain

def generate_titles_langchain(topic, chat_model):
    """Generate blog titles using LangChain."""
    chain = create_title_generation_chain(chat_model)
    return safe_invoke_chain(chain, {"topic": topic})

def generate_blog_content_langchain(title, keywords, blog_length, chat_model):
    """Generate blog content using LangChain."""
    chain = create_blog_generation_chain(chat_model)
    return safe_invoke_chain(chain, {
        "title": title,
        "keywords": keywords,
        "blog_length": blog_length
    })

# Streamlit UI
st.set_page_config(page_title="AI Blog Generator", page_icon="✍️", layout="wide")

st.title("✍️ AI Blog Content Assistant....")
st.header("Create High-Quality Blog Content Without Breaking the Bank")

if 'current_model' not in st.session_state:
    st.session_state['current_model'] = None
    st.session_state['current_model_name'] = MODEL_OPTIONS[0]
    st.session_state['model_status'] = "not_loaded"

# Sidebar for settings
with st.sidebar:
    st.header("Model")
    
    # Model selection
    selected_model = st.selectbox(
        "Choose AI Model:",
        MODEL_OPTIONS,
        index=MODEL_OPTIONS.index(st.session_state['current_model_name']) if st.session_state['current_model_name'] in MODEL_OPTIONS else 0,
        help="Different models have different capabilities. Zephyr and Mistral work best."
    )
    
    # Load model button
    if st.button("🔄 Load Model", type="primary", use_container_width=True):
        with st.spinner(f"Loading {selected_model}..."):
            model, status = initialize_langchain_model(selected_model)
            if status == "success":
                st.session_state['current_model'] = model
                st.session_state['current_model_name'] = selected_model
                st.session_state['model_status'] = "loaded"
                st.success(f"✅ Model loaded: {selected_model.split('/')[-1]}")
                st.rerun()
            else:
                st.session_state['model_status'] = "error"
                st.error(f"❌ Failed to load model: {status}")
    
    # Display model status
    st.divider()
    if st.session_state['model_status'] == "loaded":
        st.success(f"✅ Active Model: {st.session_state['current_model_name'].split('/')[-1]}")
    elif st.session_state['model_status'] == "error":
        st.error("❌ Model not loaded")
    else:
        st.info("ℹ️ Please load a model to start")
    
    st.header("🔧 Troubleshooting")
    st.write("""
    - If model fails, try a different one
    - Wait 30 seconds between requests
    - Reduce word count if timing out
    """)
    
    if st.button("🗑️ Clear All Cache", use_container_width=True):
        st.cache_resource.clear()
        if 'title_results' in st.session_state:
            del st.session_state['title_results']
        if 'blog_results' in st.session_state:
            del st.session_state['blog_results']
        st.success("Cache cleared!")
        st.rerun()

if st.session_state['model_status'] != "loaded":
    st.warning("⚠️ Please load a model from the sidebar to begin.")
    st.stop()

col1, col2 = st.columns([1, 1])

# Title Generation Section
with col1:
    st.subheader("🎯 Step 1: Generate Titles")
    with st.container(border=True):
        topic_name = st.text_input(
            "Enter your blog topic:",
            placeholder="e.g., Python Programming for Data Science",
            key="topic_input"
        )
        
        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            generate_titles_btn = st.button("✨ Generate Titles", type="primary", use_container_width=True)
        with col_btn2:
            if st.button("🗑️ Clear Titles", use_container_width=True):
                if 'title_results' in st.session_state:
                    del st.session_state['title_results']
                    st.rerun()
        
        if generate_titles_btn:
            if topic_name.strip():
                with st.spinner('🔄 generating titles... (30-60 seconds)'):
                    titles = generate_titles_langchain(topic_name.strip(), st.session_state['current_model'])
                    st.session_state['title_results'] = titles
                    st.rerun()
            else:
                st.warning("⚠️ Please enter a topic first.")
        
        # Display title results
        if 'title_results' in st.session_state:
            if st.session_state['title_results'].startswith("Error"):
                st.error(st.session_state['title_results'])
                st.info("💡 Try: Loading a different model or simplifying your topic")
            else:
                st.success("✅ Titles generated successfully!")
                st.text_area(
                    "Generated Titles:",
                    value=st.session_state['title_results'],
                    height=300,
                    key="titles_display"
                )

# Blog Generation Section
with col2:
    st.subheader("📝 Step 2: Generate Blog Post")
    with st.container(border=True):
        title_of_the_blog = st.text_input(
            "Blog Title:",
            placeholder="Paste or type your chosen title",
            key="title_input"
        )
        
        num_of_words = st.slider(
            "Target Word Count:",
            min_value=100,
            max_value=1000,
            step=50,
            value=400
        )
        
        # Keyword management
        st.write("**Keywords (SEO):**")
        
        if 'keywords' not in st.session_state:
            st.session_state['keywords'] = []
        
        col_kw1, col_kw2 = st.columns([3, 1])
        with col_kw1:
            keyword_input = st.text_input(
                "Add keyword:",
                placeholder="e.g., Python, Tutorial, Beginners",
                key="keyword_input",
                label_visibility="collapsed"
            )
        with col_kw2:
            if st.button("➕ Add", use_container_width=True):
                if keyword_input.strip() and keyword_input.strip() not in st.session_state['keywords']:
                    st.session_state['keywords'].append(keyword_input.strip())
                    st.rerun()
        
        # Display keywords
        if st.session_state['keywords']:
            keywords_html = " ".join([
                f'<span style="background-color:#ED3F27;padding:5px 10px;margin:3px;border-radius:15px;display:inline-block;">🏷️ {kw}</span>'
                for kw in st.session_state['keywords']
            ])
            st.markdown(keywords_html, unsafe_allow_html=True)
            
            if st.button("🗑️ Clear Keywords"):
                st.session_state['keywords'] = []
                st.rerun()
        
        st.write("")  # Spacing
        
        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            generate_blog_btn = st.button("🚀 Generate Blog", type="primary", use_container_width=True)
        with col_btn2:
            if st.button("🗑️ Clear Blog", use_container_width=True):
                if 'blog_results' in st.session_state:
                    del st.session_state['blog_results']
                    st.rerun()

if generate_blog_btn:
    if title_of_the_blog.strip():
        formatted_keywords = ', '.join(st.session_state['keywords']) if st.session_state['keywords'] else 'general topics'
        
        with st.spinner('🔄 Generating blog content... (1-2 minutes)'):
            blog_content = generate_blog_content_langchain(
                title_of_the_blog.strip(),
                formatted_keywords,
                num_of_words,
                st.session_state['current_model']
            )
            st.session_state['blog_results'] = {
                'title': title_of_the_blog.strip(),
                'content': blog_content,
                'keywords': formatted_keywords,
                'word_count': num_of_words
            }
            st.rerun()
    else:
        st.warning("⚠️ Please enter a blog title first.")

# Display blog results in full width
if 'blog_results' in st.session_state:
    st.divider()
    st.subheader("📄 Generated Blog Post")
    
    if st.session_state['blog_results']['content'].startswith("Error"):
        st.error(st.session_state['blog_results']['content'])
        st.info("💡 Try: Reducing word count, loading a different model, or simplifying the request")
    else:
        with st.container(border=True):
            st.markdown(f"### {st.session_state['blog_results']['title']}")
            st.caption(f"Keywords: {st.session_state['blog_results']['keywords']} | Target: {st.session_state['blog_results']['word_count']} words")
            st.markdown("---")
            st.write(st.session_state['blog_results']['content'])
        
        # Download options
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            download_txt = f"# {st.session_state['blog_results']['title']}\n\nKeywords: {st.session_state['blog_results']['keywords']}\n\n{st.session_state['blog_results']['content']}"
            st.download_button(
                label="📥 Download TXT",
                data=download_txt,
                file_name=f"{st.session_state['blog_results']['title'].replace(' ', '_')[:50]}.txt",
                mime="text/plain",
                use_container_width=True
            )
        with col2:
            download_md = f"# {st.session_state['blog_results']['title']}\n\n**Keywords:** {st.session_state['blog_results']['keywords']}\n\n---\n\n{st.session_state['blog_results']['content']}"
            st.download_button(
                label="📥 Download MD",
                data=download_md,
                file_name=f"{st.session_state['blog_results']['title'].replace(' ', '_')[:50]}.md",
                mime="text/markdown",
                use_container_width=True
            )
        
        # Word count
        actual_word_count = len(st.session_state['blog_results']['content'].split())
        st.info(f"📊 Actual Word Count: {actual_word_count} words")

# Footer
st.divider()
st.caption("🦜🔗 Powered by LangChain + Hugging Face | Built with Streamlit")