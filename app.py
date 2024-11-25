import gradio as gr
import os
# api_token = os.getenv("HF_TOKEN")
api_token = "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"


from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFaceEndpoint
import torch
# import tiktoken

# list_llm = ["meta-llama/Llama-3.2-3B-Instruct", "meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Meta-Llama-3-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.2"]  
list_llm = ["meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-70B-Instruct", "meta-llama/Llama-3.1-405B-Instruct", "meta-llama/Llama-3.1-405B-Instruct-FP8"]  
list_llm_simple = [os.path.basename(llm) for llm in list_llm]

# Load and split PDF document
def load_doc(list_file_path):
    # Processing for one document only
    # loader = PyPDFLoader(file_path)
    # pages = loader.load()
    loaders = [PyPDFLoader(x) for x in list_file_path]
    pages = []
    for loader in loaders:
        pages.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1024, 
        chunk_overlap = 64 
    )  
    doc_splits = text_splitter.split_documents(pages)
    return doc_splits

# Create vector database
def create_db(splits):
    embeddings = HuggingFaceEmbeddings()
    vectordb = FAISS.from_documents(splits, embeddings)
    return vectordb

def create_or_load_db(splits, db_path="faiss_index"):
    model_name = "all-MPNet-base-v2"
    # model_name = "nlpaueb/legal-bert-base-uncased"
    # model_name = "text-embedding-ada-002"
    
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # Check if the FAISS index already exists
    if os.path.exists(db_path):
        vectordb = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
        print("Loaded existing vector database from disk.")
    else:
        vectordb = FAISS.from_documents(splits, embeddings)
        vectordb.save_local(db_path)
        print("Created and saved new vector database.")
    
    return vectordb


# Initialize langchain LLM chain
def initialize_llmchain(llm_model, temperature, max_tokens, top_k, vector_db, progress=gr.Progress()):
    if llm_model == "meta-llama/Meta-Llama-3-8B-Instruct":
        llm = HuggingFaceEndpoint(
            repo_id=llm_model,
            huggingfacehub_api_token = api_token,
            temperature = temperature,
            max_new_tokens = max_tokens,
            top_k = top_k,
        )
    else:
        llm = HuggingFaceEndpoint(
            huggingfacehub_api_token = api_token,
            repo_id=llm_model, 
            temperature = temperature,
            max_new_tokens = max_tokens,
            top_k = top_k,
        )
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key='answer',
        return_messages=True
    )

    retriever=vector_db.as_retriever()
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        chain_type="stuff", 
        memory=memory,
        return_source_documents=True,
        verbose=False,
    )
    return qa_chain

# Initialize database
def initialize_database(list_file_obj, is_offline=False, progress=gr.Progress()):

    # Create a list of documents (when valid)
    if is_offline:
        list_file_path = list_file_obj
    else:
        list_file_path = [x.name for x in list_file_obj if x is not None]

    # Load document and create splits
    doc_splits = load_doc(list_file_path)
    print("load_doc ")
    # Create or load vector database
    vector_db = create_or_load_db(doc_splits)
    print("vector_db ")
    return vector_db, "Document is ready to be queried!"

# Initialize LLM
def initialize_LLM(llm_option, llm_temperature, max_tokens, top_k, vector_db, progress=gr.Progress()):
    # print("llm_option",llm_option)
    llm_name = list_llm[llm_option]
    print("llm_name: ",llm_name)
    qa_chain = initialize_llmchain(llm_name, llm_temperature, max_tokens, top_k, vector_db, progress)
    return qa_chain, "Chatbot is ready!"


def format_chat_history(message, chat_history):
    formatted_chat_history = []
    for user_message, bot_message in chat_history:
        formatted_chat_history.append(f"User: {user_message}")
        formatted_chat_history.append(f"Assistant: {bot_message}")
    return formatted_chat_history
    

def conversation(qa_chain, message, history):
    formatted_chat_history = format_chat_history(message, history)
    # Generate response using QA chain
    response = qa_chain.invoke({"question": message, "chat_history": formatted_chat_history})
    response_answer = response["answer"]
    if response_answer.find("Helpful Answer:") != -1:
        response_answer = response_answer.split("Helpful Answer:")[-1]
    response_sources = response["source_documents"]
    response_source1 = response_sources[0].page_content.strip()
    response_source2 = response_sources[1].page_content.strip()
    response_source3 = response_sources[2].page_content.strip()
    # Langchain sources are zero-based
    response_source1_page = response_sources[0].metadata["page"] + 1
    response_source2_page = response_sources[1].metadata["page"] + 1
    response_source3_page = response_sources[2].metadata["page"] + 1
    # Append user message and response to chat history
    new_history = history + [(message, response_answer)]
    return qa_chain, gr.update(value=""), new_history, response_source1, response_source1_page, response_source2, response_source2_page, response_source3, response_source3_page
    

def upload_file(file_obj):
    list_file_path = []
    for idx, file in enumerate(file_obj):
        file_path = file_obj.name
        list_file_path.append(file_path)
    return list_file_path


def demo_gradio(default_prompt):

    # with gr.Blocks(theme=gr.themes.Default(primary_hue="sky")) as demo:
    with gr.Blocks(theme=gr.themes.Default(primary_hue="red", secondary_hue="pink", neutral_hue = "sky")) as demo:
        vector_db = gr.State()
        qa_chain = gr.State()
        gr.HTML("<center><h1>landuseAI by MyZoning</h1><center>")
        gr.Markdown("""<b>Query area zoning regulation!</b> This AI agent delivers the zoning information for any given property in that jurisdiction. \\
        <b>Context: PoC</b>
        """)
        with gr.Row():
            with gr.Column(scale = 86):
                gr.Markdown("<b>Step 1 - Upload PDF documents and Initialize RAG pipeline</b>")
                with gr.Row():
                    document = gr.Files(height=300, file_count="multiple", file_types=["pdf"], interactive=True, label="Upload PDF documents")
                    # document = gr.Files(height=300, file_count="multiple", file_types=["pdf"], interactive=True, label="Upload PDF documents")
                    # document = gr.FileExplorer()
                with gr.Row():
                    db_btn = gr.Button("Process Document")
                with gr.Row():
                        db_progress = gr.Textbox(value="Not initialized", show_label=False) # label="Vector database status", 
                gr.Markdown("<style>body { font-size: 16px; }</style><b>Select Large Language Model (LLM) and input parameters</b>")
                with gr.Row():
                    llm_btn = gr.Radio(list_llm_simple, label="Available LLMs", value = list_llm_simple[0], type="index") # info="Select LLM", show_label=False
                with gr.Row():
                    with gr.Accordion("LLM input parameters", open=False):
                        with gr.Row():
                            slider_temperature = gr.Slider(minimum = 0.01, maximum = 1.0, value=0.5, step=0.1, label="Temperature", info="Controls randomness in token generation", interactive=True)
                        with gr.Row():
                            slider_maxtokens = gr.Slider(minimum = 128, maximum = 9192, value=4096, step=128, label="Max New Tokens", info="Maximum number of tokens to be generated",interactive=True)
                        with gr.Row():
                                slider_topk = gr.Slider(minimum = 1, maximum = 10, value=3, step=1, label="top-k", info="Number of tokens to select the next token from", interactive=True)
                with gr.Row():
                    qachain_btn = gr.Button("Initialize Question Answering Chatbot")
                with gr.Row():
                        llm_progress = gr.Textbox(value="Not initialized", show_label=False) # label="Chatbot status", 

            with gr.Column(scale = 200):
                gr.Markdown("<b>Step 2 - Chat with your Document</b>")
                chatbot = gr.Chatbot(height=505)
                with gr.Accordion("Relevent context from the source document", open=False):
                    with gr.Row():
                        doc_source1 = gr.Textbox(label="Reference 1", lines=2, container=True, scale=20)
                        source1_page = gr.Number(label="Page", scale=1)
                    with gr.Row():
                        doc_source2 = gr.Textbox(label="Reference 2", lines=2, container=True, scale=20)
                        source2_page = gr.Number(label="Page", scale=1)
                    with gr.Row():
                        doc_source3 = gr.Textbox(label="Reference 3", lines=2, container=True, scale=20)
                        source3_page = gr.Number(label="Page", scale=1)
                with gr.Row():
                    msg = gr.Textbox(value=default_prompt , placeholder="Ask a question", container=True)
                with gr.Row():
                    submit_btn = gr.Button("Submit")
                    clear_btn = gr.ClearButton([msg, chatbot], value="Clear")
            
        # Preprocessing events
        db_btn.click(initialize_database, \
            inputs=[document], \
            outputs=[vector_db, db_progress])
        qachain_btn.click(initialize_LLM, \
            inputs=[llm_btn, slider_temperature, slider_maxtokens, slider_topk, vector_db], \
            outputs=[qa_chain, llm_progress]).then(lambda:[None,"",0,"",0,"",0], \
            inputs=None, \
            outputs=[chatbot, doc_source1, source1_page, doc_source2, source2_page, doc_source3, source3_page], \
            queue=False)

        # Chatbot events
        msg.submit(conversation, \
            inputs=[qa_chain, msg, chatbot], \
            outputs=[qa_chain, msg, chatbot, doc_source1, source1_page, doc_source2, source2_page, doc_source3, source3_page], \
            queue=False)
        submit_btn.click(conversation, \
            inputs=[qa_chain, msg, chatbot], \
            outputs=[qa_chain, msg, chatbot, doc_source1, source1_page, doc_source2, source2_page, doc_source3, source3_page], \
            queue=False)
        clear_btn.click(lambda:[None,"",0,"",0,"",0], \
            inputs=None, \
            outputs=[chatbot, doc_source1, source1_page, doc_source2, source2_page, doc_source3, source3_page], \
            queue=False)
    demo.queue().launch(debug=True)


def demo_offline(default_prompt):

    document = ['examples/los_angeles-ca-1.pdf']
    vector_db, _ = initialize_database(document, True)
    qa_chain, _ = initialize_LLM(2, 0.5, 4096, 3, vector_db)

    response = conversation(qa_chain, default_prompt, [])

    print(response)


if __name__ == "__main__":

    default_prompt = '''Exclusively based on the information given in the PDF and upcoming information: The Zone is R1-1, the lot size is 3,397.6 sq ft, the lot depth is 85 ft, the lot width is 40 ft, the existing building’s footprint is 1151.6 sq ft, the existing building square footage is 936 sq ft.

    Create a table that summarizes the below information, and include a reference of the article number in the document for each result. 
    If a result has more than one option - please detail the options. 
    You can make calculation when needed, in this case state that and include the equation used for that.
    
    Table: Property Development Details

    The Rows:
    Zoning
    Land Use
    Floor Area Ratio (FAR)
    Maximum Lot Coverage
    Setbacks (front, side, rear)
    Maximum Number of Stories
    Maximum Height
    Maximum Density
    Maximum Number of Buildings
    Minimum Area Per Lot
    Minimum Area per Dwelling Unit
    Minimum Lot Width
    Number of parking required
    Maximum Buildable Area
    Available Square Footage for Addition
    Available Square Footage for Addition on Ground Floor'''

    demo_gradio(default_prompt)