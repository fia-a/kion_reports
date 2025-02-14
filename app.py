import gradio as gr

def greet(name):
    return "Hello " + name + "!!"

demo = gr.Interface(fn=greet, inputs="text", outputs="text")

import json
import uuid
import os
from groq import Groq
import gradio as gr

from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from huggingface_hub import CommitScheduler
from pathlib import Path

# Create client
os.environ['GROQ_API_KEY'] = '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'
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Define the embedding model and the vectorstore
embedding_model = SentenceTransformerEmbeddings(model_name="thenlper/gte-large")

collection_name = 'reports_collection'

vectorstore = Chroma(
    collection_name=collection_name,
    persist_directory='./reports_db',
    embedding_function=embedding_model
)

# Prepare the logging functionality
log_file = Path("logs/") / f"data_{uuid.uuid4()}.json"
log_folder = log_file.parent

scheduler = CommitScheduler(
    repo_id="reports-qna",
    repo_type="dataset",
    folder_path=log_folder,
    path_in_repo="data",
    every=2
)

# Define the Q&A system message
qna_system_message = """
You are a trusted AI assistant for KION ITS APAC...
"""

qna_message_template = """
###Context
here are some documents and their page number that are relevant to the question mention below
{context}

###Question
{question}
"""

def predict(user_input):
    filter = "/content/drive/My Drive/ragkion/dataset/2024Q3.pdf"
    relevant_document_chunks = vectorstore.similarity_search(user_input, k=5, filter={"source": filter})
    context_list = [d.page_content + "\n ###Page: " + str(d.metadata['page']) + "\n\n " for d in relevant_document_chunks]
    context_for_query = "\n".join(context_list) + "this is all the context I have"

    prompt = [
        {"role": "system", "content": qna_system_message},
        {"role": "user", "content": qna_message_template.format(context=context_for_query, question=user_input)}
    ]

    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=prompt,
            temperature=0
        )
        prediction = response.choices[0].message.content
    except Exception as e:
        prediction = str(e)

    with scheduler.lock:
        with log_file.open("a") as f:
            f.write(json.dumps({
                "user_input": user_input,
                "retrieved_context": context_for_query,
                "model_response": prediction
            }))
            f.write("\n")

    return prediction

textbox = gr.Textbox(placeholder="Enter your question here", lines=6)

demo = gr.Interface(
    inputs=textbox, fn=predict, outputs="text",
    title="KION QNA",
    description="This web API presents an interface to ask questions on Kion Company",
    concurrency_limit=16
)

demo.queue()
demo.launch()
