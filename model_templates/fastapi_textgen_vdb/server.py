import csv
import io
import json
import os
import time
import uuid

import chromadb
import numpy as np
import onnxruntime as ort
import uvicorn
from fastapi import Depends, FastAPI, File, Request, UploadFile
from models import TextContent, ToolCallRequest, ToolCallResponse, ToolCallResult
from transformers import AutoTokenizer

CHROMA_PATH = "./chroma_db"
CHROMA_COLLECTION = "sklearn"

MODEL_DIR = "onnx"
model_name = "prajjwal1/bert-tiny"

URL_PREFIX = os.getenv("URL_PREFIX", "")


class RetrieverWorker:
    def __init__(self, model_dir: str = "onnx", model_name: str = "prajjwal1/bert-tiny"):
        self.model_dir = model_dir
        
        # Load the ONNX model
        model_path = os.path.join(model_dir, "model.onnx")
        self.session = ort.InferenceSession(model_path)
        

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        self.input_names = [input.name for input in self.session.get_inputs()]
        self.output_names = [output.name for output in self.session.get_outputs()]

    def get_relevant_docs(self, query: str):
        encoded_input = self.tokenizer(
            [query], padding=True, truncation=True, max_length=256, return_tensors="np"
        )
        onnx_inputs = {}
        for key, value in encoded_input.items():
            if key in self.input_names:
                onnx_inputs[key] = value

        outputs = self.session.run(self.output_names, onnx_inputs)
        
        token_embeddings = outputs[0]
        
   
        attention_mask = encoded_input['attention_mask']
        input_mask_expanded = np.expand_dims(attention_mask, axis=-1)
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = np.sum(input_mask_expanded, axis=1)
        query_embedding = sum_embeddings / np.clip(sum_mask, a_min=1e-9, a_max=None)
        

        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection_name = CHROMA_COLLECTION
        
        collection = client.get_collection(collection_name)
        k = 3  # Return top 3 results
        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=k
        )
        
        formatted_results = []
        for i in range(len(results['ids'][0])):
            doc_id = results['ids'][0][i]
            distance = results['distances'][0][i]
            document = results['documents'][0][i]
            
            class_name = doc_id.split("_")[0]
            result = f"""{class_name} (Distance: {distance:.4f}) \n
            {document[:150]}"""
            formatted_results.append(result)
        
        return formatted_results

app = FastAPI()

@app.get(f"{URL_PREFIX}/")
async def root():
    return {"message": "Hello World"}


@app.post(f"{URL_PREFIX}/predict")
async def predict(request: Request, retriever: RetrieverWorker = Depends(RetrieverWorker)):
    content_type = request.headers.get("content-type", "")

    if "multipart/form-data" in content_type:
        # Handle multipart form-data with CSV file
        form = await request.form()
        file = form.get("file")
        if file and isinstance(file, UploadFile):
            content = await file.read()
            text = content.decode("utf-8")

    elif "text/plain" in content_type or "text/csv" in content_type:
        # Handle text/plain or text/csv input
        content = await request.body()
        text = content.decode("utf-8")
        if isinstance(text, str):
            text = [text]
    else:
        raise ValueError("Unsupported content type")

    if text:
        results_for_all_queries = []
        for query in text:
            results = retriever.get_relevant_docs(query)
            results_for_all_queries.append(results)
        return {"relevant": results_for_all_queries}
    else:
        raise ValueError("No text provided in the request")


@app.post(f"{URL_PREFIX}/predictUnstructured")
async def predict_unstructured(request: Request, retriever: RetrieverWorker = Depends(RetrieverWorker)):
    content_type = request.headers.get("content-type", "")

    if "multipart/form-data" in content_type:
        # Handle multipart form-data with CSV file
        form = await request.form()
        file = form.get("file")
        if file and isinstance(file, UploadFile):
            content = await file.read()
            text = content.decode("utf-8")

    elif "text/plain" in content_type or "text/csv" in content_type:
        # Handle text/plain or text/csv input
        content = await request.body()
        text = content.decode("utf-8")

    elif "application/json" in content_type:
        # Handle JSON input
        content = await request.body()
        try:
            data = json.loads(content)
            if isinstance(data, list):
                text = [
                    str(item) for item in data if item
                ]  # Convert all items to strings and filter empty items
            else:
                return ValueError("Invalid JSON format")
        except json.JSONDecodeError:
            return ValueError("Invalid JSON format")
    else:
        return ValueError("Unsupported content type")
    if len(text) != 1:
        raise ValueError("Only one text input is allowed")
    query = text[0]
    return retriever.get_relevant_docs(query)


@app.post(f"{URL_PREFIX}/chat/completions")
async def chat_completions(request: Request, retriever: RetrieverWorker = Depends(RetrieverWorker)):
    try:
        # Parse the request body
        body = await request.json()

        # Extract messages from the request
        messages = body.get("messages", [])

        # Find the latest user message
        latest_user_message = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                latest_user_message = message.get("content", "")
                break

        # Generate a response based on the user's message
        # In a real implementation, this would call a language model
        response_content = "Here are the relevant documents I found: \n" + "\n".join(
            retriever.get_relevant_docs(latest_user_message)
        )

        # Generate unique ID with "chatcmpl-" prefix
        completion_id = f"chatcmpl-{str(uuid.uuid4())[:8]}"

        # Get current timestamp
        timestamp = int(time.time())

        # Create response with the specified structure
        response = {
            "id": completion_id,
            "object": "chat.completion",
            "created": timestamp,
            "model": "datarobot-retriever",
            "system_fingerprint": f"fp_{str(uuid.uuid4())[:10]}",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response_content},
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
        }

        return response
    except Exception as e:
        return {"error": str(e)}


@app.post(f"{URL_PREFIX}/tools/call", response_model=ToolCallResponse)
async def tools_call(request: ToolCallRequest, retriever: RetrieverWorker = Depends(RetrieverWorker)):
    """
    Handle searches in an MCP tool call style. This
    route will be reachable at `directAccess/tools/call`.
    """
    try:
        request_id = request.id
        query = request.params.arguments.query

        text_response = "Here are the relevant documents I found: \n" + "\n".join(
            retriever.get_relevant_docs(query)
        )

        response = ToolCallResponse(
            id=request_id,
            result=ToolCallResult(content=[TextContnt(text=text_response)], isError=False),
        )

        return response
    except Exception as e:
        # Return error response
        return ToolCallResponse(
            id=request.id if hasattr(request, "id") else 0,
            result=ToolCallResult(content=[TextContent(text=f"Error: {str(e)}")], isError=True),
        )


def process_csv_content(text):
    # Parse CSV content
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)

    # Skip header row and extract strings
    if len(rows) > 0:
        data = rows[1:]  # Skip header row
        strings = [
            item for sublist in data for item in sublist if item
        ]  # Flatten and remove empty strings
        return strings

    return []




if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
