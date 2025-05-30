# FastAPI Text Generation with Vector Database

This is a template of a DataRobot Custom model using FastAPI. It implements several endpoints that are used by DataRobot to communicate with your model. It also implements one extra route `/tools/call` which can be reached at the DEPLOYMENT_URL/directAccess/tools/call.

## Overview

This template demonstrates how to build a FastAPI application that:
- Serves as a DataRobot custom model for text generation
- Uses a vector database (ChromaDB) for document retrieval
- Provides multiple API endpoints for different types of interaction
- Can be deployed automatically using Pulumi

## API Endpoints

### Required DataRobot Endpoints

- **`/`**: Simple root endpoint for health checks. This is technically the only endpoint that is "required"


### Additional DataRobot Endpoints

- **`/predict`**: Processes structured prediction requests with CSV data
- **`/predictUnstructured`**: Handles unstructured text input for retrieval
- **`/chat/completions`**: OpenAI-style chat interface

### API Endpoints 
- **`/tools/call`**: Custom endpoint for direct access (available at DEPLOYMENT_URL/directAccess/tools/call)

## Local Testing

The included `test.py` file allows you to test this model locally. It contains functions to test each endpoint with sample requests and displays the responses to verify functionality.

To run the tests:

1. Start the local server:
   ```bash
   python server.py
   ```

2. Run the test script:
   ```bash
   python test.py
   ```

The test script will send sample requests to each endpoint and display the responses, helping you verify that the model is functioning correctly before deployment. 

## Deployment

This template includes a Pulumi configuration (`infra.py`) which can automatically deploy the model to DataRobot. 

### Deployment Requirements

To use the Pulumi deployment, you will need to set two environment variables:
- **`DATAROBOT_API_TOKEN`**: Your DataRobot API token
- **`DATAROBOT_ENDPOINT`**: The DataRobot endpoint URL

### Deployment Process

The Pulumi script will:
1. Create a DataRobot custom model
2. Register the model
3. Create a deployment
4. Register and validate the vector database

