import requests
import json
import io

# Server URL
BASE_URL = "http://localhost:8080"

def test_root_endpoint():
    """Test the root endpoint"""
    print("\n=== Testing Root Endpoint ===")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")

def test_predict_endpoint():
    """Test the /predict endpoint with text data"""
    print("\n=== Testing /predict Endpoint with Text Data ===")
    
    # Test with plain text
    query = "clustering algorithm"
    headers = {"content-type": "text/plain"}
    response = requests.post(
        f"{BASE_URL}/predict",
        data=query.encode("utf-8"),
        headers=headers
    )
    
    print(f"Status Code: {response.status_code}")
    try:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except:
        print(f"Response: {response.text}")

def test_predict_with_csv():
    """Test the /predict endpoint with CSV file upload"""
    print("\n=== Testing /predict Endpoint with CSV Upload ===")
    
    # Create a simple CSV with a query
    csv_content = "query\nrandom forest"
    print(f"CSV content: {csv_content}")
    
    # Try with different file upload methods
    
    # Method 1: Simple string in files dict (like your client)
    print("\nTesting Method 1: Simple string in files dict")
    response1 = requests.post(
        f"{BASE_URL}/predict",
        files={"X": csv_content}
    )
    print(f"Status Code: {response1.status_code}")
    print(f"Response: {response1.text}")
    
    # Method 2: File-like object with filename and mimetype
    print("\nTesting Method 2: File-like object with filename")
    from io import StringIO
    file_obj = StringIO(csv_content)
    response2 = requests.post(
        f"{BASE_URL}/predict",
        files={"X": ("query.csv", file_obj, "text/csv")}
    )
    print(f"Status Code: {response2.status_code}")
    print(f"Response: {response2.text}")
    
    # Method 3: Send as data with proper content-type
    print("\nTesting Method 3: Raw data with content-type")
    response3 = requests.post(
        f"{BASE_URL}/predict",
        data=csv_content,
        headers={"Content-Type": "text/csv"}
    )
    print(f"Status Code: {response3.status_code}")
    print(f"Response: {response3.text}")
    
    # Test if any method worked
    assert any(r.status_code == 200 for r in [response1, response2, response3]), \
        "All methods failed to get a 200 response"

def test_predict_unstructured():
    """Test the /predictUnstructured endpoint"""
    print("\n=== Testing /predictUnstructured Endpoint ===")
    
    # Test with plain text
    query = "support vector machine"
    headers = {"content-type": "text/plain"}
    response = requests.post(
        f"{BASE_URL}/predictUnstructured",
        data=query.encode("utf-8"),
        headers=headers
    )
    
    print(f"Status Code: {response.status_code}")
    try:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except:
        print(f"Response: {response.text}")
    
    # Test with JSON input
    print("\n--- Testing /predictUnstructured with JSON ---")
    data = ["decision tree"]
    headers = {"content-type": "application/json"}
    response = requests.post(
        f"{BASE_URL}/predictUnstructured",
        data=json.dumps(data),
        headers=headers
    )
    
    print(f"Status Code: {response.status_code}")
    try:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except:
        print(f"Response: {response.text}")

def test_chat_completions():
    """Test the /chat/completions endpoint"""
    print("\n=== Testing /chat/completions Endpoint ===")
    
    data = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me about linear regression"}
        ]
    }
    
    response = requests.post(
        f"{BASE_URL}/chat/completions",
        json=data
    )
    
    print(f"Status Code: {response.status_code}")
    try:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except:
        print(f"Response: {response.text}")

def test_tools_call():
    """Test the /tools/call endpoint"""
    print("\n=== Testing /tools/call Endpoint ===")
    
    data = {
        "id": 123,
        "method": "tools/call",
        "params": {
            "name": "scikit_search",
            "arguments": {
                "query": "gradient boosting"
            }
        }
    }
    
    response = requests.post(
        f"{BASE_URL}/tools/call",
        json=data
    )
    
    print(f"Status Code: {response.status_code}")
    try:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except:
        print(f"Response: {response.text}")

if __name__ == "__main__":
    print("Starting tests for Scikit-learn FAISS Retrieval Server...")
    
    # Run all tests
    test_root_endpoint()
    # test_predict_endpoint()
    test_predict_with_csv()
    # test_predict_unstructured()
    test_chat_completions()
    test_tools_call()
    
    print("\nAll tests completed!")