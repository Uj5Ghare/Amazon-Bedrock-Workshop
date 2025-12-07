import json
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import get_bedrock_client

bedrock = get_bedrock_client(service_name='bedrock-runtime') #creates a Bedrock client
                                                                                                                
bedrock_model_id = "us.amazon.nova-lite-v1:0" #set the foundation model

prompt = "What is the largest city in New Hampshire?" #the prompt to send to the model

messages = [
    {
        "role": "user",
        "content": [
            {"text": prompt}
        ]
    }
]

body = json.dumps({
    "schemaVersion": "messages-v1",
    "messages": messages,
    "inferenceConfig": {
        "maxTokens": 1024,
        "topP": 0.5,
        "topK": 20,
        "temperature": 0.0
    }
}) #build the request payload


response = bedrock.invoke_model(body=body, modelId=bedrock_model_id, accept='application/json', contentType='application/json') #send the payload to Amazon Bedrock

response_body = json.loads(response.get('body').read()) # read the response

response_text = response_body["output"]["message"]["content"][0]["text"] #extract the text from the JSON response

print(response_text)

