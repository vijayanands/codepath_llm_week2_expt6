from utils import download_data_and_create_embedding, openai_api_key, retrieve_and_generate
import pprint

vector_store = download_data_and_create_embedding()

# question we ask the chat model
query_str = "What are some good sci-fi movies from the 1980s?"

retrieved_nodes = retrieve_and_generate(query_str, vector_store)
    
def format_response(response):
    formatted_response = ""
    if isinstance(response, dict):
        formatted_response = json.dumps(response, indent=4)  # Pretty-print JSON
    elif hasattr(response, '__dict__'):
        formatted_response = json.dumps(response.__dict__, indent=4)  # Pretty-print object attributes
    else:
        formatted_response = str(response)
    return formatted_response

# Print the formatted response
# print(format_response(retrieved_nodes))
print("=== Result ====")
for node in retrieved_nodes:
    print(f"{node}")

