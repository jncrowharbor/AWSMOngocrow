import boto3
import pymongo
from utils import bedrock
from langchain_community.embeddings import BedrockEmbeddings
from utils import aws_utils

# Function to retrieve the MongoDB URI from AWS Secrets Manager
def get_mongo_uri(secret_name, region_name="us-east-1"):
    client = boto3.client("secretsmanager", region_name=region_name)
    try:
        # Retrieve the secret value, no JSON parsing needed as it's a plain string
        secret_value_response = client.get_secret_value(SecretId=secret_name)
        mongo_uri = secret_value_response['SecretString']
        return mongo_uri
    except Exception as e:
        print(f"Error retrieving MongoDB URI: {e}")
        return None

# Define the Bedrock client
boto3_bedrock = bedrock.get_bedrock_client()

# Get the MongoDB URI from Secrets Manager
mongo_uri = get_mongo_uri("workshop/atlas_secret")

# Ensure the MongoDB URI was successfully retrieved
if not mongo_uri:
    raise Exception("MongoDB URI not retrieved from AWS Secrets Manager")

# Connect to MongoDB
client = pymongo.MongoClient(mongo_uri)
db = client["sample_mflix"]
collection = db["movies"]

# Set the embedding model ID for Amazon Bedrock
embedding_model_id = "amazon.titan-embed-text-v1"

# Initiate the embedding object
embeddings = BedrockEmbeddings(model_id=embedding_model_id, client=boto3_bedrock)

# Define the vector field name and the document field to be vectorized
vector_field_name = "eg_vector"
field_name_to_be_vectorized = "fullplot"

# Find documents in the collection where the year is greater than 2014
documents = collection.find({"year": {"$gt": 2014}})

print("Started processing...")

# Process and vectorize documents
i = 0
for document in documents:
    i += 1
    query = {'_id': document['_id']}
    
    # Ensure the field exists and the vector field hasn't already been set
    if field_name_to_be_vectorized in document and vector_field_name not in document:
        # Generate embeddings for the text
        text_to_vectorize = document["title"] + " " + document[field_name_to_be_vectorized]
        text_as_embeddings = embeddings.embed_documents([text_to_vectorize])
        
        # Update the document in MongoDB with the new embedding vector
        update = {'$set': {vector_field_name: text_as_embeddings[0]}}
        collection.update_one(query, update)

    # Print progress every 5 documents
    if i % 5 == 0:
        print(f"Processed: {i} records")

    # Limit the processing to 200 documents
    if i > 200:
        break

print(f"Finished processing: {i} records")
