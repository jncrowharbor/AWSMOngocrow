# Building Generative AI application MongoDB Atlas, Bedrock, Langchain, and Streamlit

## Reference Architecture
![Reference Architecture](images/Reference_Architecture.png)

## Creating a Secret in Secrets Manager
* You can skip this section and create the secret via CLI in the next section
* In AWS Console navigate to Secret Manager and click on Store a New secret
> ![Secret Manager](images/console_secret_manager.png)

* Choose Other type of secret then Plaintext tab and provide the connection string of the MongoDB Atlas Cluster. We have created the connection string in the previous lab.
> ![Other Type](images/other_type.png)

* Proceed to the next page where you supply secret name as `workshop/atlas_secret`
> ![Secret Name](images/secret_name.png)  

* Continue to other pages without any changes to store the secret
> ![Store Secret](images/store_secret.png)

* Confirm the secret is created by observing it listed on the page.  Note: you might need hit refresh button to reload the secrets.
> ![Confirm Secret](images/list_secret.png)



* Run the cloudshell

![CloudShell](images/cloudshell.png)


* If you skipped secret creation in the previous section, run the following command to create the secret.  Make sure to provide your own user ID/password/cluster.
```
aws secretsmanager create-secret --name workshop/atlas_secret  --secret-string 'mongodb+srv://<your_user>:<your_password>@<your_cluster_dns>/?retryWrites=true&w=majority'
```
* Clone the repo
```
git clone https://github.com/mongodb-partners/AWS_MongoDB_Generative_AI.git
cd AWS_MongoDB_Generative_AI/bedrock_atlas_vector_search_streamlit/
```

## Load MongoDB Atlas Sample Dataset
* In MongoDB Atlas Console, from your Cluster menu select Load Sample Dataset
>![Load Sample](images/load_sample.png)
## Vector Search Index Creation
> * Navigate Data Services|Your Cluster|Brows Collections|Atlas Search
> ![Create Index Button](images/create_index_button.png)


* Select `movies` collection from `sample_mflix` database and copy and paste the JSON snippet below.

```
  {
  "fields": [
    {
      "numDimensions": 1536,
      "path": "eg_vector",
      "similarity": "cosine",
      "type": "vector"
    }
  ]
}
```

* Select Next
> ![Search Index Confirm](images/search_index_confirm.png)

* Confirm Index Creation by clicking on Create Search Index
> ![Confirm Create](images/index_creat_confirm.png)

* Observe index being created
> ![Indexing Progress ](images/index_progress.png)

* Once the index is in status Active, it is ready for use.
> ![Index Active](images/index_active.png)


# Enable Model Access
* In this step we need to enable access to several models
* In AWS Console, navigate to Amazon Bedrock|Model Access. 
> ![Model Access](images/model_access.png)
* Click on Manage Model Access and select all models under Amazon. Click on Request model to proceed.
> ![Amazon Models](images/amazon_models.png)
* Observe Access status to be Access Granted
> ![Access Granted](images/access_granted.png)




## Create Embeddings



* Run the following code in terminal and wait for the program to finish:

```
python create_embeddings.py
```
> ![Create Embeddings](images/run_create_embeddings1.png)\

* Now observe the vector containing embeddings created in MongoDB Atlas.  Note: because we are not updating the full dataset you might need to filter the records by supplying this filter expression: `{"eg_vector":{"$exists": true}}`
> ![Vector in Atlas](images/vector_in_atlas.png)

## Run Search to verify
* In terminal run the following command to perform vector search
```
 python query_atlas.py
```
* Verify that program returns search results:
> ![Search Results](images/search_results1.png)


* Next, we run the program that adds a generative feature.  
```
 python llm_atlas.py
```
* Based on the retrieved description, we are now generating a description for a new movie. 
>![New Description](images/new_description1.png)



## Run Streamlit app
* Before we can run the streamlit app, we need install ECS Copilot CLI by running the following commands

```
 sudo curl -Lo /usr/local/bin/copilot https://github.com/aws/copilot-cli/releases/latest/download/copilot-linux    && sudo chmod +x /usr/local/bin/copilot    && copilot --help
```
![ECS Copilot](images/copilot.png)

* Install the Application and Environment using ECS copilot

```
copilot init
```

* Select the Load Balanced Web Services
![alt text](images/selectLoadbalancewebservices.png)


* Name the service 
![alt text](images/giveservicename.png)

* Enter the custom path for Dockerfile 
![alt text](images/dockerfilepath.png)

* Give the path for the Dockerfile
![alt text](images/dockerfile.png)

* Give yes to "Would you like to deploy an environment?"
![alt text](images/environmentselection.png)

* Give a name to the environment
![alt text](images/environmentname.png)



* Run Streamlit app to create chatbot using the command below
```
streamlit run app.py
```
> ![Streamlit App](images/streamlit_app.png)

* Open the external URL in a separate browser tab. You should see the app loaded
> ![app loaded](images/app_loaded.png)
* Type several keywords to generate a new movie description. You can add or remove keywords to generate new descriptions.  
> ![Movie Description](images/movie_description.png)

# Summary
* Congratulations!  You have deployed an app that performs semantic search in MongoDB Atlas and mashes up several descriptions into one!