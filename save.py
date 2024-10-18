import numpy as np
import pandas as pd
import logging
from embedding import openai_embedding_model,sparse_vector, late_interaction
from qdrant import Qdrant
import traceback

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Save:

    def __init__(self) -> None:
        self.qdrant = Qdrant()

    def get_contents_embed(self, df,type):
        try:
            if type == "dense":
                # Generate dense embeddings
                embeddings = openai_embedding_model(df['content'].tolist())
                embed = np.array(embeddings)
                embedding = embed.reshape(-1, 1536)    
                df['dense'] = embedding.tolist()
                return df

            # Generate dense embeddings
            embeddings = openai_embedding_model(df['content'].tolist())
            embed = np.array(embeddings)
            embedding = embed.reshape(-1, 1536)
            print(embedding)
            sparse = sparse_vector(df['content'].tolist())
            late = late_interaction(df['content'].tolist())
            
            df['dense'] = embedding.tolist()

            df['sparse'] = sparse
            df['latent'] = late
            return df

        except Exception as e:
            logger.error(f"Error generating dense embeddings: {e}")           
            traceback.print_exc()

    def save_to_collection(self, collection_name, df,type="hybrid"):
        try:
            df = self.get_contents_embed(df,type)
            if df is not None:
                self.qdrant.upsert_data(collection_name, df,type)
            else:
                logger.error(f"Embedding generation failed, data not upserted to collection: {collection_name}")
        except Exception as e:
            logger.error(f"Error saving to collection {collection_name}: {e}")
            traceback.print_exc()



# test saving
data1 = [
    {
        "id": 1,
        "content": "Our webinar on Building the Ultimate Hybrid Search takes you through the process of building a hybrid search system with Qdrant Query API.",
        "url": "https://example.com/webinar-hybrid-search"
    },
    {
        "id": 2,
        "content": "Explore the latest advancements in AI-powered chatbots and their applications in customer service.",
        "url": "https://example.com/ai-chatbots-customer-service"
    },
    {
        "id": 3,
        "content": "Join our workshop on effective machine learning model deployment using Docker and Kubernetes.",
        "url": "https://example.com/ml-deployment-docker-kubernetes"
    }
]
df = pd.DataFrame(data1)

# storing dense embedding only
# test = Save().save_to_collection("testing_dense",df,"dense")
   
# storing hybrid collection
test = Save().save_to_collection("testing",df)
