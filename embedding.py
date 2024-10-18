import openai
import time
import numpy as np
import os
from fastembed import SparseTextEmbedding, LateInteractionTextEmbedding
import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

openai.api_key = os.getenv('OPENAI_KEY')
EMBEDDING_MODEL = "text-embedding-3-small"
SPARSE_MODEL = "Qdrant/bm42-all-minilm-l6-v2-attentions"
LATENT_MODEL = "colbert-ir/colbertv2.0"


# Function to generate OpenAI embeddings
def openai_embedding_model(batch):
    embeddings = []
    batch_size = 1000
    sleep_time = 10

    for i in range(0, len(batch), batch_size):
        batch_segment = batch[i:i + batch_size]
        print(batch_segment)
        logger.info(f"Embedding batch {i // batch_size + 1} of {len(batch) // batch_size + 1}")

        try:
            response = openai.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch_segment
            )
            batch_embeddings = [data.embedding for data in response.data]
            embeddings.extend(batch_embeddings)

        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            time.sleep(sleep_time)
    
    return embeddings

# Function to generate sparse vectors
def sparse_vector(context_list):
    try:
        logger.info("Generating sparse embeddings")
        model_bm42 = SparseTextEmbedding(model_name=SPARSE_MODEL)
        sparse_embeddings = list(model_bm42.passage_embed(context_list))
        return sparse_embeddings

    except Exception as e:
        logger.error(f"Error generating sparse vector: {e}")
        return None

# Function for late interaction embeddings
def late_interaction(context_list):
    
    logger.info(f"Embedding batch ")
    try:
        logger.info("sparse_embed")
        lateInteraction = LateInteractionTextEmbedding(LATENT_MODEL)
        late_interaction = list(lateInteraction.passage_embed(context_list))
        return late_interaction
    except Exception as e:
        logger.info("error getting latent interaction")
