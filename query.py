import numpy as np
from embedding import openai_embedding_model, sparse_vector, late_interaction
from qdrant import Qdrant
from llm_response import get_result_from_llm_background
import traceback
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
class RAG:

    def __init__(self) -> None:
        self.client = Qdrant()

    def query(self, query_str, collection_name,type):
        try:
            logger.info("Query embedding started")

            if isinstance(query_str, str):
                query_str = [query_str]

            query = {}
            embeddings = openai_embedding_model(query_str)
            if not embeddings or len(embeddings) == 0:
                logger.error("Failed to generate dense embeddings")
                return None
            
            embed = np.array(embeddings)
            query["dense"] = embed.reshape(-1, 1536).tolist()[0]

            if type=="dense":
                result = self.client.retrieve_data(collection_name, query,type)
                return result
                
            sparse = sparse_vector(query_str)
            if sparse is None or len(sparse) == 0:
                logger.error("Failed to generate sparse embeddings")
                return None

            query["sparse"] = sparse[0].as_object()

            latent = late_interaction(query_str)
            if latent is None or len(latent) == 0:
                logger.error("Failed to generate latent interaction embeddings")
                return None

            query["latent"] = latent[0].tolist()

            '''
            todo: if one faild only use the one working
            '''
            result = self.client.retrieve_data(collection_name, query,type)
            logger.info("Query embedding finished")
            return result

        except Exception as e:
            logger.error(f"An error occurred during query processing: {e}")
            traceback.print_exc()
            return None

    def result(self, query_str, collection_name, type="hybrid"):
        try:
            query_result = self.query(query_str, collection_name, type)
            if query_result is None:
                logger.error("No query result to process")
                return None

            result = get_result_from_llm_background(query_str, query_result)
            return result

        except Exception as e:
            logger.error(f"An error occurred while generating the result: {e}")
            traceback.print_exc()
            return None

# testing making a query from dense
# print(RAG().result("advancementes in ai", "testing_dense","dense"))

# testing making query from hybrid
print(RAG().result("advancementes in ai", "testing"))
