
from qdrant_client import QdrantClient
from qdrant_client.http import models
import os
import traceback
import pandas as pd
from typing import List
from qdrant_client.models import PointStruct, PointIdsList
from dotenv import load_dotenv

OPEN_AI_VECTOR_SIZE=1536
COLBERT_VECTOR_SIZE=128

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
class Qdrant:

    def __init__(self):

        try:
            self.client = QdrantClient(os.environ.get('QDRANT_CLIENT'))
            print(f"qdrant connected")
        except:
            print('qdrant connection is failed')


    def get_create_collection(self,collection_name, type): 
        
        try:
            self.client.get_collection(collection_name)
        except: 
            print("no such collection exists")
            try:
                logger.info(f"creating collection {collection_name}")
                if type=='hybrid':
                    self.client.create_collection(
                        collection_name,
                        vectors_config={
                            "open-ai":models.VectorParams(
                            size=OPEN_AI_VECTOR_SIZE,
                            distance=models.Distance.COSINE,
                            ),
                            "colbert":models.VectorParams(
                            size=COLBERT_VECTOR_SIZE,
                            distance=models.Distance.COSINE,
                            multivector_config=models.MultiVectorConfig(
                                comparator=models.MultiVectorComparator.MAX_SIM
                                )
                            ),
                        },
                        sparse_vectors_config={
                            "bm42": models.SparseVectorParams(
                                modifier=models.Modifier.IDF
                            )
                        }
                    )
                elif type=='dense':
                    self.client.create_collection(
                        collection_name,
                        vectors_config=models.VectorParams(size=OPEN_AI_VECTOR_SIZE, distance=models.Distance.DOT) )
                    print(f"Collection '{collection_name}' CREATED.")
            except:
                traceback.print_exc()
                logger.info("error creating a collection")


    def upsert_data(self,collection_name,df, type):
        try:
            if type== 'hybrid':
                
                self.get_create_collection(collection_name,type)
                
                excluded_columns = {"dense", "sparse", "latent"}
                payload_columns = [col for col in df.columns if col not in excluded_columns]

                points = [
                    models.PointStruct(
                        id=row["id"],
                        vector={
                            "open-ai": row["dense"],
                            "bm42": row["sparse"].as_object(),
                            "colbert": row["latent"]
                        },
                        payload={col: row[col] for col in payload_columns}  
                    )
                    for _, row in df.iterrows()
                ]

                self.client.upload_points(
                        collection_name,
                        points = points)

                print("data uploaded")
            elif type == 'dense':
                excluded_columns = {"dense"}
                payload_columns = [col for col in df.columns if col not in excluded_columns]
                payloads_list = [
                            {col: getattr(item, col) for col in payload_columns}
                            for item in df.itertuples(index=False)]

                self.get_create_collection(collection_name,type)
                self.client.upsert(
                    collection_name=collection_name,
                    points=models.Batch(
                        ids=df["id"].tolist(),
                        vectors=df["dense"].tolist(),
                        payloads=payloads_list,
                    ),)
                print("embedding saved")

        except:
                traceback.print_exc()
                print("error saving")
            

    def retrieve_data(self,collection,query,type):

        if type == "dense":
            result = self.client.search(
                    collection_name=collection,
                    query_vector=query["dense"],
                    with_payload=True,
                    limit=1000)

            response = {}
            for point in result:
                response["result"] = [ {
                    "score": point.score,**point.payload 
                }]
            print(response)
            return response

        prefetch = [
                models.Prefetch(
                            query=query["dense"],
                            using="open-ai",
                            limit=100,
                        ),
                # Here we just add another 25 documents using the sparse
                # vectors only.
                models.Prefetch(
                            query=query["sparse"],
                            using="bm42",
                            limit=25,
                )
        ]

        # Finally rerank the results with the late interaction model.
        result = self.client.query_points(
            collection,
            prefetch=prefetch,
            query=query["latent"],
            using="colbert",
            with_payload=True,
            limit=10,)
        
       
        response = {}
        response["result"] = [ {
                "score": point.score,
                **point.payload  # Merge the payload contents into the main dictionary
            }
            for point in result.points
        ]

        return response


