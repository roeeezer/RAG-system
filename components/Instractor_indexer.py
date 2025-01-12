from components.index_data_interface import IndexerInferface
from components.web_text_unit import WebTextUnit
from components.query import Query
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from tqdm import tqdm
from typing import List

class InstractorIndexer(IndexerInferface):
    def __init__(self, model, batch_size=64):
        self.model = SentenceTransformer(model)
        self.doc_instraction = "Represents the document for retrieval: "
        self.query_instraction = "Represents the query for retrieval: "
        
        self.embeddings = None
        self.web_text_units = None
        self.doc_ids = []
        self.batch_size = batch_size
        self.st_vectors = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Move the model to the chosen device
        self.model.to(self.device)

    def index_data(self, web_text_units: List[WebTextUnit]):
        """
        1. Stores the WebTextUnits.
        2. Embeds each document using INSTRUCTOR format: [ [doc_instraction, text], ... ].
        3. Accumulates embeddings (self.st_vectors) and document IDs (self.doc_ids).
        """
        self.web_text_units = web_text_units
        self.embeddings = []
        self.doc_ids = []

        # Split into batches
        batches = [
            web_text_units[i : i + self.batch_size] 
            for i in range(0, len(web_text_units), self.batch_size)
        ]

        for batch in tqdm(batches, desc="Indexing documents"):
            # Prepare INSTRUCTOR-formatted inputs: [[instruction, text], ...]
            instructor_batch = [
                [self.doc_instraction, web_text.get_content()] 
                for web_text in batch
            ]
            # Encode (normalize for cosine similarity)
            doc_embs = self.model.encode(
                instructor_batch,
                device=self.device,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            self.embeddings.extend(doc_embs)

            # Store doc IDs in the same order
            self.doc_ids.extend([web_text.get_doc_id() for web_text in batch])

        # Convert list of embeddings to a NumPy array
        self.st_vectors = np.array(self.embeddings)

    def retrieve_answer_source(self, queries: List[Query], k: int) -> List[List[WebTextUnit]]:
        """
        1. Embeds each query with INSTRUCTOR format: [[query_instraction, query_text]].
        2. Computes similarity (dot product) with document embeddings.
        3. For each query, retrieves top-k documents (highest dot scores).
        4. Stores these in `query.answer_sources`.
        5. Returns a list of lists, where each inner list is the top-k WebTextUnits for that query.
        """
        # Step 1: Embed queries
        query_embeddings = []
        for query in queries:
            instructor_input = [[self.query_instraction, query.query]]
            q_emb = self.model.encode(
                instructor_input,
                device=self.device,
                convert_to_numpy=True,
                normalize_embeddings=True
            )[0]  # Single query → single embedding
            query_embeddings.append(q_emb)

        q_emb = np.array(query_embeddings)  # Shape: (num_queries, emb_dim)

        # Dot product for similarity (same as cosine if normalized)
        dot_scores = self.st_vectors @ q_emb.T

        # Step 3: For each query, find top-k docs
        sorted_indices = np.argsort(-dot_scores, axis=0)  # shape: (num_docs, num_queries)
        topk_indices = sorted_indices[:k, :]              # shape: (k, num_queries)

        # Step 4: Assign results to each query
        # We'll build a list of lists: results_for_all_queries
        results_for_all_queries = []
        for query_idx, query in enumerate(queries):
            doc_indices_for_this_query = topk_indices[:, query_idx]
            retrieved_docs = [self.web_text_units[i] for i in doc_indices_for_this_query]
            # Store in the query object
            query.answer_sources = retrieved_docs
            results_for_all_queries.append(retrieved_docs)

        # Step 5: Return top-k WebTextUnits for each query
        return results_for_all_queries
