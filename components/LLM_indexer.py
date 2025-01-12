from components.index_data_interface import IndexerInferface
from components.web_text_unit import WebTextUnit
from components.query import Query
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from tqdm import tqdm
from typing import List

class LlmIndexer(IndexerInferface):
    def __init__(self, model, batch_size=64):
        self.model = SentenceTransformer(model)
        self.embeddings = None
        self.web_text_units = None
        self.batch_size = batch_size
        self.st_vectors = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def index_data(self, web_text_units : list[WebTextUnit]):

        self.web_text_units = web_text_units
        self.embeddings = []
        self.doc_ids = []
        self.model.to(self.device)
        batches = [web_text_units[i:i + self.batch_size] for i in range(0, len(web_text_units), self.batch_size)]
        for batch in tqdm(batches):
            texts = [web_text_unit.get_content() for web_text_unit in batch]
            self.doc_ids.extend([web_text_unit.get_doc_id() for web_text_unit in batch])
            embedding = self.model.encode(texts, normalize_embeddings=True)
            self.embeddings.extend(embedding)
        
        self.st_vectors = np.array(self.embeddings)
        
        
    def retrieve_answer_source(self, queries: List[Query], k: int) -> list[WebTextUnit]:
        queries_text = [query.query for query in queries]
        q_emb = self.model.encode(queries_text, normalize_embeddings=True)
        print(q_emb.shape)
        print(self.st_vectors.shape)
        dot_scores = q_emb @ self.st_vectors.T

        sorted_indices = np.argsort(-dot_scores)  # (num_docs, num_queries)
        print(sorted_indices.shape)
        topk_indices = sorted_indices[:, :k] # (num_queries)

        for query_idx, query in enumerate(queries):
            doc_indices_for_query = topk_indices[query_idx]
            query.answer_sources.extend([self.web_text_units[i] for i in doc_indices_for_query])

        