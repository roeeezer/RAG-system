from components.index_data_interface import IndexerInferface
from components.web_text_unit import WebTextUnit
from components.query import Query
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from tqdm import tqdm
from typing import List

class LlmIndexer(IndexerInferface):
    def __init__(self, model, batch_size=8):
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
            self.embeddings.extend(self.model.encode(texts, normalize_embeddings=True))
        
        self.st_vectors = np.array(self.embeddings)
        
        
    def retrieve_answer_source(self, queries: List[Query], k: int) -> list[WebTextUnit]:
        queries_text = [query.query for query in queries]
        q_emb = self.model.encode(queries_text, normalize_embeddings=True)
        dot_scores = self.st_vectors @ q_emb.T


        
        topk_indices = np.argsort(-dot_scores)[:k]
        
        # Iterate over each query to extract the top-k document IDs for each query
        topk_doc_ids = [[self.doc_ids[i] for i in topk_indices[:, j]] for j in range(topk_indices.shape[1])]
        topk_web_text_units = [[self.web_text_units[i] for i in topk_indices[:, j]] for j in range(topk_indices.shape[1])]
        
        # Check if the topk_doc_ids are correct
        for q_doc, q_doc_id in zip(topk_web_text_units, topk_doc_ids):
            for doc, doc_id in zip(q_doc, q_doc_id):
                assert doc.get_doc_id() == doc_id

        for query, doc_ids, web_text_units in zip(queries, topk_doc_ids, topk_web_text_units):
            query.answer_sources = web_text_units

        return topk_web_text_units
        