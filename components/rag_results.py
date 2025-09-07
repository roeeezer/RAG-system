import json
import os
from components.query import Query
from components.rag import Rag

class RagResults:
    def __init__(self, rag: Rag, queries: list[Query], text_units_to_search_for: int):
        self.version = "1.0.2"
        self.queries = queries
        self.wrong_retrieved_queries = self.get_wrong_retrieved_queries(queries)
        self.pre_proccessor_name = rag.pre_proccessor.__class__.__name__
        self.index_optimizer_names = [ optimizer.__class__.__name__ for optimizer in rag.indexing_optimizers]
        self.index_data_impl_name = [ indexer.__class__.__name__ for indexer in rag.data_indexers]
        self.get_final_answers_impl_name = rag.final_answers_retrievers.__class__.__name__
        self.text_units_to_search_for = text_units_to_search_for
        self.effective_k_for_recall_20 = min(20, self.text_units_to_search_for)
        self.effective_k_for_recall_5  = min(5,  self.text_units_to_search_for)

        self.recall_20 = self.recall_at_k(queries, k=self.effective_k_for_recall_20)
        self.recall_5 = self.recall_at_k(queries, k=self.effective_k_for_recall_5)
        self.mmr = self.mrr(queries, k=text_units_to_search_for)
        self.llm_tokens_counter = rag.final_answers_retrievers.get_sent_tokens_counter()

    def get_wrong_retrieved_queries(self, queries : list[Query]):
        res = []
        for query in queries:
            retrived_doc_ids = [answer_source.get_doc_id() for answer_source in query.answer_sources]
            if query.gold_doc_id not in retrived_doc_ids:
                res.append(query)
        return res

    def to_dict(self):
        recall20_key = f"recall_20_effk_{self.effective_k_for_recall_20}"
        recall5_key  = f"recall_5_effk_{self.effective_k_for_recall_5}"
        data = {
            "version": self.version,
            "pre_proccessor_name": self.pre_proccessor_name,
            "index_optimizer_names": self.index_optimizer_names,
            "index_data_impl_name": self.index_data_impl_name,
            "get_final_answers_impl_name": self.get_final_answers_impl_name,
            "mmr": self.mmr,
            recall20_key: self.recall_20,
            recall5_key: self.recall_5,
            "wrong_retrieved_queries": [self.query_to_dict(query) for query in self.wrong_retrieved_queries],
            "queries": [self.query_to_dict(query) for query in self.queries],
        }
        return data

    def query_to_dict(self, query: Query):
        return {
            "number": query.number,
            "gold_doc_id": query.gold_doc_id,
            "query": query.query,
            "rank": query.rank,
            "indexing_optimized_query": query.indexing_optimized_query,
            "final_answer": query.final_answer,
            "answer_source": [section.to_dict() for section in query.answer_sources] if query.answer_sources else None,
        }
    

    def save_to_file(self, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=4)
    
    
    @staticmethod
    def mrr(queries : list[Query], k=1000):
        s = 0.0
        for query in queries:
            relevant_id = query.gold_doc_id
            topk_doc_ids = [section.doc_id for section in query.answer_sources[:k]]
            rr = 0.0
            for rank, doc_id in enumerate(topk_doc_ids, start=1):
                if doc_id == relevant_id:
                    query.rank = rank
                    rr = 1.0 / rank
                    break
            s += rr
        return s / len(queries) if queries else 0.0
    
    @staticmethod
    def recall_at_k(queries : list[Query], k):
        hits = 0
        for q in queries:
            # Our query object is assumed to have a .gold_doc_id and .query
            relevant_id = q.gold_doc_id
            # Retrieve top-k doc_ids based on the query text
            topk_doc_ids = set([section.doc_id for section in q.answer_sources[:k]])
            if relevant_id in topk_doc_ids:
                hits += 1
        return hits / len(queries) if queries else 0.0
