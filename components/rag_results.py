import json
import os
from components.query import Query
from components.rag import Rag

class RagResults:
    def __init__(self, rag: Rag, queries: list[Query]):
        self.version = "1.0.2"
        self.queries = queries
        self.wrong_retrieved_queries = self.get_wrong_retrieved_queries(queries)
        self.pre_proccessor_name = rag.pre_proccessor.__class__.__name__
        self.index_optimizer_names = [ optimizer.__class__.__name__ for optimizer in rag.indexing_optimizers]
        self.index_data_impl_name = "rag.index_data_impl.__class__.__name__"
        self.get_final_answers_impl_name = rag.final_answers_retrievers.__class__.__name__
        self.recall_20 = self.recall_at_k(queries, k=20)
        self.recall_5 = self.recall_at_k(queries, k=5)
        self.mmr = self.mrr(queries, k=20)
        self.llm_tokens_counter = rag.final_answers_retrievers.get_sent_tokens_counter()

    def get_wrong_retrieved_queries(self, queries : list[Query]):
        res = []
        for query in queries:
            retrived_doc_ids = [answer_source.get_doc_id() for answer_source in query.answer_sources]
            if query.gold_doc_id not in retrived_doc_ids:
                res.append(query)
        return res

    def to_dict(self):
        return {
            "version": self.version,
            "pre_proccessor_name": self.pre_proccessor_name,
            "index_optimizer_names": self.index_optimizer_names,
            "index_data_impl_name": self.index_data_impl_name,
            "get_final_answers_impl_name": self.get_final_answers_impl_name,
            "recall_20": self.recall_20,
            "recall_5": self.recall_5,
            "mmr": self.mmr,
            "average_recall_20_mrr": (self.recall_20 + self.mmr) / 2,
            "wrong_retrieved_queries": [self.query_to_dict(query) for query in self.wrong_retrieved_queries],
            "queries": [self.query_to_dict(query) for query in self.queries],
        }

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
    def recall_at_k(queries : list[Query], k=20):
        hits = 0
        for q in queries:
            # Our query object is assumed to have a .gold_doc_id and .query
            relevant_id = q.gold_doc_id
            # Retrieve top-k doc_ids based on the query text
            topk_doc_ids = set([section.doc_id for section in q.answer_sources[:k]])
            if relevant_id in topk_doc_ids:
                hits += 1
        return hits / len(queries) if queries else 0.0
