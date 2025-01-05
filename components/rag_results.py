import json
import os

class RagResults:
    def __init__(self, queries, pre_proccessor_name, index_data_impl_name, get_final_answers_impl_name):
        self.queries = queries
        self.pre_proccessor_name = pre_proccessor_name
        self.index_data_impl_name = index_data_impl_name
        self.get_final_answers_impl_name = get_final_answers_impl_name

    def to_dict(self):
        return {
            "queries": [self.query_to_dict(query) for query in self.queries],
            "pre_proccessor_name": self.pre_proccessor_name,
            "index_data_impl_name": self.index_data_impl_name,
            "get_final_answers_impl_name": self.get_final_answers_impl_name
        }

    def query_to_dict(self, query):
        return {
            "gold_doc_id": query.gold_doc_id,
            "query": query.query,
            "answer_source": [section.to_dict() for section in query.answer_source] if query.answer_source else None,
            "final_answer": query.final_answer
        }

    def save_to_file(self, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=4)