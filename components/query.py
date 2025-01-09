from components.web_text_unit import WebTextUnit

class Query:
    def __init__(self, gold_doc_id, query):
        self.gold_doc_id = gold_doc_id
        self.query = query
        self.indexing_optimized_query: None | str = None
        self.retrieved_doc_id: None | str = None 
        self.answer_sources: None | list[WebTextUnit] = None
        self.final_answer: None | str = None 