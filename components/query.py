from components.web_text_unit import WebTextUnit

class Query:
    _counter = 0

    def __init__(self, gold_doc_id, query):
        Query._counter += 1
        self.number = Query._counter
        self.gold_doc_id = gold_doc_id
        self.query = query
        self.indexing_optimized_query: None | str = None
        self.retrieved_doc_id: None | str = None 
        self.answer_sources: list[WebTextUnit] = []
        self.final_answer: None | str = None 
        self.rank: None | str = None