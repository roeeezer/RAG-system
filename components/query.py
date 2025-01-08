class Query:
    def __init__(self, gold_doc_id, query):
        self.gold_doc_id = gold_doc_id
        self.query = query
        self.retrieved_doc_id = None
        self.answer_sources = None
        self.final_answer = None