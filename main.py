import csv
from components.query import Query
from components.pre_process_data_interface import PreProcessDataInterface, WebDataPreProccessor
from components.index_data_interface import IndexerInferface, Bm25Indexer
from components.llm_answer_retriever_interface import LlmAnswerRetrieverInterface, GeminiFreeTierAnswerRetriever
from components.rag_results import RagResults

use_small_data = False
small_suffix = "_small" if use_small_data else ""
eval_set_name = f'eval-set{small_suffix}.csv'
web_database_name = f'created_kol_zchut_corpus{small_suffix}'

class Rag:
    def __init__(self, pre_proccessor: PreProcessDataInterface, index_data_impl: IndexerInferface, get_final_answers_impl: LlmAnswerRetrieverInterface):
        self.pre_proccessor = pre_proccessor
        self.index_data_impl = index_data_impl
        self.final_answers_retrievers = get_final_answers_impl

    def answer_queries(self, queries: list[Query]):
        web_text_units = self.pre_proccessor.load_or_process_data()
        self.index_data_impl.index_data(web_text_units)
        self.index_data_impl.retrieve_answer_source(queries)
        self.final_answers_retrievers.retrieve_final_answers(queries)

def parse_queries_csv(file_path):
    queries = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            gold_doc_id, query = row
            queries.append(Query(gold_doc_id, query))
    return queries

def run_rag():
    queries = parse_queries_csv(eval_set_name)
    queries = queries[:20]
    pre_proccessor = WebDataPreProccessor(web_database_name)
    index_data_impl = Bm25Indexer()
    get_final_answers_impl = GeminiFreeTierAnswerRetriever()

    rag = Rag(pre_proccessor, index_data_impl, get_final_answers_impl)
    rag.answer_queries(queries)

    results = RagResults(
        queries=queries,
        pre_proccessor_name=pre_proccessor.__class__.__name__,
        index_data_impl_name=index_data_impl.__class__.__name__,
        get_final_answers_impl_name=get_final_answers_impl.__class__.__name__
    )
    results.save_to_file('test_results/rag_results.json')

if __name__ == "__main__":
    run_rag()