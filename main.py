import csv
from components.query import Query
from components.pre_process_data_interface import WebDataPreProccessor
from components.index_data_interface import Bm25Indexer
from components.LlmAnswerRetriever.llm_answer_retriever_interface import EmptyAnswerRetrieverInterface
from components.LlmAnswerRetriever.GeminiFreeTierAnswerRetriever import GeminiFreeTierAnswerRetriever
from components.rag_results import RagResults
from components.IndexOptimizer.indexing_text_optimizer_interface import  LemmatizerIndexOptimizer as Lema
from components.rag import Rag

use_small_data = False
small_suffix = "_small" if use_small_data else ""
eval_set_name = f'eval-set{small_suffix}.csv'
web_database_name = f'kolzchut{small_suffix}'

def parse_queries_csv(file_path) -> list[Query]:
    queries = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            gold_doc_id, query = row
            queries.append(Query(gold_doc_id, query))
    return queries

def run_rag():
    queries = parse_queries_csv(eval_set_name)
    queries = queries[:15]
    pre_proccessor = WebDataPreProccessor(web_database_name)
    index_optimizers = [Lema()]
    index_data_impl = Bm25Indexer()
    get_final_answers_retriever = EmptyAnswerRetrieverInterface()

    rag = Rag(pre_proccessor, 
              index_data_impl, 
              get_final_answers_retriever,
              index_optimizers)
    rag.answer_queries(queries)

    results = RagResults(rag=rag, queries=queries)
    results.save_to_file('test_results/rag_results.json')

if __name__ == "__main__":
    run_rag()