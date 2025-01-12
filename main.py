import csv
import os
from components.query import Query
from components.pre_process_data_interface import WebDataPreProccessor
from components.index_data_interface import Bm25Indexer
from components.LlmAnswerRetriever.llm_answer_retriever_interface import EmptyAnswerRetrieverInterface
from components.LlmAnswerRetriever.GeminiFreeTierAnswerRetriever import GeminiFreeTierAnswerRetriever
from components.rag_results import RagResults
from components.rag import Rag
from components.IndexOptimizer.word_filtering_indexing_optimizer import WordFilteringIndexingOptimizer
from components.LlmAnswerRetriever.gemini import Gemini
from components.IndexOptimizer.hyde_indexing_optimizer import HydeIndexingOptimizer
from components.Instractor_indexer import InstractorIndexer
from components.IndexOptimizer.synonym_encrichment_optimizer import SynonymEnrichmentOptimizer
from components.IndexOptimizer.prefix_suffix_splitter_optimizer import PrefixSuffixSplitterOptimizer
from components.LLM_indexer import LlmIndexer 


data_set_name = f'eval-set'
web_database_name = f'kolzchut'
constrained_model = False
cons = "_cons" if constrained_model else ""

def parse_queries_csv(file_path) -> list[Query]:
    queries = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            gold_doc_id, query = row
            queries.append(Query(gold_doc_id, query))
    return queries

def run_rag():
    queries = parse_queries_csv(f"{data_set_name}.csv")
    queries = queries[:100]

    rag = build_rag()
    rag.answer_queries(queries)

    results = RagResults(rag=rag, queries=queries)
    result_path = os.path.join("test_results",f"{data_set_name}", f"res{cons}.json")
    results.save_to_file(result_path)

def build_rag():
    gemini = Gemini(constraint_model=constrained_model)
    st_model = 'intfloat/multilingual-e5-large'

    pre_proccessor = WebDataPreProccessor(web_database_name)
    index_optimizers = []
    index_data_impl = InstractorIndexer(model=st_model)
    get_final_answers_retriever = EmptyAnswerRetrieverInterface()
    rag = Rag(pre_proccessor, 
              index_data_impl, 
              get_final_answers_retriever,
              index_optimizers)
              
    return rag

def query(query_object: str)->str:
    rag = build_rag()
    query_object = Query(None, query_object)
    rag.answer_queries([query_object])
    return query_object.final_answer

if __name__ == "__main__":
    # print(query("האם עיוור אמור לשלם על כלב נחייה בתחבורה ציבורית?"))
    run_rag()