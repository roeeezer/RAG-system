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
from components.logger import Logger


data_set_name = f'eval-set_small'
web_database_name = f'kolzchut_small'
constrained_model = False
cons = "_cons" if constrained_model else ""
text_units_to_retrieve_per_indexer = 2

def parse_queries_csv(file_path) -> list[Query]:
    logger = Logger().get_logger()
    logger.debug(f'Entering parse_queries_csv with file_path={file_path}')
    queries = []
    try:
        with open(file_path, newline='', encoding='utf-8-sig') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                gold_doc_id, query = row
                queries.append(Query(gold_doc_id, query))
        logger.debug(f'Parsed {len(queries)} queries from {file_path}')
    except Exception as e:
        logger.error(f'Error in parse_queries_csv: {e}')
        raise
    logger.debug('Exiting parse_queries_csv')
    return queries

def run_rag():
    logger = Logger().get_logger()
    logger.debug('Entering run_rag')
    try:
        queries = parse_queries_csv(f"{data_set_name}.csv")
        queries = queries[:10]

        rag = build_rag()
        rag.answer_queries(queries)

        results = RagResults(rag=rag, queries=queries, text_units_to_retrieve_per_indexer=text_units_to_retrieve_per_indexer)
        result_path = os.path.join("test_results",f"{data_set_name}", f"res{cons}.json")
        results.save_to_file(result_path)
        logger.debug(f'run_rag completed and results saved to {result_path}')
    except Exception as e:
        logger.error(f'Error in run_rag: {e}')
        raise
    logger.debug('Exiting run_rag')

def build_rag():
    logger = Logger().get_logger()
    logger.debug('Entering build_rag')
    try:
        gemini = Gemini(constraint_model=constrained_model)
        st_model = 'intfloat/multilingual-e5-large'

        pre_proccessor = WebDataPreProccessor(web_database_name)
        index_optimizers = [PrefixSuffixSplitterOptimizer()]
        index_data_impl = [Bm25Indexer(), InstractorIndexer(model=st_model)]
        get_final_answers_retriever = GeminiFreeTierAnswerRetriever(gemini)
        rag = Rag(pre_proccessor, 
                  index_data_impl, 
                  get_final_answers_retriever,
                  index_optimizers,
                  text_units_to_retrieve_per_indexer)
        logger.debug('build_rag created Rag instance')
    except Exception as e:
        logger.error(f'Error in build_rag: {e}')
        raise
    logger.debug('Exiting build_rag')
    return rag

def query(query_object: str)->str:
    logger = Logger().get_logger()
    logger.debug(f'Entering query with query_object={query_object}')
    try:
        rag = build_rag()
        query_obj = Query(None, query_object)
        rag.answer_queries([query_obj])
        logger.debug('Exiting query')
        return query_obj.final_answer
    except Exception as e:
        logger.error(f'Error in query: {e}')
        raise

if __name__ == "__main__":
    # print(query("האם עיוור אמור לשלם על כלב נחייה בתחבורה ציבורית?"))
    run_rag()