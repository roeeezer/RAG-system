from components.query import Query
from components.pre_process_data_interface import PreProcessDataInterface
from components.index_data_interface import IndexerInferface
from components.LlmAnswerRetriever.llm_answer_retriever_interface import LlmAnswerRetrieverInterface
from components.IndexOptimizer.indexing_text_optimizer_interface import IndexingTextOptimizerInterface
from components.web_text_unit import WebTextUnit
from tqdm import tqdm

class Rag:
    def __init__(self, 
                 pre_proccessor: PreProcessDataInterface, 
                 index_data_impl: IndexerInferface, 
                 get_final_answers_impl: LlmAnswerRetrieverInterface,
                 index_optimizers: list[IndexingTextOptimizerInterface]):
        self.pre_proccessor = pre_proccessor
        self.index_data_impl = index_data_impl
        self.final_answers_retrievers = get_final_answers_impl
        self.indexing_optimizers = index_optimizers
        self.batch_size = 10

    def answer_queries(self, queries: list[Query]):
        print("Loading or processing data")
        web_text_units = self.pre_proccessor.load_or_process_data()
        print("Optimizing queries")
        self.optimize_queries(queries)
        print("Optimizing text units")
        self.optimize_text_units(web_text_units)
        print("Indexing data")
        self.index_data_impl.index_data(web_text_units)
        print("Retrieving answers")
        self.index_data_impl.retrieve_answer_source(queries, k=20)
        print("Retrieving final answers")
        self.final_answers_retrievers.retrieve_final_answers(queries)

    def optimize_queries(self, queries: list[Query]) -> None:
        batched_queries = [
            [queries[i].query for i in range(start, min(start + self.batch_size, len(queries)))] 
            for start in range(0, len(queries), self.batch_size)]
        for batch in tqdm(batched_queries):
            for optimizer in self.indexing_optimizers:
                optimized_queries = optimizer.optimize_queries(batch)
            for i, query in enumerate(batch):
                queries[i].indexing_optimized_query = optimized_queries[i]
            
    
    def optimize_text_units(self, web_text_units: list[WebTextUnit]) -> None:
        batched_units = [
            [web_text_units[i].get_content() for i in range(start, min(start + self.batch_size, len(web_text_units)))] 
            for start in range(0, len(web_text_units), self.batch_size)
        ]
        for batch in tqdm(batched_units):
            optimized_texts = self.op
            for i, unit in enumerate(batch):
                web_text_units[i].indexing_optimized_content = optimized_texts[i]