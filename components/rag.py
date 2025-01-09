from components.query import Query
from components.pre_process_data_interface import PreProcessDataInterface
from components.index_data_interface import IndexerInferface
from components.LlmAnswerRetriever.llm_answer_retriever_interface import LlmAnswerRetrieverInterface
from components.IndexOptimizer.indexing_text_optimizer_interface import IndexingTextOptimizerInterface
from components.web_text_unit import WebTextUnit

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

    def answer_queries(self, queries: list[Query]):
        web_text_units = self.pre_proccessor.load_or_process_data()
        self.optimize_queries(queries)
        self.optimize_text_units(web_text_units)
        self.index_data_impl.index_data(web_text_units)
        self.index_data_impl.retrieve_answer_source(queries, k=20)
        self.final_answers_retrievers.retrieve_final_answers(queries)

    def optimize_queries(self, queries: list[Query]) -> None:
        for query in queries:
            optimized_query = query.query
            for optimizer in self.indexing_optimizers:
                optimized_query = optimizer.optimize_query(optimized_query)
            query.indexing_optimized_query = optimized_query
    
    def optimize_text_units(self, web_text_units: list[WebTextUnit]) -> None:
        for unit in web_text_units:
            optimized_text = unit.get_content()
            for optimizer in self.indexing_optimizers:
                optimized_text = optimizer.optimize_document(optimized_text)
            unit.indexing_optimized_content = optimized_text