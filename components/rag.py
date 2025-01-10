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
            queries[start:min(start + self.batch_size, len(queries))]  # Keep `Query` objects instead of just strings
            for start in range(0, len(queries), self.batch_size)
        ]
        
        for batch in tqdm(batched_queries):
            query_texts = [q.query for q in batch]  # Extract the text of the queries for optimization
            optimized_queries = query_texts  # Initialize with the raw texts of queries

            for optimizer in self.indexing_optimizers:
                optimized_queries = optimizer.optimize_query(optimized_queries)  # Optimize the list of query texts
            # Assign optimized queries back to the corresponding `Query` objects
            for i, query in enumerate(batch):
                query.indexing_optimized_query = optimized_queries[i]
           
    def optimize_text_units(self, web_text_units: list[WebTextUnit]) -> None:
        batched_units = [
            web_text_units[start:min(start + self.batch_size, len(web_text_units))]  # Keep `WebTextUnit` objects instead of just strings
            for start in range(0, len(web_text_units), self.batch_size)
        ]
        
        for batch in tqdm(batched_units):
            unit_contents = [unit.get_content() for unit in batch]  # Extract the text contents for optimization
            optimized_contents = unit_contents  # Initialize with the raw contents
            
            for optimizer in self.indexing_optimizers:
                optimized_contents = optimizer.optimize_document(optimized_contents)  # Optimize the list of contents
            
            # Assign optimized contents back to the corresponding `WebTextUnit` objects
            for i, unit in enumerate(batch):
                unit.indexing_optimized_content = optimized_contents[i]

