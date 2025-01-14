from components.query import Query
from components.pre_process_data_interface import PreProcessDataInterface
from components.index_data_interface import IndexerInferface
from components.LlmAnswerRetriever.llm_answer_retriever_interface import LlmAnswerRetrieverInterface
from components.IndexOptimizer.indexing_text_optimizer_interface import IndexingTextOptimizerInterface
from components.web_text_unit import WebTextUnit
from tqdm import tqdm
from typing import List
import numpy as np

class Rag:
    def __init__(
        self, 
        pre_proccessor: PreProcessDataInterface, 
        data_indexers: List[IndexerInferface],
        final_answer_retriever: LlmAnswerRetrieverInterface,
        index_optimizers: List[IndexingTextOptimizerInterface]
    ):
        self.pre_proccessor = pre_proccessor
        self.data_indexers = data_indexers
        self.final_answers_retrievers = final_answer_retriever
        self.indexing_optimizers = index_optimizers
        self.batch_size = 64

    def answer_queries(self, queries: List[Query]):
        """
        1. Load data or process data (e.g. from files).
        2. Optimize the queries.
        3. Optimize the text units.
        4. Index data using each indexer in data_indexers.
        5. Retrieve 'k' documents for each query from each indexer.
        6. Retrieve final answers (e.g. from an LLM) using the aggregated answer_sources.
        """
        print("Loading or processing data")
        web_text_units = self.pre_proccessor.load_or_process_data()

        print("Optimizing queries")
        self.optimize_queries(queries)

        print("Optimizing text units")
        self.optimize_text_units(web_text_units)

        print("Indexing data with each indexer")
        for idx, indexer in enumerate(self.data_indexers, 1):
            print(f" - Indexing data with indexer #{idx}")
            indexer.index_data(web_text_units)

        print("Retrieving answers from each indexer")
        # Retrieve top-k docs from each indexer, then aggregate
        self.retrieve_from_all_indexers(queries, k=20)

        print("Retrieving final answers")
        self.final_answers_retrievers.retrieve_final_answers(queries)

    def retrieve_from_all_indexers(self, queries: List[Query], k: int):
        """
        For each query, retrieve top-k documents from EACH indexer,
        and combine them into the query's 'answer_sources'.
        Round-robin approach ensures an interleaving of results.
        """
        # Initialize empty lists so we can accumulate from multiple indexers
        for query in queries:
            query.answer_sources = []

        # Retrieve from each indexer
        for idx, indexer in enumerate(self.data_indexers, 1):
            print(f" - Retrieving from indexer #{idx}")
            indexer.retrieve_answer_source(queries, k)

        # Combine the retrieved documents from all indexers
        for query in queries:
            answer_sources_by_indexer = [
                query.answer_sources[i*k : (i+1)*k]
                for i in range(len(self.data_indexers))
            ]

            # Check if any indexer returned fewer than k docs
            min_len = min(len(sources) for sources in answer_sources_by_indexer)
            if min_len < k:
                print(f" - Warning: Indexer(s) returned fewer than {k} docs")

            # We will round-robin through these slices.
            res = []
            set_ids = set()
            for i in range(min_len):
                for sources in answer_sources_by_indexer:
                    if sources[i].get_id() not in set_ids:
                        res.append(sources[i])
                        set_ids.add(sources[i].get_id())

            query.answer_sources = res


    def optimize_queries(self, queries: List[Query]) -> None:
        """
        Runs all index_optimizers on batched queries. 
        Optimized text is stored in query.indexing_optimized_query.
        """
        batched_queries = [
            queries[start : min(start + self.batch_size, len(queries))]
            for start in range(0, len(queries), self.batch_size)
        ]
        
        for batch in tqdm(batched_queries, desc="Optimizing Queries"):
            query_texts = [q.query for q in batch]
            optimized_queries = query_texts  # raw texts as initial

            for optimizer in self.indexing_optimizers:
                optimized_queries = optimizer.optimize_queries(optimized_queries)
            for i, query in enumerate(batch):
                query.indexing_optimized_query = optimized_queries[i]

    def optimize_text_units(self, web_text_units: List[WebTextUnit]) -> None:
        batched_units = [
            web_text_units[start : min(start + self.batch_size, len(web_text_units))]
            for start in range(0, len(web_text_units), self.batch_size)
        ]
        
        for batch in tqdm(batched_units, desc="Optimizing Text Units"):
            unit_contents = [unit.get_content() for unit in batch]
            optimized_contents = unit_contents  # raw contents as initial

            for optimizer in self.indexing_optimizers:
                optimized_contents = optimizer.optimize_documents(optimized_contents)

            for i, unit in enumerate(batch):
                unit.indexing_optimized_content = optimized_contents[i]
