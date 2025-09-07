from abc import ABC, abstractmethod
import bm25s
from components.web_text_unit import WebTextUnit
from components.query import Query
from components.logger import Logger

class IndexerInferface(ABC):
    @abstractmethod
    def index_data(self, web_text_units : list[WebTextUnit]):
        pass

    @abstractmethod
    def retrieve_answer_source(self, queries, k) -> list[WebTextUnit]:
        pass

class Bm25Indexer(IndexerInferface):

    def __init__(self):
        self.logger = Logger().get_logger()
        self.web_text_units = None
        self.corpus_tokens = None
        self.dictionary = None
        self.index = None

    def index_data(self, web_text_units : list[WebTextUnit]):
        self.logger.debug(f'Entering index_data with {len(web_text_units)} web_text_units')
        try:
            self.web_text_units = web_text_units
            # Prepare the corpus for BM25
            corpus = [u.get_indexing_optimized_content() for u in web_text_units]

            # Initialize BM25 index
            self.index = bm25s.BM25()

            # 1) Tokenize the corpus and retrieve both tokens & dictionary
            self.corpus_tokens, self.dictionary = bm25s.tokenize(corpus)

            # 2) Build the index using these tokens
            self.index.index(self.corpus_tokens)
            self.logger.debug('BM25 index built successfully')
        except Exception as e:
            self.logger.error(f'Error in index_data: {e}')
            raise
        self.logger.debug('Exiting index_data')
        return self.index

    def bm25_retrieve(self, query, k):
        self.logger.debug(f'Entering bm25_retrieve with query="{query}", k={k}')
        try:
            ids = [self.dictionary.get(token) for token in query.split() if self.dictionary.get(token) is not None]

            if not ids:
                self.logger.warning(f'No valid tokens found for query: {query}')
                return []

            # Retrieve top-k results using BM25 index
            results = self.index.retrieve(query_tokens=[ids], k=k)  # No need to pass `corpus` here

            # Extract top-k document indices and scores
            doc_indices = results.documents[0]  # First query's results
            scores = results.scores[0]  # First query's scores

            self.logger.debug(f'bm25_retrieve found {len(doc_indices)} documents')
            return [self.web_text_units[doc_idx] for doc_idx in doc_indices]
        except Exception as e:
            self.logger.error(f'Error in bm25_retrieve: {e}')
            raise
        finally:
            self.logger.debug('Exiting bm25_retrieve')
    
    def retrieve_answer_source(self, queries: list[Query], k=1) -> list[WebTextUnit]:
        self.logger.debug(f'Entering retrieve_answer_source with {len(queries)} queries, k={k}')
        try:
            for query in queries:
                query.answer_sources = self.bm25_retrieve(query.indexing_optimized_query, k)
                self.logger.debug(f'Query "{query.query}" retrieved {len(query.answer_sources)} answer sources')
        except Exception as e:
            self.logger.error(f'Error in retrieve_answer_source: {e}')
            raise
        self.logger.debug('Exiting retrieve_answer_source')

    def _preprocess_with_trankit(self, text: str) -> str:
        self.logger.debug(f'Entering _preprocess_with_trankit with text of length {len(text) if text else 0}')
        if not text:
            return text
        try:
            # Perform lemmatization using Trankit
            lemmatize_tokens = self.pipeline.lemmatize(text)

            # Construct lemmatized text by iterating over tokens
            lemmatized_text = " ".join(
                token.get('lemma', token['text']) for sentence in lemmatize_tokens['sentences'] for token in sentence['tokens']
            )
            self.logger.debug('_preprocess_with_trankit completed successfully')
            return lemmatized_text
        except Exception as e:
            self.logger.error(f'Error in _preprocess_with_trankit: {e}')
            raise