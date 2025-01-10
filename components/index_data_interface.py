from abc import ABC, abstractmethod
import bm25s
from components.web_text_unit import WebTextUnit
from components.query import Query
import trankit

class IndexerInferface(ABC):
    @abstractmethod
    def index_data(self, web_text_units : list[WebTextUnit]):
        pass

    @abstractmethod
    def retrieve_answer_source(self, queries, k) -> list[WebTextUnit]:
        pass

class Bm25Indexer(IndexerInferface):

    def __init__(self):
        self.web_text_units = None
        self.corpus_tokens = None
        self.dictionary = None
        self.index = None
        self.pipeline = trankit.Pipeline("hebrew")

    def index_data(self, web_text_units : list[WebTextUnit]):
        self.web_text_units = web_text_units
        # Prepare the corpus for BM25
        corpus = [u.get_indexing_optimized_content() for u in web_text_units]

        # Initialize BM25 index
        self.index = bm25s.BM25()

        # 1) Tokenize the corpus and retrieve both tokens & dictionary
        self.corpus_tokens, self.dictionary = bm25s.tokenize(corpus)

        # 2) Build the index using these tokens
        self.index.index(self.corpus_tokens)

        return self.index

    def bm25_retrieve(self, query, k):

        ids = [self.dictionary.get(token) for token in query.split() if self.dictionary.get(token) is not None]

        if not ids:
            print(f"No valid tokens found for query: {query}")
            return []


        # Retrieve top-k results using BM25 index
        results = self.index.retrieve(query_tokens=[ids], k=k)  # No need to pass `corpus` here

        # Extract top-k document indices and scores
        doc_indices = results.documents[0]  # First query's results
        scores = results.scores[0]  # First query's scores


        return [self.web_text_units[doc_idx] for doc_idx in doc_indices]
    
    def retrieve_answer_source(self, queries: list[Query], k=1) -> list[WebTextUnit]:
        for query in queries:
            query.answer_sources = self.bm25_retrieve(query.indexing_optimized_query, k)

    def _preprocess_with_trankit(self, text: str) -> str:
        """Use Trankit to tokenize and lemmatize the input text"""
        if not text:
            return text
        
        # Perform lemmatization using Trankit
        lemmatize_tokens = self.pipeline.lemmatize(text)

        # Construct lemmatized text by iterating over tokens
        lemmatized_text = " ".join(
            token.get('lemma', token['text']) for sentence in lemmatize_tokens['sentences'] for token in sentence['tokens']
        )
    
        return lemmatized_text