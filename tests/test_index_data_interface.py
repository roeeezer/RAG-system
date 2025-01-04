import unittest
from components.index_data_interface import Bm25Indexer
from components.common import WebTextSection

class TestBm25Indexer(unittest.TestCase):
    def setUp(self):
        self.indexer = Bm25Indexer()

    def test_index_data(self):
        data = [
            WebTextSection(doc_id="1", section_id="1", text="This is a test document."),
            WebTextSection(doc_id="2", section_id="1", text="This is another test document.")
        ]
        index = self.indexer.index_data(data)
        self.assertIsNotNone(index)
        self.assertIsInstance(index, dict)
        self.assertIn("1_1", index)
        self.assertIn("2_1", index)

    def test_retrieve_answer_source(self):
        queries = ["test document"]
        data = [
            WebTextSection(doc_id="1", section_id="1", text="This is a test document."),
            WebTextSection(doc_id="2", section_id="1", text="This is another test document.")
        ]
        self.indexer.index_data(data)
        answer_sources = self.indexer.retrieve_answer_source(queries)
        self.assertIsNotNone(answer_sources)
        self.assertIsInstance(answer_sources, list)
        self.assertGreater(len(answer_sources), 0)

if __name__ == '__main__':
    unittest.main()