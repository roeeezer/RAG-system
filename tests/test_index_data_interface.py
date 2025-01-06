import unittest
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from components.index_data_interface import Bm25Indexer
from components.web_text_unit import WebTextSection
from components.pre_process_data_interface import WebDataPreProccessor

class TestBm25Indexer(unittest.TestCase):
    def setUp(self):
        self.indexer = Bm25Indexer()

    def test_retrieve_answer_source(self):
        try:
            data = [
                WebTextSection(doc_id="1", section_id="1", content="This is a test document."),
                WebTextSection(doc_id="2", section_id="1", content="This is another document.")
            ]
            self.indexer.index_data(data)
            queries = ["test document"]
            answer_sources = self.indexer.retrieve_answer_source(queries, 1)
            self.assertIsNotNone(answer_sources)
            self.assertIsInstance(answer_sources, list)
            self.assertGreater(len(answer_sources), 0)
            self.assertIsInstance(answer_sources[0][0], WebTextSection)
            answer_sources[0][0].get_id() == "1_1"
        except Exception as e:
            self.fail(f"Exception raised: {e}")

    def test_retrieve_answer_real_data(self):
        try:
            pre_proccessor = WebDataPreProccessor("created_kol_zchut_corpus_small")
            web_text_units = pre_proccessor.pre_proccess_data()
            self.indexer.index_data(web_text_units)
            queries = ["מי ממן את הוצאות הקבורה ושירותי הקבורה המקובלים?"]
            answer_sources = self.indexer.retrieve_answer_source(queries, 5)
            self.assertIsNotNone(answer_sources)
            self.assertIsInstance(answer_sources, list)
            self.assertGreater(len(answer_sources), 0)
            self.assertIsInstance(answer_sources[0][0], WebTextSection)
            self.assertTrue(answer_sources[0][0].get_id().startswith("00a18116"))
        except Exception as e:
            self.fail(f"Exception raised: {e}")

if __name__ == '__main__':
    unittest.main()