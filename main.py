from components.pre_process_data_interface import PreProcessDataInterface, PreProcessDataImplementation
from components.index_data_interface import IndexerInferface, Bm25Indexer
from components.retrieve_answer_source_interface import RetrieveAnswerSourceInterface, RetrieveAnswerSourceImplementation
from components.get_final_answers_interface import GetFinalAnswersInterface, GetFinalAnswersImplementation

class Rag:
    def __init__(self, pre_process_impl: PreProcessDataInterface, index_data_impl: IndexerInferface, retrieve_answer_source_impl: RetrieveAnswerSourceInterface, get_final_answers_impl: GetFinalAnswersInterface):
        self.pre_process_impl = pre_process_impl
        self.index_data_impl = index_data_impl
        self.retrieve_answer_source_impl = retrieve_answer_source_impl # Roee: delete this unused class
        self.get_final_answers_impl = get_final_answers_impl

    def answer_queries(self, queries):
        data = self.pre_process_impl.pre_proccess_data()
        self.index_data_impl.index_data(data)
        answer_sources = self.index_data_impl.retrieve_answer_source(queries)
        return self.get_final_answers_impl.get_final_answers(answer_sources)

def run_rag():
    queries = ["מאיזה גיל אפשר לפרוש לפנסיה?"]

    pre_process_impl = PreProcessDataImplementation()
    index_data_impl = Bm25Indexer()
    retrieve_answer_source_impl = RetrieveAnswerSourceImplementation()
    get_final_answers_impl = GetFinalAnswersImplementation()

    rag = Rag(pre_process_impl, index_data_impl, retrieve_answer_source_impl, get_final_answers_impl)
    answers = rag.answer_queries(queries)
    print(answers)

def main():
    run_rag()

if __name__ == "__main__":
    main()