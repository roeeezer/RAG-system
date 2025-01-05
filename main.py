from components.pre_process_data_interface import PreProcessDataInterface, WebDataPreProccessor
from components.index_data_interface import IndexerInferface, Bm25Indexer
from components.get_final_answers_interface import GetFinalAnswersInterface, GetFinalAnswersImplementation

class Rag:
    def __init__(self, pre_proccessor: PreProcessDataInterface, index_data_impl: IndexerInferface, get_final_answers_impl: GetFinalAnswersInterface):
        self.pre_proccessor = pre_proccessor
        self.index_data_impl = index_data_impl
        self.get_final_answers_impl = get_final_answers_impl

    def answer_queries(self, queries):
        web_text_units = self.pre_proccessor.pre_proccess_data()
        self.index_data_impl.index_data(web_text_units)
        answer_sources = self.index_data_impl.retrieve_answer_source(queries)
        return self.get_final_answers_impl.get_final_answers(queries, answer_sources)

def run_rag():
    queries = ["מי ממן את הוצאות הקבורה ושירותי הקבורה המקובלים?"]

    pre_proccessor = WebDataPreProccessor("created_kol_zchut_corpus_small")
    index_data_impl = Bm25Indexer()
    get_final_answers_impl = GetFinalAnswersImplementation()

    rag = Rag(pre_proccessor, index_data_impl, get_final_answers_impl)
    answers = rag.answer_queries(queries)
    print(answers)

def main():
    run_rag()

if __name__ == "__main__":
    main()