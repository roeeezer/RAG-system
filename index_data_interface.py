from abc import ABC, abstractmethod

class IndexDataInterface(ABC):
    @abstractmethod
    def index_data(self, data):
        pass

class IndexDataImplementation(IndexDataInterface):
    def index_data(self, data):
        # Your implementation here
        pass