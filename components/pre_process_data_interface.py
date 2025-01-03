from abc import ABC, abstractmethod

class PreProcessDataInterface(ABC):
    @abstractmethod
    def pre_proccess_data(self):
        pass

class PreProcessDataImplementation(PreProcessDataInterface):
    def pre_proccess_data(self):
        # Your implementation here
        pass