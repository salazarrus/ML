from abc import ABC, abstractmethod

class Vehicle(ABC):
    def __init__(self, id):
        self.id = id


    @abstractmethod
    def getMaxVelocity(self):
        pass
