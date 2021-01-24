from enum import Enum

class BoundaryCondition(Enum):
    NO_SLIP = 2
    FREE_SLIP = 0

    @classmethod
    def get(cls, type: str):
        if type == 'no-slip':
            return cls.NO_SLIP
        elif type == 'free-slip':
            return cls.FREE_SLIP