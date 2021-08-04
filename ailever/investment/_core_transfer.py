from typing import TypeVar, Generic

DTC = TypeVar('DTC')
class DataTransferCore(Generic[DTC]):
    def __init__(self):
        self.log = None
        self.pkl = None
        self.list = None
        self.dict = None
        self.hdfs = None
        self.pdframe = None
        self.ndarray = None
        self.pttensor = None
        self.tftensor = None
    
MTC = TypeVar('MTC')
class ModelTransferCore(Generic[MTC]):
    def __init__(self):
        self.log = None
        self.pkl = None
        self.joblib = None
        self.onnx = None
        self.pt = None
        self.pb = None
    

