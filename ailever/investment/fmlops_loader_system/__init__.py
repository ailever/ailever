from .integrated_loader import Loader
from .parallelizer import Parallelization_Loader
from .preprocessor import Preprocessor

PLoader = Parallelization_Loader()
parallelize = PLoader.parallelize

