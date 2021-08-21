from .parallelizer import Parallelization_Loader
from .integrated_loader import Loader
from .preprocessor import Preprocessor

prllz_loader = Parallelization_Loader()
parallelize = prllz_loader.parallelize

