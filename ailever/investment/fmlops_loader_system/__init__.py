from .parallelizer import Parallelization_Loader
from .integrated_loader import Loader
from .preprocessor import Preprocessor

prllz_loader = Parallelized_Loader()
parallelize = prllz_loader.parallelize

