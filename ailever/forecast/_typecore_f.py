from .._typecore import _TypeCore, _TypeCoreLoad, _TypeCoreSave, BaseTypeCaster
import numpy as np
import pandas as pd
import torch

def ForecastTypeCaster(obj, outtype='array'):
    obj = BaseTypeCaster(obj, outtype='BTC')
    _TypeCoreSave(obj, module_path=['ailever', 'forecast'], DS='AIL_DS_Forecast.pkl')
    obj = _TypeCoreLoad(module_path=['ailever', 'forecast'], DS='AIL_DS_Forecast.pkl')
    obj = getattr(obj, outtype)
    return obj
