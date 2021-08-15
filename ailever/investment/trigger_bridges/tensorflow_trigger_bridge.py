from ailever.investment import __fmlops_bs__ as fmlops_bs
from ..__base_structures import BaseTriggerBridge

import os
from importlib import import_module

class TensorflowTriggerBridge(BaseTriggerBridge):
    def __init__(self):
        self.core_instances = dict()

    @staticmethod
    def _dynamic_import():
        pass
    
    def into_trigger_bridge(self):
        pass

    def into_trigger_block(self):
        pass

    def load_from_local_feature_store(self):
        pass

    def load_from_local_source_repository(self):
        pass

    def load_from_local_model_registry(self):
        pass

    def load_from_local_monitoring_source(self):
        pass

    def load_from_local_analysis_report_repository(self):
        pass

    def load_from_local_investment_outcome_repository(self):
        pass

    def load_from_local_metadata_store(self):
        pass

    def load_from_remote_feature_store(self):
        pass

    def load_from_remote_source_repository(self):
        pass

    def load_from_remote_model_registry(self):
        pass

    def load_from_remote_monitoring_source(self):
        pass

    def load_from_remote_analysis_report_repository(self):
        pass

    def load_from_remote_investment_outcome_repository(self):
        pass

    def load_from_remote_metadata_store(self):
        pass

    
    def save_in_local_feature_store(self):
        pass

    def save_in_local_source_repository(self):
        pass

    def save_in_local_model_registry(self):
        pass

    def save_from_local_monitoring_source(self):
        pass

    def save_from_local_analysis_report_repository(self):
        pass

    def save_from_local_investment_outcome_repository(self):
        pass

    
    def save_in_local_metadata_store(self):
        pass

    def save_in_remote_feature_store(self):
        pass

    def save_in_remote_source_repository(self):
        pass

    def save_in_remote_model_registry(self):
        pass

    def save_from_remote_monitoring_source(self):
        pass

    def save_from_remote_analysis_report_repository(self):
        pass

    def save_from_remote_investment_outcome_repository(self):
        pass

    def save_in_remote_metadata_store(self):
        pass

