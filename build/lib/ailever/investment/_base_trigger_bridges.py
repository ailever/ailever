from ailever.investment import fmlops_bs
from .__base_structures import BaseTriggerBridge

from importlib import import_module

__all__ = ['TorchTriggerBridge', 'TensorflowTriggerBridge', 'SklearnTriggerBridge', 'StatsmodelsTriggerBridge']

class TorchTriggerBridge(BaseTriggerBridge):
    def initializing_local_model_registry(self):
        pass

    def initializing_remote_model_registry(self):
        pass

    def load_from_ailever_feature_store(self, train_specification):
        # [-]
        # return : features
        return None

    def load_from_ailever_source_repository(self, train_specification):
        # [-]
        # return : dataloader ,model, criterion, optimizer
        return None

    def load_from_ailever_model_registry(self, train_specification):
        # return : model, optimizer
        return None

    def load_from_ailever_metadata_store(self, train_specification):
        # return : model_specifications, outcome_reports
        return None


    def load_from_local_feature_store(self, train_specification):
        # return : features
        return None

    def load_from_local_source_repository(self, train_specification):
        # return : dataloader, model, criterion, optimizer
        return None

    def load_from_local_model_registry(self, train_specification):
        # [-]
        # instance update
        """
        checkpoint = torch.load(os.path.join(source_repository['source_repository'], source))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        """
        # return : model, optimizer
        return None

    def load_from_local_metadata_store(self, train_specification):
        # return : model_specifications, outcome_reports
        return None


    def load_from_remote_feature_store(self, train_specification):
        return None

    def load_from_remote_source_repository(self, train_specification):
        return None

    def load_from_remote_model_registry(self, train_specification):
        return None

    def load_from_remote_metadata_store(self, train_specification):
        return None


    def save_in_ailever_feature_store(self, train_specification):
        pass

    def save_in_ailever_source_repository(self, train_specification):
        pass

    def save_in_ailever_model_registry(self, train_specification):
        pass

    def save_in_ailever_metadata_store(self, train_specification):
        pass


    def save_in_local_feature_store(self, train_specification):
        # [-]
        pass

    def save_in_local_source_repository(self, train_specification):
        # [-]
        pass

    def save_in_local_model_registry(self, train_specification):
        # [-]
        saving_path = os.path.join(dir_path['model_specifications'], train_specification['saving_name']+'.pt')
        print(f"* Model's informations is saved({saving_path}).")
        torch.save({
            'model_state_dict': self.registry['model'].to('cpu').state_dict(),
            'optimizer_state_dict': self.registry['optimizer'].state_dict(),
            'epochs' : self.registry['epochs'],
            'cumulative_epochs' : self.registry['cumulative_epochs'],
            'training_loss': self.registry['train_mse'],
            'validation_loss': self.registry['validation_mse']}, saving_path)

    def save_in_local_metadata_store(self, train_specification):
        pass


    def save_in_remote_feature_store(self, train_specification):
        pass

    def save_in_remote_source_repository(self, train_specification):
        pass

    def save_in_remote_model_registry(self, train_specification):
        pass

    def save_in_remote_metadata_store(self, train_specification):
        pass



class TensorflowTriggerBridge(BaseTriggerBridge):
    def initializing_local_model_registry(self):
        pass

    def initializing_remote_model_registry(self):
        pass

    def load_from_ailever_feature_store(self, train_specification):
        # return : features
        return None

    def load_from_ailever_source_repository(self, train_specification):
        # return : dataloader ,model, criterion, optimizer
        return None

    def load_from_ailever_model_registry(self, train_specification):
        # return : model, optimizer
        return None

    def load_from_ailever_metadata_store(self, train_specification):
        # return : model_specifications, outcome_reports
        return None


    def load_from_local_feature_store(self, train_specification):
        # return : features
        return None

    def load_from_local_source_repository(self, train_specification):
        # return : dataloader, model, criterion, optimizer
        return None

    def load_from_local_model_registry(self, train_specification):
        # return : model, optimizer
        return None

    def load_from_local_metadata_store(self, train_specification):
        # return : model_specifications, outcome_reports
        return None


    def load_from_remote_feature_store(self, train_specification):
        return None

    def load_from_remote_source_repository(self, train_specification):
        return None

    def load_from_remote_model_registry(self, train_specification):
        return None

    def load_from_remote_metadata_store(self, train_specification):
        return None


    def save_in_ailever_feature_store(self, train_specification):
        pass

    def save_in_ailever_source_repository(self, train_specification):
        pass

    def save_in_ailever_model_registry(self, train_specification):
        pass

    def save_in_ailever_metadata_store(self, train_specification):
        pass


    def save_in_local_feature_store(self, train_specification):
        pass

    def save_in_local_source_repository(self, train_specification):
        pass

    def save_in_local_model_registry(self, train_specification):
        pass

    def save_in_local_metadata_store(self, train_specification):
        pass


    def save_in_remote_feature_store(self, train_specification):
        pass

    def save_in_remote_source_repository(self, train_specification):
        pass

    def save_in_remote_model_registry(self, train_specification):
        pass

    def save_in_remote_metadata_store(self, train_specification):
        pass



class SklearnTriggerBridge(BaseTriggerBridge):
    def initializing_local_model_registry(self):
        pass

    def initializing_remote_model_registry(self):
        pass

    def load_from_ailever_feature_store(self, train_specification):
        # return : features
        return None

    def load_from_ailever_source_repository(self, train_specification):
        # return : dataloader ,model, criterion, optimizer
        return None

    def load_from_ailever_model_registry(self, train_specification):
        # return : model, optimizer
        return None

    def load_from_ailever_metadata_store(self, train_specification):
        # return : model_specifications, outcome_reports
        return None


    def load_from_local_feature_store(self, train_specification):
        # return : features
        return None

    def load_from_local_source_repository(self, train_specification):
        # return : dataloader, model, criterion, optimizer
        return None

    def load_from_local_model_registry(self, train_specification):
        # return : model, optimizer
        return None

    def load_from_local_metadata_store(self, train_specification):
        # return : model_specifications, outcome_reports
        return None


    def load_from_remote_feature_store(self, train_specification):
        return None

    def load_from_remote_source_repository(self, train_specification):
        return None

    def load_from_remote_model_registry(self, train_specification):
        return None

    def load_from_remote_metadata_store(self, train_specification):
        return None


    def save_in_ailever_feature_store(self, train_specification):
        pass

    def save_in_ailever_source_repository(self, train_specification):
        pass

    def save_in_ailever_model_registry(self, train_specification):
        pass

    def save_in_ailever_metadata_store(self, train_specification):
        pass


    def save_in_local_feature_store(self, train_specification):
        pass

    def save_in_local_source_repository(self, train_specification):
        pass

    def save_in_local_model_registry(self, train_specification):
        pass

    def save_in_local_metadata_store(self, train_specification):
        pass


    def save_in_remote_feature_store(self, train_specification):
        pass

    def save_in_remote_source_repository(self, train_specification):
        pass

    def save_in_remote_model_registry(self, train_specification):
        pass

    def save_in_remote_metadata_store(self, train_specification):
        pass



class StatsmodelsTriggerBridge(BaseTriggerBridge):
    def initializing_local_model_registry(self):
        pass

    def initializing_remote_model_registry(self):
        pass

    def load_from_ailever_feature_store(self, train_specification):
        # return : features
        return None

    def load_from_ailever_source_repository(self, train_specification):
        # return : dataloader ,model, criterion, optimizer
        return None

    def load_from_ailever_model_registry(self, train_specification):
        # return : model, optimizer
        return None

    def load_from_ailever_metadata_store(self, train_specification):
        # return : model_specifications, outcome_reports
        return None


    def load_from_local_feature_store(self, train_specification):
        # return : features
        return None

    def load_from_local_source_repository(self, train_specification):
        # return : dataloader, model, criterion, optimizer
        return None

    def load_from_local_model_registry(self, train_specification):
        # return : model, optimizer
        return None

    def load_from_local_metadata_store(self, train_specification):
        # return : model_specifications, outcome_reports
        return None


    def load_from_remote_feature_store(self, train_specification):
        return None

    def load_from_remote_source_repository(self, train_specification):
        return None

    def load_from_remote_model_registry(self, train_specification):
        return None

    def load_from_remote_metadata_store(self, train_specification):
        return None


    def save_in_ailever_feature_store(self, train_specification):
        pass

    def save_in_ailever_source_repository(self, train_specification):
        pass

    def save_in_ailever_model_registry(self, train_specification):
        pass

    def save_in_ailever_metadata_store(self, train_specification):
        pass


    def save_in_local_feature_store(self, train_specification):
        pass

    def save_in_local_source_repository(self, train_specification):
        pass

    def save_in_local_model_registry(self, train_specification):
        pass

    def save_in_local_metadata_store(self, train_specification):
        pass


    def save_in_remote_feature_store(self, train_specification):
        pass

    def save_in_remote_source_repository(self, train_specification):
        pass

    def save_in_remote_model_registry(self, train_specification):
        pass

    def save_in_remote_metadata_store(self, train_specification):
        pass



