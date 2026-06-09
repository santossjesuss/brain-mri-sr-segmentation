from datasets.base_seg_dataset import BaseSegmentationDataset

class MSLesSegDataset(BaseSegmentationDataset):
    def get_default_lesion_folder_name(self):
        return 'MRIms_kde'
    
    def get_default_control_folder_name(self):
        return 'MRIcontrol_kde'