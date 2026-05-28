import pandas as pd
from experiments.mslesseg_experiments import MSLesSegExperiments
from experiments.fcdlesseg_experiments import FCDLesSegExperiments

def test_to_excel(dataset_experiments):
    rows = []
    for dataset_name, dataset_experiment in dataset_experiments.items():
        model_experiments = default_model_experiments(dataset_experiment)
        for model_name, model_experiment in model_experiments.items():
            metrics = model_experiment.test()
            rows.append({
                'Dataset': dataset_name,
                'Model': model_name,
                **metrics
            })
    
    df = pd.DataFrame(rows)
    df.to_excel('results.xlsx', index=False)

def default_dataset_experiments():
    return {
        'MS': MSLesSegExperiments(),
        # 'FCD': FCDLesSegExperiments()
    }

def default_model_experiments(dataset_experiments):
    return {
        'LR Seg': dataset_experiments.get_lr_segmentation(),
        'Frozen SR -> Frozen Seg': dataset_experiments.get_frozen_sr_frozen_seg(),
        'Frozen SR -> Trainable Seg': dataset_experiments.get_frozen_sr_trainable_seg(),
        'Trainable SR -> Frozen Seg': dataset_experiments.get_trainable_sr_frozen_seg(),
        'Joint E2E': dataset_experiments.get_joint_sr_seg_e2e(),
        'Joint Combined': dataset_experiments.get_joint_sr_seg_combined()
    }