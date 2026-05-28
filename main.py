from experiments.mslesseg_experiments import MSLesSegExperiments
from experiments.fcdlesseg_experiments import FCDLesSegExperiments
from utils.execution_batches import execute_training_batch, execute_testing_batch
from utils.test_to_excel import test_to_excel, default_dataset_experiments, default_model_experiments

def main():
    # -|Quick checkings|-
    # experiments = MSLesSegExperiments()
    # experiments = FCDLesSegExperiments()
    # experiment = experiments.get_super_resolution()
    # experiment = experiments.get_lr_segmentation()
    # experiment = experiments.get_hr_segmentation()
    # experiment = experiments.get_frozen_sr_frozen_seg()
    # experiment = experiments.get_frozen_sr_trainable_seg()
    # experiment = experiments.get_trainable_sr_frozen_seg()
    # experiment = experiments.get_joint_sr_seg_e2e()
    # experiment = experiments.get_joint_sr_seg_combined()

    # experiment.run()
    # experiment.test()

    ms_experiments = MSLesSegExperiments()
    # fcd_experiments = FCDLesSegExperiments()
    
    # -|Complete experiments training|-
    # execute_training_batch(experiments=ms_experiments)
    # execute_training_batch(experiments=fcd_experiments)

    # -|Complete experiments testing|-
    dataset_experiments = default_dataset_experiments()
    test_to_excel(dataset_experiments)

if __name__ == "__main__":
    main()