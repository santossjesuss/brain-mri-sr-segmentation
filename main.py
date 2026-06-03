import random
from datasets.mslesseg_dataset import MSLesSegDataset
from experiments.mslesseg_experiments import MSLesSegExperiments
from experiments.fcdlesseg_experiments import FCDLesSegExperiments
from utils.execution_batches import execute_training_batch, execute_testing_batch, predict_batch
from utils.train_to_excel import train_to_excel
from utils.test_to_excel import test_to_excel
from utils.results_visualizer import visualize_superres_model, visualize_seg_model, visualize_all_experiments

def main():
    ms_experiments = MSLesSegExperiments()
    # fcd_experiments = FCDLesSegExperiments()
    
    # -|Complete experiments training|-
    # training_results = {}
    # training_results['MS'] = execute_training_batch(experiments=ms_experiments)
    # training_results['FCD'] = execute_training_batch(experiments=fcd_experiments)
    # train_to_excel(training_results)

    # -|Complete experiments testing|-
    # dataset_experiments = default_dataset_experiments()
    # test_to_excel(dataset_experiments)

    # ------------------------------------------------------

    # -|Visualization|-
    # --- Batch -------- #
    test_dataset = ms_experiments.test_dataset
    random_idx = random.randint(0, len(test_dataset) - 1)
    experiments_dict = predict_batch(experiments=ms_experiments, idx=random_idx)
    visualize_all_experiments(experiments_dict)

    # --- Individual --- #
    # experiment = ms_experiments.get_super_resolution()
    # experiment = ms_experiments.get_lr_segmentation()
    # experiment = ms_experiments.get_hr_segmentation()
    # experiment = ms_experiments.get_frozen_sr_frozen_seg()
    # experiment = ms_experiments.get_frozen_sr_trainable_seg()
    # experiment = ms_experiments.get_trainable_sr_frozen_seg()
    # experiment = ms_experiments.get_joint_sr_seg_e2e()
    # experiment = ms_experiments.get_joint_sr_seg_combined()
    
    # results = experiment.predict_random()
    # visualize_superres_model(results)     # Super-Resolution
    # visualize_seg_model(results)          # Segmentation

    # ------------------------------------------------------

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

if __name__ == "__main__":
    main()