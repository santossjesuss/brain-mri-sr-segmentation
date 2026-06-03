def execute_training_batch(experiments):
    training_results = {}

    sr_experiment = experiments.get_super_resolution()
    training_results['SR'] = sr_experiment.run()

    lr_seg_experiment = experiments.get_lr_segmentation()
    training_results['LR Seg'] = lr_seg_experiment.run()

    hr_seg_experiment = experiments.get_hr_segmentation()
    training_results['HR Seg'] = hr_seg_experiment.run()

    frozen_sr_trainable_seg_experiment = experiments.get_frozen_sr_trainable_seg()
    training_results['Frozen SR -> Trainable Seg'] = frozen_sr_trainable_seg_experiment.run()

    trainable_sr_frozen_seg_experiment = experiments.get_trainable_sr_frozen_seg()
    training_results['Trainable SR -> Frozen Seg'] = trainable_sr_frozen_seg_experiment.run()

    joint_sr_seg_e2e_experiment = experiments.get_joint_sr_seg_e2e()
    training_results['Joint E2E'] = joint_sr_seg_e2e_experiment.run()

    joint_sr_seg_combined_experiment = experiments.get_joint_sr_seg_combined()
    training_results['Joint Combined'] = joint_sr_seg_combined_experiment.run()

    return training_results

def execute_testing_batch(experiments):
    lr_seg_experiment = experiments.get_lr_segmentation()
    lr_seg_experiment.test()

    frozen_sr_frozen_seg_experiment = experiments.get_frozen_sr_frozen_seg()
    frozen_sr_frozen_seg_experiment.test()

    frozen_sr_trainable_seg_experiment = experiments.get_frozen_sr_trainable_seg()
    frozen_sr_trainable_seg_experiment.test()

    trainable_sr_frozen_seg_experiment = experiments.get_trainable_sr_frozen_seg()
    trainable_sr_frozen_seg_experiment.test()

    joint_sr_seg_e2e_experiment = experiments.get_joint_sr_seg_e2e()
    joint_sr_seg_e2e_experiment.test()

    joint_sr_seg_combined_experiment = experiments.get_joint_sr_seg_combined()
    joint_sr_seg_combined_experiment.test()

def predict_batch(experiments, idx):
    experiments_dict = {}

    lr_seg_experiment = experiments.get_lr_segmentation()
    experiments_dict['LR Segmentation'] = lr_seg_experiment.predict(idx)

    frozen_sr_frozen_seg_experiment = experiments.get_frozen_sr_frozen_seg()
    experiments_dict['Frozen SR -> Frozen Seg'] = frozen_sr_frozen_seg_experiment.predict(idx)

    frozen_sr_trainable_seg_experiment = experiments.get_frozen_sr_trainable_seg()
    experiments_dict['Frozen SR -> Trainable Seg'] = frozen_sr_trainable_seg_experiment.predict(idx)

    trainable_sr_frozen_seg_experiment = experiments.get_trainable_sr_frozen_seg()
    experiments_dict['Trainable SR -> Frozen Seg'] = trainable_sr_frozen_seg_experiment.predict(idx)

    joint_sr_seg_e2e_experiment = experiments.get_joint_sr_seg_e2e()
    experiments_dict['Joint E2E'] = joint_sr_seg_e2e_experiment.predict(idx)

    joint_sr_seg_combined_experiment = experiments.get_joint_sr_seg_combined()
    experiments_dict['Joint Combined'] = joint_sr_seg_combined_experiment.predict(idx)

    return experiments_dict

def predict_random_batch(experiments):
    experiments_dict = {}

    lr_seg_experiment = experiments.get_lr_segmentation()
    experiments_dict['LR Segmentation'] = lr_seg_experiment.predict_random()

    frozen_sr_frozen_seg_experiment = experiments.get_frozen_sr_frozen_seg()
    experiments_dict['Frozen SR -> Frozen Seg'] = frozen_sr_frozen_seg_experiment.predict_random()

    frozen_sr_trainable_seg_experiment = experiments.get_frozen_sr_trainable_seg()
    experiments_dict['Frozen SR -> Trainable Seg'] = frozen_sr_trainable_seg_experiment.predict_random()

    trainable_sr_frozen_seg_experiment = experiments.get_trainable_sr_frozen_seg()
    experiments_dict['Trainable SR -> Frozen Seg'] = trainable_sr_frozen_seg_experiment.predict_random()

    joint_sr_seg_e2e_experiment = experiments.get_joint_sr_seg_e2e()
    experiments_dict['Joint E2E'] = joint_sr_seg_e2e_experiment.predict_random()

    joint_sr_seg_combined_experiment = experiments.get_joint_sr_seg_combined()
    experiments_dict['Joint Combined'] = joint_sr_seg_combined_experiment.predict_random()

    return experiments_dict