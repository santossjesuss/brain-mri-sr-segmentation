class Experiment:
    def __init__(self, name, pipeline, config, training_dataset, validation_dataset, test_dataset, dataset_name, data_resolution=None):
        self.name = name
        self.pipeline = pipeline
        self.config = config
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        self.dataset_name = dataset_name
        self.data_resolution = data_resolution

    def run(self):
        print(f"Starting experiment: {self.name}")

        pipeline = self.pipeline(config=self.config, experiment_name=self.name, dataset_name=self.dataset_name, data_resolution=self.data_resolution)
        model, best_metrics = pipeline.run(self.training_dataset, self.validation_dataset)
        return model, best_metrics

    def test(self):
        print(f"Testing experiment: {self.name}")
        
        pipeline = self.pipeline(config=self.config, experiment_name=self.name, data_resolution=self.data_resolution)
        return pipeline.test(self.test_dataset)

    def predict(self, idx):
        print(f"Predicting on input image using experiment '{self.name}'")
        
        pipeline = self.pipeline(config=self.config, experiment_name=self.name, data_resolution=self.data_resolution)
        return pipeline.predict(self.test_dataset[idx])
    
    def predict_random(self):
        print(f"Predicting random sample from dataset '{self.dataset_name}' using experiment '{self.name}'")
        
        pipeline = self.pipeline(config=self.config, experiment_name=self.name, data_resolution=self.data_resolution)
        return pipeline.predict_random(self.test_dataset)