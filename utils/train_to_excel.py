import pandas as pd

def train_to_excel(results):
    rows = []
    for dataset_name, dataset_results in results.items():
        for model_name, model_results in dataset_results.items():
            _, metrics = model_results
            rows.append({
                'Dataset': dataset_name,
                'Model': model_name,
                **metrics
            })
        
    df = pd.DataFrame(rows)
    df.to_excel('training_results.xlsx', index=False)