import bentoml
import mlflow 

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.get_experiment_by_name('ghrs')

BEST_MODELS = [{"autoencoder_artifacts": 'runs:/97e4a424d2ce41c5b8e39bad15c540a7/autoencoder_artifacts'}, 
                {"kmeans_artifacts": 'runs:/97e4a424d2ce41c5b8e39bad15c540a7/kmeans_artifacts'}]

def save_model_to_bentoml(models: dict) -> None:
    bento_model = bentoml.mlflow.import_model(
        name=list(models.keys())[0],
        model_uri=list(models.values())[0],
        signatures={
            'predict': {
                'batchable': True
            }
        }
    )
    print(bento_model.tag)

if __name__ == '__main__':
    for model in BEST_MODELS:
        save_model_to_bentoml(models=model)