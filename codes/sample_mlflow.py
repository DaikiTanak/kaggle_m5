import mlflow
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME



if __name__ == "__main__":

    experiment_name = "sample_ex"  # 要修正
    mlflow.set_experiment(experiment_name)

    run_id = None  # 要修正

    if run_id is None:
        mlflow.start_run(run_id=None)
        run_id = mlflow.active_run().info.run_id  # run_idはこれで取得できる
    elif run_id:  # 後から、同じrun_idに追記する時
        mlflow.start_run(run_id=run_id)


    run_name = "task1"
    mlflow.set_tag(MLFLOW_RUN_NAME, run_name)  # Run Nameを指定


    # Log a parameter (key-value pair)
    mlflow.log_param("param1", 5)

    # Log a metric; metrics can be updated throughout the run
    mlflow.log_metric("AUC", 1)
    fold_scores = {"fold1_acc":0.91, "fold2_acc":0.99}
    mlflow.log_metrics(fold_scores)
