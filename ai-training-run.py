# Kubeflow Pipeline Definition: AI Training Run

import kfp.dsl as dsl
import kfp.onprem as onprem

# Define Kubeflow Pipeline
@dsl.pipeline(
    # Define pipeline metadata
    name="AI Training Run",
    description="Template for executing an AI training run with built-in training dataset traceability and trained model versioning"
)
def ai_training_run(
    # Define variables that the user can set in the pipelines UI; set default values
    dataset_volume_pvc_existing: str = "dataset-vol",
    trained_model_volume_pvc_existing: str = "model-vol",
    execute_data_prep_step__yes_or_no: str = "yes",
    train_step_epochs: int = 15,
    train_step_batch_size: int = 128,
) :
    # Set GPU limits; Due to SDK limitations, this must be hardcoded
    train_step_num_gpu = 1
    validation_step_num_gpu = 0

    # Pipeline Steps:

    # Execute data prep step
    with dsl.Condition(execute_data_prep_step__yes_or_no == "yes") :
        data_prep = dsl.ContainerOp(
            name="data-prep",
            image="python:3",
            command=["sh", "-c"],
            arguments=["\
                python3 -m pip install pandas && \
                git clone https://github.com/yshimizu37/image_classification.git && \
                chmod -R 755 image_classification &&  \
                python3 ./image_classification/data_prep.py --datadir /mnt/dataset/cats_and_dogs_filtered"],
        )
        # Mount dataset volume/pvc
        data_prep.apply(
            onprem.mount_pvc(dataset_volume_pvc_existing, 'dataset', '/mnt/dataset')
        )

    # Create a snapshot of the dataset volume/pvc for traceability
    volume_snapshot_name = "dataset-{{workflow.uid}}"
    dataset_snapshot = dsl.ContainerOp(
        name="dataset-snapshot",
        image="python:3",
        command=["/bin/bash", "-c"],
        arguments=["\
            python3 -m pip install ipython kubernetes pandas tabulate && \
            git clone https://github.com/NetApp/netapp-data-science-toolkit && \
            mv /netapp-data-science-toolkit/Kubernetes/ntap_dsutil_k8s.py / && \
            echo '" + volume_snapshot_name + "' > /volume_snapshot_name.txt && \
            /ntap_dsutil_k8s.py create volume-snapshot --pvc-name=" + str(dataset_volume_pvc_existing) + " --snapshot-name=" + str(volume_snapshot_name) + " --namespace={{workflow.namespace}}"],
        file_outputs={"volume_snapshot_name": "/volume_snapshot_name.txt"}
    )
    # State that snapshot should be created after the data prep job completes
    dataset_snapshot.after(data_prep)

    # Execute training step
    train = dsl.ContainerOp(
        name="train-model",
        image="nvcr.io/nvidia/tensorflow:21.03-tf1-py3",
        command=["sh", "-c"],
        arguments=["\
            python3 -m pip install pandas && \
            git clone https://github.com/yshimizu37/image_classification.git && \
            chmod -R 755 image_classification &&  \
            python3 ./image_classification/train.py --datadir /mnt/dataset/cats_and_dogs_filtered --modeldir /mnt/model --batchsize " + str(train_step_batch_size) + " --epochs " + str(train_step_epochs)],
    )
    # Mount dataset volume/pvc
    train.apply(
        onprem.mount_pvc(dataset_volume_pvc_existing, 'datavol', '/mnt/dataset')
    )
    # Mount model volume/pvc
    train.apply(
        onprem.mount_pvc(trained_model_volume_pvc_existing, 'modelvol', '/mnt/model')
    )
    # Request that GPUs be allocated to training job pod
    if train_step_num_gpu > 0 :
        train.set_gpu_limit(train_step_num_gpu, 'nvidia')
    # State that training job should be executed after dataset volume snapshot is taken
    train.after(dataset_snapshot)

    # Create a snapshot of the model volume/pvc for model versioning
    volume_snapshot_name = "kfp-model-{{workflow.uid}}"
    model_snapshot = dsl.ContainerOp(
        name="model-snapshot",
        image="python:3",
        command=["/bin/bash", "-c"],
        arguments=["\
            python3 -m pip install ipython kubernetes pandas tabulate && \
            git clone https://github.com/NetApp/netapp-data-science-toolkit && \
            mv /netapp-data-science-toolkit/Kubernetes/ntap_dsutil_k8s.py / && \
            echo '" + volume_snapshot_name + "' > /volume_snapshot_name.txt && \
            /ntap_dsutil_k8s.py create volume-snapshot --pvc-name=" + str(trained_model_volume_pvc_existing) + " --snapshot-name=" + str(volume_snapshot_name) + " --namespace={{workflow.namespace}}"],
        file_outputs={"volume_snapshot_name": "/volume_snapshot_name.txt"}
    )
    # State that snapshot should be created after the training job completes
    model_snapshot.after(train)

    # Execute inference validation job
    inference_validation = dsl.ContainerOp(
        name="validate-model",
        image="python:3",
        command=["sh", "-c"],
        arguments=["\
            python3 -m pip install pandas matplotlib && \
            git clone https://github.com/yshimizu37/image_classification.git && \
            chmod -R 755 image_classification &&  \
            python3 ./image_classification/validation.py --modeldir /mnt/model --epochs=" + str(train_step_epochs)],
    )
    # Mount model volume/pvc
    inference_validation.apply(
        onprem.mount_pvc(trained_model_volume_pvc_existing, 'modelvol', '/mnt/model')
    )
    # Request that GPUs be allocated to pod
    if validation_step_num_gpu > 0 :
        inference_validation.set_gpu_limit(validation_step_num_gpu, 'nvidia')
    # State that inference validation job should be executed after model volume snapshot is taken
    inference_validation.after(model_snapshot)

if __name__ == "__main__" :
    import kfp.compiler as compiler
    compiler.Compiler().compile(ai_training_run, __file__ + ".yaml")
