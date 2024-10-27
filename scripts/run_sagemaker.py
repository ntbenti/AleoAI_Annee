import sagemaker
from sagemaker.huggingface import HuggingFace
from sagemaker import image_uris
import os

def main():
    # Replace with your IAM role ARN
    role = 'arn:aws:iam::992382619842:role/AleoAI'

    # Hyperparameters for the training
    hyperparameters = {
        'num_train_epochs': 3,
        'per_device_train_batch_size': 1,
        'gradient_accumulation_steps': 8,
        'fp16': True,
        'learning_rate': 5e-5,
    }

    # Specify the AWS region
    os.environ['AWS_DEFAULT_REGION'] = 'eu-central-1'  # Replace with your AWS region

    # Initialize the SageMaker session
    sess = sagemaker.Session(default_bucket='aleoai-training-bucket')

    # Upload your training data to S3
    s3_train_data = sess.upload_data(
        path='data/processed/training_data.txt',
        bucket='aleoai-training-bucket',
        key_prefix='codellama-training/data'
    )

    # Retrieve the image URI
    image_uri = image_uris.retrieve(
        framework='huggingface',
        region='eu-central-1',
        version='4.33.0',  # Transformers version
        base_framework_version='2.0.1',  # PyTorch version
        py_version='py39',
        instance_type='ml.g5.xlarge',
        image_scope='training',
    )

    # Define the Hugging Face estimator
    huggingface_estimator = HuggingFace(
        entry_point='train_codellama.py',
        source_dir='scripts',
        role=role,
        image_uri=image_uri,
        instance_count=1,
        instance_type='ml.g5.xlarge',
        hyperparameters=hyperparameters,
        use_spot_instances=True,
        max_wait=7200,
        max_run=7200,
        checkpoint_s3_uri='s3://aleoai-training-bucket/checkpoints/',
        checkpoint_local_path='/opt/ml/checkpoints',
        environment={
            'HUGGING_FACE_HUB_TOKEN': 'your_hf_token_here'  # Replace with your actual token
        },
    )

    # Start the training job
    huggingface_estimator.fit({'train': s3_train_data})

if __name__ == "__main__":
    main()