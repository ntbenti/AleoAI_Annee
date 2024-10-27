import sagemaker
from sagemaker.huggingface import HuggingFace
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

    # Define the Hugging Face estimator
    huggingface_estimator = HuggingFace(
        entry_point='train_codellama.py',
        source_dir='scripts',
        instance_type='ml.g5.xlarge',  # Updated instance type
        instance_count=1,
        role=role,
        transformers_version='4.31',
        pytorch_version='2.0.0',
        py_version='py39',
        hyperparameters=hyperparameters,
        use_spot_instances=True,  # Enable spot instances
        max_wait=7200,            # Maximum wait time in seconds (adjust as needed)
        max_run=7200,             # Maximum run time in seconds (adjust as needed)
        checkpoint_s3_uri='s3://aleoai-training-bucket/checkpoints/',  # Replace with your S3 bucket
        checkpoint_local_path='/opt/ml/checkpoints',             # Local path for checkpoints
    )

    # Start the training job
    huggingface_estimator.fit({'train': s3_train_data})

if __name__ == "__main__":
    main()