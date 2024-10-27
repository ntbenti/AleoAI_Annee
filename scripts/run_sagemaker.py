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
    os.environ['AWS_DEFAULT_REGION'] = 'eu-central-1'  # e.g., 'us-east-1'

    # Initialize the SageMaker session
    sess = sagemaker.Session()

    # Upload your training data to S3
    s3_train_data = sess.upload_data(
        path='data/processed/training_data.txt',
        key_prefix='codellama-training/data'
    )

    # Define the Hugging Face estimator
    huggingface_estimator = HuggingFace(
        entry_point='train_codellama.py',
        source_dir='scripts',
        instance_type='ml.p3.2xlarge',  # Adjust based on your needs
        instance_count=1,
        role=role,
        transformers_version='4.26',
        pytorch_version='1.13',
        py_version='py39',
        hyperparameters=hyperparameters,
    )

    # Start the training job
    huggingface_estimator.fit({'train': s3_train_data})

if __name__ == "__main__":
    main()