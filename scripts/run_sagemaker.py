import os
import logging
import sagemaker
from sagemaker.huggingface import HuggingFace

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Replace with your IAM role ARN
    role = 'arn:aws:iam::992382619842:role/AleoAI'

    # Hyperparameters for the training
    hyperparameters = {
        'num_train_epochs': 3,
        'per_device_train_batch_size': 1,  # Keep batch size small to reduce memory usage
        'gradient_accumulation_steps': 8,
        'fp16': True,
        'learning_rate': 5e-4,
        'gradient_checkpointing': True,  # Enable gradient checkpointing
    }

    # Specify the AWS region
    os.environ['AWS_DEFAULT_REGION'] = 'eu-central-1'  # Replace with your AWS region

    # Initialize the SageMaker session
    sess = sagemaker.Session(default_bucket='aleoai-training-bucket')
    logger.info('Initialized SageMaker session.')

    # Upload your training data to S3
    logger.info('Uploading training data to S3...')
    s3_train_data = sess.upload_data(
        path='data/processed/training_data.txt',
        bucket='aleoai-training-bucket',
        key_prefix='codellama-training/data'
    )
    logger.info(f'Training data uploaded to {s3_train_data}')

    # Define the Hugging Face estimator with supported versions
    huggingface_estimator = HuggingFace(
        entry_point='train_codellama.py',
        source_dir='scripts',
        role=role,
        transformers_version='4.36.0',
        pytorch_version='2.1.0',        # Ensure compatibility
        py_version='py310',
        instance_count=1,
        instance_type='ml.p3.2xlarge',  # Changed instance type
        hyperparameters=hyperparameters,
        use_spot_instances=False,       # Consider setting to False to avoid interruptions
        max_run=7200,
        checkpoint_s3_uri='s3://aleoai-training-bucket/checkpoints/',
        checkpoint_local_path='/opt/ml/checkpoints',
        environment={
            'HUGGING_FACE_HUB_TOKEN': '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'  # Replace with your actual token
        },
    )
    logger.info('Hugging Face estimator initialized.')

    # Start the training job
    logger.info('Starting SageMaker training job...')
    huggingface_estimator.fit({'train': s3_train_data})
    logger.info('Training job initiated.')

if __name__ == "__main__":
    main()