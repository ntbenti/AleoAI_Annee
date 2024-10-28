import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

def main():
    # Get Hugging Face token from environment variable
    hf_token = os.environ.get('HUGGING_FACE_HUB_TOKEN')

    model_name = "codellama/CodeLlama-7b-hf"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_auth_token=hf_token,
        trust_remote_code=True  # Added this line
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        use_auth_token=hf_token,
        load_in_8bit=True,
        device_map='auto',
        trust_remote_code=True  # Added this line
    )


    # Load your dataset
    data_files = {'train': '/opt/ml/input/data/train/training_data.txt'}
    dataset = load_dataset('text', data_files=data_files)

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=512)

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=['text']
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    # Retrieve hyperparameters from environment variables
    num_train_epochs = int(os.environ.get('SM_HP_NUM_TRAIN_EPOCHS', '3'))
    per_device_train_batch_size = int(os.environ.get('SM_HP_PER_DEVICE_TRAIN_BATCH_SIZE', '1'))
    gradient_accumulation_steps = int(os.environ.get('SM_HP_GRADIENT_ACCUMULATION_STEPS', '8'))
    learning_rate = float(os.environ.get('SM_HP_LEARNING_RATE', '5e-5'))
    fp16 = os.environ.get('SM_HP_FP16', 'True') == 'True'

    # Training arguments
    training_args = TrainingArguments(
        output_dir='/opt/ml/model',
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        fp16=fp16,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        report_to='none',
        save_strategy='steps',
        push_to_hub=False,
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        data_collator=data_collator,
    )

    # Start training
    trainer.train()

    # Save the fine-tuned model
    trainer.save_model('/opt/ml/model')

if __name__ == "__main__":
    main()