import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

def main():
    # Get Hugging Face token from environment variable
    hf_token = os.environ.get('HUGGING_FACE_HUB_TOKEN')

    model_name = "codellama/CodeLlama-7b-hf"

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        use_auth_token=hf_token,
        load_in_8bit=True,
        device_map='auto'
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
        num_proc=4,  # Number of CPU cores
        remove_columns=['text']
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir='/opt/ml/model',
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        learning_rate=5e-5,
        fp16=True,
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