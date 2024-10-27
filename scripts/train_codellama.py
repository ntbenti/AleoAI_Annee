import os
import torch
import subprocess
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training

# Install required packages
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("transformers==4.33.0")

def main():
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load the tokenizer and model
    model_name = "codellama/CodeLlama-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,  # Use 8-bit precision
        device_map='auto'   # Automatically map layers to devices
    )

    # Load your dataset
    data_files = {'train': 'data/processed/training_data.txt'}
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
        output_dir='/opt/ml/checkpoints',  # Changed output directory for checkpointing
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        save_steps=500,                # Save checkpoint every 500 steps
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
    trainer.train(resume_from_checkpoint=True)  # Enable resuming from checkpoint

    # Save the fine-tuned model
    trainer.save_model('/opt/ml/model')  # Save the model to the designated Sagemaker model directory

if __name__ == "__main__":
    main()