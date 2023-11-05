from torch.utils.data import DataLoader, random_split
from losses.custom_loss import MDN_loss_function
from models.lstm_model import Condition_LSTM
from data.stroke_dataset import Stroke_dataset
import torch
import yaml
from transformers import Trainer, TrainingArguments


def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def main():
    # Load configuration
    config = load_config()
    training_args = TrainingArguments(**config['training'])

    # Prepare dataset
    stroke_data = Stroke_dataset(train=True)
    dataset_size = len(stroke_data)
    train_size = int(config['dataset']['train_split'] * dataset_size)
    val_size = int(config['dataset']['val_split'] * dataset_size)
    train_dataset, val_dataset, _ = random_split(
        stroke_data, [train_size, val_size, dataset_size - train_size - val_size])

    # Prepare model
    model = Condition_LSTM(**config['model_params']).to(config['device'])

    # Instantiate the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=None
    )

    # Training
    trainer.train()

    # Evaluation
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")

    # Saving the final model & tokenizer
    trainer.save_model(training_args.output_dir)
    print(f"Model saved to {training_args.output_dir}")


if __name__ == '__main__':
    main()
