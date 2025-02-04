import torch
import argparse
import model_utils
import data_utils

def main():
  # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a deep learning model")
    parser.add_argument('data_dir', type=str, help="Path to dataset directory")
    parser.add_argument('--save_dir', type=str, default='.', help="Directory to save the trained model")
    parser.add_argument('--arch', type=str, default='vgg16', choices=['vgg16', 'resnet18', 'alexnet'], help="Model architecture")
    parser.add_argument('--learning_rate', type=float, default=0.003, help="Learning rate")
    parser.add_argument('--hidden_units', type=int, default=512, help="Number of hidden units")
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs")
    parser.add_argument('--gpu', action='store_true', help="Use GPU for training")

    args = parser.parse_args()

    # Load data
    train_loader, valid_loader, test_loader, class_to_idx = data_utils.load_data(args.data_dir)

    # Load the model
    model, criterion, optimizer = model_utils.create_model(args.arch, args.hidden_units, args.learning_rate)

    # Train the model
    model_utils.train_model(model, train_loader, valid_loader, criterion, optimizer, args.epochs, args.gpu)

    # Save checkpoint
    model_utils.save_checkpoint(model, args.save_dir, args.arch, class_to_idx)

    print("Training complete. Model saved.")

if __name__ == '__main__':
    main()
