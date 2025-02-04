import torch
import argparse
import model_utils
import process_image
import json

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Predict flower class from an image")
    parser.add_argument('image_path', type=str, help="Path to image file")
    parser.add_argument('checkpoint', type=str, help="Path to model checkpoint")
    parser.add_argument('--top_k', type=int, default=5, help="Number of top predictions to return")
    parser.add_argument('--category_names', type=str, default=None, help="Path to category names JSON file")
    parser.add_argument('--gpu', action='store_true', help="Use GPU for inference")

    args = parser.parse_args()

    # Load model
    model = model_utils.load_checkpoint(args.checkpoint)

    # Process image
    image = process_image.process_image(args.image_path)

    # Predict
    probs, classes = model_utils.predict(image, model, args.top_k, args.gpu)

    # Modify class indices to real names using `cat_to_name.json`
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        classes = [cat_to_name[str(cls)] for cls in classes]

    print(f"Top-{args.top_k} Predictions:")
    for i in range(len(probs)):
        print(f"{classes[i]}: {probs[i]:.3f}")

if __name__ == '__main__':
    main()
