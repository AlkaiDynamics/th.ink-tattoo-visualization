import os
import sys

def calculate_accuracy(predictions, targets):
    correct = sum(p == t for p, t in zip(predictions, targets))
    return (correct / len(targets)) * 100

def main():
    # Placeholder for loading predictions and targets
    # Replace with actual logic to load model predictions and true labels
    try:
        # Example: Load from files or database
        with open('data/predictions.txt', 'r') as f:
            predictions = [line.strip() for line in f.readlines()]
        
        with open('data/targets.txt', 'r') as f:
            targets = [line.strip() for line in f.readlines()]
        
        if len(predictions) != len(targets):
            print("Mismatch between number of predictions and targets.")
            sys.exit(1)
        
        accuracy = calculate_accuracy(predictions, targets)
        print(f"{accuracy}")  # Output the accuracy as plain text for easy parsing
        
        if accuracy >= 98.0:
            sys.exit(0)  # Indicates success
        else:
            sys.exit(1)  # Indicates failure
    except Exception as e:
        print(f"Error during evaluation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()