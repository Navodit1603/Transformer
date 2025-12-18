from data import GPTTokenizedData
from model import Transformer
from train import train_model
from evaluation import perplexity
from model import get_best_model_definition
from util import count_trainable_parameters

import torch



def main():
    # --- 1. Setup ---
    # Using mps when training on mac and then cuda for nvdia gpu
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # --- 2. Get Data ---
    tokenized = GPTTokenizedData()
    dataloaders = tokenized.dataloaders 
    vocab_size = tokenized.vocab_size    

    # --- 3. Instantiate Model ---
    print("Instantiating model...")
    model = get_best_model_definition(vocab_size)
    model.to(device)
    
    print("Number of trainable parameters: ", count_trainable_parameters(model)) # Check to see if not exceeding the 50 million parameter limit
    # --- 4. Train Model ---
    print("Starting training...")
    # Pass the device and the full dataloaders dict
    train_model(model, dataloaders, device, num_epochs=15) 
    print("Training finished.")

    
    
    # --- 5. Evaluate Best Model ---
    print("Loading best model for evaluation...")
    # Instantiate a fresh model skeleton
    best_model = get_best_model_definition(vocab_size)
    # Load the saved weights
    best_model.load_state_dict(torch.load('best_model.pt', map_location=device))
    best_model.to(device)
    best_model.eval() # Set model to evaluation mode

    print("\n--- Perplexity Report ---")

    # --- 6. Report All Perplexities ---
    
    # Evaluate on Train, Validation and Test set

    # Evaluate on Train set
    train_ppl = perplexity(best_model, dataloaders['train'])
    print(f"Train Perplexity: {train_ppl}")
    
    # Evaluate on Validation set
    val_ppl = perplexity(best_model, dataloaders['val'])
    print(f"Validation Perplexity: {val_ppl}")

    # Evaluate on Test set
    test_ppl = perplexity(best_model, dataloaders['test'])
    print(f"Test Perplexity (Final): {test_ppl}")

    print("-------------------------\n")


if __name__ == "__main__":
    main()
