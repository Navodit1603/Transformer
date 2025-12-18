import torch

"""
TODO write your training loop here.
Things to take care with:
    - make sure to use the correct loss function for the task
    - make sure that the targets are correct (each token should predict the next token in the sequence)
    - there should be no loss for padding tokens.
"""

def train_model(model, dataloader, device, num_epochs=15, save_path='best_model.pt'):
    
    model.to(device)

    model.train()
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch in dataloader['train']:
            device = next(model.parameters()).device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            inputs = input_ids[:, :-1]
            # Targets are all but the first token
            targets = input_ids[:, 1:]
            
            # Input mask is all but the last token
            input_mask = attention_mask[:, :-1]
            # Target mask is all but the first token
            target_mask = attention_mask[:, 1:]

            targets = targets.masked_fill(target_mask == 0, -100)

            # --- 4. Forward and Backward Pass ---
            optimizer.zero_grad()

            # Pass the correct inputs and input_mask
            logits = model(inputs, input_mask) 

            # Reshape for CrossEntropyLoss (B, S, V) -> (B*S, V)
            logits_flat = logits.view(-1, logits.size(-1))
            
            # Reshape targets (B, S) -> (B*S)
            targets_flat = targets.view(-1)
            
            loss = loss_fn(logits_flat, targets_flat)

            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()


        avg_train_loss = epoch_loss / len(dataloader['train'])

        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad(): # No gradients needed for validation
            for batch in dataloader['val']:
                
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                inputs = input_ids[:, :-1]
                targets = input_ids[:, 1:]
                input_mask = attention_mask[:, :-1]
                target_mask = attention_mask[:, 1:]
                
                # Also mask targets for validation loss calculation
                targets = targets.masked_fill(target_mask == 0, -100)

                logits = model(inputs, input_mask)
                
                logits_flat = logits.view(-1, logits.size(-1))
                targets_flat = targets.view(-1)
                
                loss = loss_fn(logits_flat, targets_flat)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(dataloader['val'])

        print(f"Epoch {epoch+1:02}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"    -> New best model saved to {save_path}")

    print("--- Training Finished ---")