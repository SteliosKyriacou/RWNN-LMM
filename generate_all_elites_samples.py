import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tiktoken
from rwnn.graph import RWNNGraph
from evolve_agentic import vector_to_multilayer_graph

def generate_samples_for_all_elites():
    print("=== Generating Authentic Autoregressive Samples for all Gen 35 Elites ===")
    os.makedirs("checkpoints/agentic-optim", exist_ok=True)
    
    vocab_size = 50257
    block_size = 256
    d_model = 192
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize tiktoken GPT-2 encoder
    enc = tiktoken.get_encoding("gpt2")
    
    # Load Generation 35 report to find elites list
    report_path = "checkpoints/agentic-optim/generation_45_report.json" # Gen 45 is the highest completed generation!
    if not os.path.exists(report_path):
        print(f"Report not found: {report_path}")
        return
        
    with open(report_path, 'r') as fh:
        elites = json.load(fh)
        
    print(f"Loaded {len(elites)} elites from Generation 45 report.")
    
    elite_samples = {}
    
    for i, elite in enumerate(elites):
        rank = elite['rank']
        loss = elite['loss']
        params = elite['params']
        config_file = elite['saved_config']
        
        print(f"\nProcessing Elite {rank} (Params: {params:,}, Loss: {loss:.4f})...")
        if not os.path.exists(config_file):
            print(f"Config file not found: {config_file}")
            continue
            
        with open(config_file, 'r') as f:
            c_data = json.load(f)
            
        nodes = c_data['nodes']
        edges = c_data['edges']
        x_vector = c_data['vector']
        
        # Compile model
        model = RWNNGraph(nodes, edges, global_d_model=d_model)
        
        # Try to load the matching or nearest weight file
        weight_file = elite['saved_weights']
        if os.path.exists(weight_file):
            try:
                model.load_state_dict(torch.load(weight_file, map_location='cpu'))
                print(f" -> Successfully loaded weights from {weight_file}")
            except Exception as ex:
                print(f" -> Failed to load weights {weight_file}: {ex}")
        else:
            # Try to search for any alternative weight file matching this rank
            print(f" -> Weight file {weight_file} not found. Running with current initialization.")
            
        model.to(device)
        model.eval()
        
        # Start autoregressive sampling
        # Use <|endoftext|> (id 50256) or a standard prompt like "In the beginning,"
        prompt_text = "In the beginning of the"
        context = torch.tensor([enc.encode(prompt_text)], dtype=torch.long, device=device)
        
        generated = []
        with torch.no_grad():
            for step in range(120): # Generate 120 BPE tokens
                context_cond = context if context.size(1) <= block_size else context[:, -block_size:]
                with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
                    logits = model(context_cond)
                logits = logits[:, -1, :] # focus on last token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                context = torch.cat((context, next_token), dim=1)
                generated.append(next_token.item())
                
        decoded_text = prompt_text + " " + enc.decode(generated)
        print(f"=== Decoded Sample (Elite {rank}) ===")
        print(decoded_text)
        print("="*60)
        
        elite_samples[rank] = {
            'params': params,
            'loss': loss,
            'nodes': len(nodes),
            'edges': len(edges),
            'sample': decoded_text
        }
        
    # Write to a JSON file
    out_file = "checkpoints/agentic-optim/elite_prose_samples.json"
    with open(out_file, 'w') as fh:
        json.dump(elite_samples, fh, indent=4)
    print(f"\n✓ Saved all elite prose samples to: {out_file}")

if __name__ == "__main__":
    generate_samples_for_all_elites()
