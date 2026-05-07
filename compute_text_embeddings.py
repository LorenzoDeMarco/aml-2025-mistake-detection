import os
import json
import torch
import numpy as np
from tqdm import tqdm
import sys
import pathlib

from transformers import AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_egovlp_text_model():
    """Loads the EgoVLP architecture and the associated tokenizer."""
    sys.path.append("EgoVLP") 
    from model.model import FrozenInTime
    
    config_path = os.path.join("EgoVLP", "configs", "pt", "egoclip.json")
    with open(config_path, "r") as f:
        config = json.load(f)
        
    model = FrozenInTime(**config['arch']['args'])
    
    print("Loading EgoVLP weights...")
    weights_path = "feature_extraction/egovlp.pth"
        
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()
    model.to(device)
    
    # EgoVLP uses DistilBERT as the default tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    return model, tokenizer

def extract_text_features(model, tokenizer, texts):
    """Passes a list of strings through the EgoVLP text encoder."""
    # Tokenize the sentences (adding padding and truncation)
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
    
    # EgoVLP expects a dictionary with 'input_ids' and 'attention_mask'
    text_data = {
        'text': inputs['input_ids'].to(device),
        'mask': inputs['attention_mask'].to(device)
    }
    
    with torch.no_grad():
        # Extracts the textual embedding (typically dimension 256 in the text projection)
        text_features = model.compute_text(text_data)
        
    return text_features.cpu().numpy()

def main():
    model, tokenizer = load_egovlp_text_model()
    
    # Read the annotation files provided by the Captain Cook4D dataset
    annotations_file = "annotations/annotation_json/step_annotations.json" 
    output_path = "data/embeddings/text_task_graphs.npz"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
        
    extracted_graphs = {}
    
    print("Extracting textual embeddings...")
    # Iterate over all available recipes
    for recording_id, data in tqdm(annotations.items()):
        # Extract the textual description of each step
        step_descriptions = [step['description'] for step in data['steps'] if 'description' in step]
        
        if not step_descriptions:
            continue
            
        # Generate features for the entire Task Graph (Num_Nodes, 256)
        text_feats = extract_text_features(model, tokenizer, step_descriptions)
        extracted_graphs[recording_id] = text_feats
        
    # Save the NumPy dictionary to a compressed file
    np.savez(output_path, **extracted_graphs)
    print(f"Extraction completed! File saved in {output_path}")

if __name__ == "__main__":
    main()