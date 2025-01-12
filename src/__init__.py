import torch 
import numpy as np
from typing import List, Tuple
from src.models.multilingual_hypergraph_transformer import MultilingualHypergraphTransformer 
from src.data.data import load_data 

model: MultilingualHypergraphTransformer = MultilingualHypergraphTransformer() 

training_texts: List[str]
training_labels: List[int]  
(training_texts, training_labels) = load_data()  
training_labels_tensor: torch.Tensor = torch.tensor(training_labels) 

model.train_model(training_texts, training_labels_tensor) 

def classify(texts: List[str]) -> np.ndarray: 
    new_texts: List[str] = texts * (32 // len(texts)) 
    new_texts += new_texts[:32 % len(new_texts)] 

    predictions: torch.Tensor = model.predict(new_texts) 
    return predictions.numpy()
