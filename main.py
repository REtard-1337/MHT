from src import classify 
from pyfiglet import figlet_format 
from os import system 
from typing import List

LABELS_NAMES: List[str] = ['positive', 'negative', 'neutral'] 

system('cls') 

print( 
    figlet_format('Multilingual Hypergraph Transformer', font='slant') 
) 

def to_array(string: str) -> List[str]: 
    return [string] 

while True: 
    text: str = input("Enter text to classify: ") 
    classified: List[int] = classify(to_array(text)) 
     
    print( 
        f"\nText {text} classified as: {LABELS_NAMES[classified[0]]}" 
    )
