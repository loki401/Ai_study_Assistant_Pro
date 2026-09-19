from spellchecker import SpellChecker
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Initialize spellchecker
spell = SpellChecker()

# Initialize lightweight local model for next-word prediction
MODEL_NAME = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()

def correct_text(text: str) -> str:
    """Detects and fixes spelling mistakes in a sentence."""
    if not text.strip():
        return text

    words = text.split()
    corrected_words = []

    for word in words:
        # Strip simple punctuation for checking
        clean_word = word.strip(".,!?;:()[]")
        if clean_word and clean_word.isalpha():
            # Check if misspelled
            if clean_word.lower() not in spell:
                correction = spell.correction(clean_word.lower())
                corrected_words.append(correction if correction else word)
            else:
                corrected_words.append(word)
        else:
            corrected_words.append(word)

    return " ".join(corrected_words)

def predict_next_words(prompt_text: str, top_k: int = 3) -> list:
    """Predicts the most probable next word(s) given the current sentence."""
    if not prompt_text.strip():
        return []

    inputs = tokenizer(prompt_text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]  # Logits of the final token
        top_indices = torch.topk(logits, top_k).indices

    predictions = []
    for idx in top_indices:
        token = tokenizer.decode([idx.item()]).strip()
        if token and token not in predictions:
            predictions.append(token)

    return predictions