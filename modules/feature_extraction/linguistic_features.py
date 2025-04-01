"""
linguistic_features.py
----------------------
This script extracts linguistic features from transcript text.
Features include:
- Lexical Diversity (Unique words / Total words)
- Readability Score (Flesch-Kincaid Grade Level)
- Hesitation Markers ("um", "uh" count)

Author: Lavan Aditya
"""

import spacy
import textstat

# Load NLP model
nlp = spacy.load("en_core_web_sm")

def extract_linguistic_features(transcript_text):
    """
    Extracts key linguistic features from a given transcript.
    
    Parameters:
    -----------
    transcript_text : str
        The transcript text from a .cha file
    
    Returns:
    --------
    dict
        Dictionary containing:
        - Lexical Diversity
        - Readability Score
        - Hesitation Markers Count
    """
    
    doc = nlp(transcript_text)
    
    # Compute lexical diversity (Ratio of unique words to total words)
    unique_words = len(set([token.text.lower() for token in doc if token.is_alpha]))
    total_words = len([token.text for token in doc if token.is_alpha])
    lexical_diversity = unique_words / total_words if total_words > 0 else 0
    
    # Compute Readability Score (Lower = More difficult to read)
    readability = textstat.flesch_kincaid_grade(transcript_text)
    
    # Count Hesitation Markers ("um", "uh")
    hesitation_count = transcript_text.lower().count("um") + transcript_text.lower().count("uh")
    
    return {
        "Lexical Diversity": lexical_diversity,
        "Readability Score": readability,
        "Hesitation Markers": hesitation_count
    }
