import fasttext
import numpy as np

model = fasttext.load_model('C:/Users/shiva/OneDrive/Desktop/clustering dataset/cc.en.300.bin')

def cosine_similarity(word1, word2, model):
    """Calculate cosine similarity between two words"""
    vec1 = model.get_word_vector(word1)
    vec2 = model.get_word_vector(word2)
    
    # Cosine similarity formula
    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    print(f"Similarity between '{word1}' and '{word2}': {similarity:.3f}")
    return similarity

def similarity_drop(fluency_list):
    """
    Input: A fluency list (e.g., ['dog', 'cat', 'parrot', 'eagle'])
    Output: Array of 0s and 1s (size n-1) where 1 = switch
    
    Example: ['dog', 'cat', 'parrot', 'eagle'] 
    Returns: [0, 1, 0] 
    (no switch after dog, switch after cat, no switch after parrot)
    """
    n = len(fluency_list)
    switches = np.zeros(n - 1, dtype=int)  # Array of size n-1

    for i in range(1, n - 2):
        # Get the three similarities we need
        sim_before = cosine_similarity(fluency_list[i-1], fluency_list[i], model)  # A to B
        sim_current = cosine_similarity(fluency_list[i], fluency_list[i+1], model)  # B to C
        sim_after = cosine_similarity(fluency_list[i+1], fluency_list[i+2], model)  # C to D
        
        # Check if it's a drop then rise
        if sim_before > sim_current and sim_current < sim_after:
            switches[i] = 1  # Mark transition after item i as a switch

    print(f"\n Final switches array: {switches}")
    
    return switches

def delta_similarity(fluency_list, rise_threshold=0, fall_threshold=0):
    """
    Input: A fluency list and thresholds
    Output: Array of 0s and 1s (size n-1) where 1 = switch
    
    This one is more complex - it uses z-scores and thresholds
    """
    n = len(fluency_list)
    switches = np.zeros(n - 1, dtype=int)
    
    all_sims = []
    for i in range(n - 1):
        sim = cosine_similarity(fluency_list[i], fluency_list[i+1], model)
        all_sims.append(sim)
    
    median_sim = np.median(all_sims)
    if all_sims[0] < median_sim:
        switches[0] = 1
    
    mean_sim = np.mean(all_sims)
    std_sim = np.std(all_sims)
    z_scores = [(sim - mean_sim) / std_sim for sim in all_sims]
    print(f"Z-scores: {[f'{z:.3f}' for z in z_scores]}")
    
    return switches

if __name__ == "__main__":
    # Test the function
    test_list = ['eagle','dog', 'cat', 'ant', 'spider']
    result = similarity_drop(test_list)
    print(f"\n Final Result: {result}")
    