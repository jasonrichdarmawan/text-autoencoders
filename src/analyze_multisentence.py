# %%
import torch
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize

# Load SONAR models
def load_sonar_models():
    from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading SONAR models on {device}...")

    text2vec = TextToEmbeddingModelPipeline(
        encoder="text_sonar_basic_encoder",
        tokenizer="text_sonar_basic_encoder",
        device=device
    )
    print("SONAR text-to-embedding model loaded successfully.")
    return text2vec, device

def get_embeddings(text2vec, texts, batch_size=32):
    """Generates embeddings for a list of texts with batching."""
    all_embeddings_np = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings_tensor = text2vec.predict(batch_texts, "eng_Latn")
            batch_embeddings_np = [emb.cpu().numpy() for emb in batch_embeddings_tensor]
            all_embeddings_np.extend(batch_embeddings_np)
            print(f"\rProcessing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}", end="")
    print("\nEmbedding generation complete.")
    return np.vstack(all_embeddings_np)

def objective_scalar(params, v1, v2, v12, embedding_dim):
    """Calculates sum of squared errors for: v12[i] ≈ C + a1*v1[i] + a2*v2[i]"""
    if len(params) != 2 + embedding_dim:
        raise ValueError(f"Expected {2 + embedding_dim} params, got {len(params)}")

    a1, a2 = params[0], params[1]
    C = params[2:]

    v12_pred = C.reshape(1, -1) + a1 * v1 + a2 * v2
    return np.sum((v12 - v12_pred)**2)

def objective_scalar_grad(params, v1, v2, v12, embedding_dim):
    """Calculates gradient of objective_scalar function."""
    if len(params) != 2 + embedding_dim:
        raise ValueError(f"Expected {2 + embedding_dim} params, got {len(params)}")

    a1, a2 = params[0], params[1]
    C = params[2:]

    v12_pred = C.reshape(1, -1) + a1 * v1 + a2 * v2
    delta = v12_pred - v12

    grad_a1 = 2 * np.sum(delta * v1)
    grad_a2 = 2 * np.sum(delta * v2)
    grad_C = 2 * np.sum(delta, axis=0)

    return np.concatenate(([grad_a1, grad_a2], grad_C))

def load_decoder_model(device):
    """Load SONAR decoder model."""
    from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline
    decoder = EmbeddingToTextModelPipeline(
        decoder="text_sonar_basic_decoder",
        tokenizer="text_sonar_basic_decoder",
        device=device
    )
    return decoder

def main():
    # Load models
    text2vec, device = load_sonar_models()

    # Define sentence pairs
    sentence_pairs = [
        ("The cat sat on the mat", "The dog chased the ball"),
        ("Apples are red", "Bananas are yellow"),
        ("The sun rises in the east", "The moon orbits the Earth"),
        ("Water is essential for life", "Plants need sunlight to grow"),
        ("He reads books every day", "She enjoys listening to music"),
        ("The train arrived late", "The passengers were annoyed"),
        ("Coding can be challenging", "It is also very rewarding"),
        ("Birds fly in the sky", "Fish swim in the sea"),
        ("Paris is the capital of France", "Berlin is the capital of Germany"),
        ("Machine learning requires data", "Deep learning uses neural networks"),
    ]

    print(f"\n--- Multi-Sentence Scalar Analysis ({len(sentence_pairs)} pairs) ---")

    # Prepare text lists
    s1_list = [pair[0] for pair in sentence_pairs]
    s2_list = [pair[1] for pair in sentence_pairs]
    s1s2_list = [f"{pair[0]}. {pair[1]}" for pair in sentence_pairs]

    # Generate embeddings
    all_texts = list(set(s1_list + s2_list + s1s2_list))
    print(f"Generating embeddings for {len(all_texts)} unique texts...")

    all_embeddings = get_embeddings(text2vec, all_texts)
    text_to_embedding = {text: emb for text, emb in zip(all_texts, all_embeddings)}

    v1_embeds = np.array([text_to_embedding[text] for text in s1_list])
    v2_embeds = np.array([text_to_embedding[text] for text in s2_list])
    v12_embeds = np.array([text_to_embedding[text] for text in s1s2_list])

    num_pairs, embedding_dim = v1_embeds.shape
    print(f"Embedding shapes: v1{v1_embeds.shape}, v2{v2_embeds.shape}, v12{v12_embeds.shape}")

    # Optimize parameters
    print("\nOptimizing parameters (a1, a2, C)...")
    initial_a1, initial_a2 = 1.0, 1.0
    initial_C = np.mean(v12_embeds - initial_a1 * v1_embeds - initial_a2 * v2_embeds, axis=0)
    initial_params = np.concatenate(([initial_a1, initial_a2], initial_C))

    result = minimize(
        objective_scalar,
        initial_params,
        args=(v1_embeds, v2_embeds, v12_embeds, embedding_dim),
        method='L-BFGS-B',
        jac=objective_scalar_grad,
        options={'disp': True, 'maxiter': 1000, 'ftol': 1e-9, 'gtol': 1e-7}
    )

    # Evaluate results
    if not result.success:
        print(f"Optimization failed: {result.message}")
        return

    final_a1, final_a2 = result.x[0], result.x[1]
    final_C = result.x[2:]
    min_error = result.fun

    print(f"\nOptimization successful!")
    print(f"Fitted a1: {final_a1:.4f}")
    print(f"Fitted a2: {final_a2:.4f}")
    print(f"Fitted C norm: {np.linalg.norm(final_C):.4f}")

    # Calculate metrics
    v12_pred_scalar = final_C.reshape(1, -1) + final_a1 * v1_embeds + final_a2 * v2_embeds
    mse_per_sample = min_error / num_pairs
    mse_per_dim = min_error / (num_pairs * embedding_dim)

    mean_v12 = np.mean(v12_embeds, axis=0)
    ss_total = np.sum((v12_embeds - mean_v12)**2)
    r2_score = 1 - (min_error / ss_total) if ss_total > 1e-9 else 0

    print(f"MSE per sample: {mse_per_sample:.6f}")
    print(f"MSE per dimension: {mse_per_dim:.8f}")
    print(f"R-squared: {r2_score:.4f}")

    # Decode embeddings for comparison
    print("\n--- Decoding Comparison ---")
    decoder = load_decoder_model(device)

    num_decode = min(5, len(sentence_pairs))
    v12_orig_tensor = torch.tensor(v12_embeds[:num_decode], dtype=torch.float32, device=device)
    v12_pred_tensor = torch.tensor(v12_pred_scalar[:num_decode], dtype=torch.float32, device=device)

    with torch.no_grad():
        decoded_orig = decoder.predict(v12_orig_tensor, "eng_Latn")
        decoded_pred = decoder.predict(v12_pred_tensor, "eng_Latn")

    print(f"{'Original':<60} | {'Reconstructed':<60}")
    print("-" * 125)
    for i in range(num_decode):
        print(f"{decoded_orig[i]:<60} | {decoded_pred[i]:<60}")

    print(f"\nAnalysis complete. R² = {r2_score:.3f} indicates model fit quality.")

if __name__ == "__main__":
    main()

# %% [markdown]
# Original                                                     | Reconstructed
# -----------------------------------------------------------------------------------------------------------------------------
# The cat sat on the mat. The dog chased the ball              | The dog chased the cat. The cat chased the ball
# Apples are red. Bananas are yellow                           | Apples are blue. Yellow are red.
# The sun rises in the east, the moon orbits the Earth         | The sun rises in the east. The sun rises in the east.
# Water is essential for life. Plants need sunlight to grow    | Plants need water for life. Sunlight is essential for life
# He reads books every day. She likes to listen to music       | He enjoys reading books. Music.

# %%
