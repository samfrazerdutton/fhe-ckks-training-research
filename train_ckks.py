import tenseal as ts
import numpy as np

print("\n--- FHE RESEARCH: CKKS ENCRYPTED TRAINING ---")

# 1. Initialize the CKKS Context 
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=16384,
    # THE FIX: Expanded to a massive 360-bit depth chain
    # [60, 40, 40, 40, 40, 40, 40, 60] gives us 6 full multiplications of runway
    coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 40, 40, 60] 
)
context.global_scale = 2**40
context.generate_galois_keys()
context.generate_relin_keys()

context.auto_relin = True
context.auto_rescale = True
context.auto_mod_switch = True

print("[System] CKKS Context initialized (N=16384, Scale=2^40, Depth=8)")

# 2. Create Synthetic Data
x_train = [0.85, -0.32] 
y_true = [1.0]
weights = [0.15, 0.50]

print(f"Original Input (x):  {x_train}")
print(f"Original Weights (w): {weights}")

# 3. Encrypt
enc_x = ts.ckks_vector(context, x_train)
enc_w = ts.ckks_vector(context, weights)
print("[Encrypt] Input and Weights successfully encrypted into CKKS vectors.")

# 4. Forward Pass (Dot Product)
enc_z = enc_x.dot(enc_w)

# 5. Activation (Polynomial Approximation)
print("[Compute] Applying Degree-3 Polynomial Approximation of Sigmoid...")
enc_pred = enc_z.polyval([0.5, 0.197, 0.0, -0.004])

# 6. Backprop / Gradient Calculation
enc_error = enc_pred - y_true
enc_grad = enc_x * enc_error
print("[Compute] Encrypted Gradient (dW) calculated.")

# 7. Weight Update
learning_rate = 0.1
enc_w_new = enc_w - (enc_grad * learning_rate)

# 8. Decrypt
dec_grad = enc_grad.decrypt()
dec_w_new = enc_w_new.decrypt()

print("\n--- RESULTS ---")
print(f"Decrypted Gradient: {dec_grad}")
print(f"Updated Weights:    {dec_w_new}")
