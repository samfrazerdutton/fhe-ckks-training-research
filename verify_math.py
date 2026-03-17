import numpy as np

print("\n--- GROUND TRUTH: PLAINTEXT MATH ---")

# 1. The exact same starting data
x_train = np.array([0.85, -0.32])
y_true = np.array([1.0])
weights = np.array([0.15, 0.50])
learning_rate = 0.1

print(f"Input (x):   {x_train}")
print(f"Weights (w): {weights}")

# 2. Standard Forward Pass (Dot Product)
z = np.dot(x_train, weights)

# 3. The Activation (Using the exact same Taylor Polynomial)
# f(x) = 0.5 + 0.197x - 0.004x^3
pred = 0.5 + (0.197 * z) - (0.004 * (z**3))

# 4. Standard Backprop / Gradient Calculation
error = pred - y_true
grad = x_train * error
print("\n[Compute] Plaintext Gradient (dW) calculated.")

# 5. Standard Weight Update
w_new = weights - (grad * learning_rate)

print("\n--- PLAINTEXT RESULTS ---")
print(f"Gradient:       {grad}")
print(f"Updated Weights: {w_new}")
