# Proof-of-Concept: Autograd under Homomorphic Encryption (CKKS)

An experimental research repository focused on **Privacy-Preserving Machine Learning (PPML)**. 

While my primary architecture focuses on high-speed edge **inference** using custom GPU kernels and the BFV scheme, this repository explores the computational complexities of encrypted **model training**. It serves as a proof-of-concept that gradient descent and backpropagation are mathematically viable on fully encrypted ciphertext.

## Research Architecture & Constraints
Training a model under Fully Homomorphic Encryption (FHE) requires bypassing several strict cryptographic limitations:

1. **Floating-Point Arithmetic:** Transitioned from the BFV scheme (exact integers) to the **CKKS scheme** to support the decimal mathematics required for gradient calculus.
2. **Non-Linear Activations:** Because FHE cannot execute conditional logic (making standard ReLU or Sigmoid functions impossible), this engine evaluates a **3rd-degree Taylor Polynomial approximation** to execute the forward pass.
3. **Multiplicative Depth Exhaustion:** Architected a 360-bit deep modulus chain (N=16384) with automated relinearization and rescaling to actively manage cryptographic noise during continuous ciphertext multiplication.

## The Mathematical Proof (Ground Truth Testing)
To mathematically validate the zero-trust architecture, this repository includes a dual-execution test:
* `verify_math.py`: Executes standard, unencrypted backpropagation using NumPy.
* `train_ckks.py`: Executes fully encrypted backpropagation using TenSEAL.

**Results:** The blindfolded CKKS engine successfully calculates the gradient trajectory, matching the plaintext unencrypted math down to the 5th decimal place (the microscopic variance is due to standard CKKS cryptographic noise). 

## How to Run the Proof
```bash
# 1. Install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Run the unencrypted Answer Key
python3 verify_math.py

# 3. Run the Encrypted Engine
python3 train_ckks.py
