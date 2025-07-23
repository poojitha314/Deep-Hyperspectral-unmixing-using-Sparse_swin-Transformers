import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import os

# List of datasets to visualize
datasets = ['samson', 'apex', 'dc', 'jasperRidge', 'urban']

for name in datasets:
    result_dir = f'trans_mod_{name}'
    abd_path = f'{result_dir}/{name}_abd_map.mat'
    endmem_path = f'{result_dir}/{name}_endmem.mat'

    if not os.path.exists(abd_path) or not os.path.exists(endmem_path):
        print(f"❌ Missing results for {name.upper()} (Check .mat files)")
        continue

    print(f"\n Visualizing SparseSwin predictions for: {name.upper()}")

    # === Load abundance maps ===
    abd_data = sio.loadmat(abd_path)
    A_est = abd_data.get('A_est')
    if A_est is None:
        print("'A_est' not found in", abd_path)
        continue

    H, W, P = A_est.shape

    # Plot and save each abundance map
    for i in range(P):
        plt.figure(figsize=(4, 4))
        plt.imshow(A_est[:, :, i], cmap='viridis')
        plt.title(f"{name.upper()} - Abundance Map {i + 1}")
        plt.axis('off')
        plt.colorbar()
        plt.tight_layout()

        save_path = os.path.join(result_dir, f"{name}_abundance_map_{i+1}.png")
        plt.savefig(save_path, bbox_inches='tight')
        print(f" Saved: {save_path}")
        plt.show()

    # === Load and plot endmembers ===
    E_est = sio.loadmat(endmem_path).get('E_est')
    if E_est is None:
        print("'E_est' not found in", endmem_path)
        continue

    plt.figure(figsize=(8, 5))
    for i in range(P):
        plt.plot(E_est[:, i], label=f'Endmember {i + 1}')
    plt.title(f"{name.upper()} - Predicted Endmembers (SparseSwin)")
    plt.xlabel("Spectral Band Index")
    plt.ylabel("Reflectance")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(result_dir, f"{name}_endmembers.png")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.show()
