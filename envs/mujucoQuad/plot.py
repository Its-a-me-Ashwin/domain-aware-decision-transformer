import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def load_configs_from_directory(base_dir="./dataset"):
    """Recursively load and flatten all config.json files in base_dir."""
    data = []
    for root, dirs, files in os.walk(base_dir):
        if "config.json" in files:
            config_path = os.path.join(root, "config.json")
            with open(config_path, "r") as f:
                config_dict = json.load(f)
                # Flatten into a list; order must match the config_to_vector definition
                row = [
                    config_dict["gravity_strength"],
                    config_dict["gravity_angle_x"],
                    config_dict["gravity_angle_y"],
                    config_dict["max_motor_torque"],
                    config_dict["friction"],
                    config_dict["incline_x"],
                    config_dict["incline_y"],
                ]
                data.append(row)
    return np.array(data)

def plot_pca_2d(data):
    """Perform PCA on the data to reduce it to 2D, then plot it."""
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)

    plt.scatter(reduced_data[:, 0], reduced_data[:, 1])
    # Label axes with explained variance ratio for clarity
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)")
    plt.title("2D PCA of QuadWorldConfig Parameters")
    plt.show()

if __name__ == "__main__":
    data = load_configs_from_directory("./dataset")
    if data.size == 0:
        print("No config.json files found or dataset is empty.")
    else:
        plot_pca_2d(data)
