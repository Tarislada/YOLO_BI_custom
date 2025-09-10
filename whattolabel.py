"""
Core Pipeline Skeleton Code for Data Selection
This file contains the core classes and functions for the data selection pipeline.
"""

import os
import glob
import shutil
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch 
import torchvision as tv
import torch.nn as nn
from ultralytics import YOLO
from sklearn.metrics.pairwise import pairwise_distances
import sys

from PIL import Image  # or use OpenCV
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN, AgglomerativeClustering, SpectralClustering

# from kneed import KneeLocator

######################################################
# Configuration & Parameter Setup
######################################################

def parse_arguments():
    """
    Parse command-line arguments or configure parameters.
    Return a namespace with relevant attributes such as:
    - input_dir
    - output_dir
    - model_type (e.g., 'yolo_nano', 'resnet')
    - pca_variance_threshold
    - cluster_algorithm
    - k_range (for elbow analysis)
    - num_images (number of images to label)
    """
    parser = argparse.ArgumentParser(description="Data Selection Pipeline")
    parser.add_argument('--input_dir', type=str, default='input_images', help='Path to raw images')
    parser.add_argument('--output_dir', type=str, default='selected_images', help='Path to save selected images')
    parser.add_argument('--model_type', type=str, default='resnet', help='Backbone model type: yolo_nano or resnet')
    parser.add_argument('--pca_variance_threshold', type=float, default=0.95, help='Variance threshold for PCA')
    parser.add_argument('--cluster_algorithm', type=str, default='kcenter', help='Clustering algorithm to use')
    parser.add_argument('--k_min', type=int, default=2, help='Minimum number of clusters for elbow analysis')
    parser.add_argument('--k_max', type=int, default=100, help='Maximum number of clusters for elbow analysis')
    parser.add_argument('--num_images', type=int, default=None, help='Number of images to label (if known)')
    parser.add_argument('--pca_components', type=int, default=None, help='Number of PCA components to keep')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')

    args = parser.parse_args()
    return args

######################################################
# Feature Extraction Module
######################################################

def load_and_preprocess_image(file_path, resize_dims=(224, 224)):
    """
    Load an image from disk and preprocess it for the backbone.
    Placeholder for user-defined logic.
    """
    try:
        # Example with PIL
        img = Image.open(file_path).convert('RGB')
        img = img.resize(resize_dims)
        # Convert to numpy array and normalize if needed
        img_array = np.asarray(img, dtype=np.float32)
        # Example normalization (mean, std placeholder)
        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]
        # img_array = (img_array / 255.0 - mean) / std
        return img_array
    except Exception as e:
        print(f"Error loading image {file_path}: {e}")
        return None


#FIXME: Loading function is not different from other backbones, but the inference must be different
def load_yolo_nano_backbone(existing_backbone=None, weights_path=None):
    """
    Load a YOLO Nano backbone for feature extraction.

    If an existing backbone instance is provided via `existing_backbone`, this function will
    attempt to remove its detection heads (if applicable) and return only the feature extractor.
    If not provided, it will try to load a model from the given `weights_path`.

    Parameters:
        existing_backbone (torch.nn.Module, optional): A pre-loaded YOLO Nano backbone model.
        weights_path (str, optional): Path to the saved model weights for YOLO Nano.

    Returns:
        backbone_model (torch.nn.Module): A YOLO Nano backbone in evaluation mode for feature extraction.
                                           Returns None if loading fails.
    """
    try:
        # backbone_model = ultralytics.YOLO('yolo11n.pt')
        
        # Case 1: An existing backbone instance is provided
        if existing_backbone is not None:
            # Check if the backbone has a separate attribute for feature extraction.
            # This is application-specific; adjust attribute names as needed.
            if hasattr(existing_backbone, 'features'):
                backbone_model = existing_backbone.features
            else:
                # If no explicit features attribute, assume the entire model is the feature extractor.
                backbone_model = existing_backbone

        # Case 2: No existing backbone; attempt to load from weights_path.
        else:
            if weights_path is None:
                raise ValueError("No existing YOLO Nano backbone provided and no weights_path specified.")
            # Load the backbone model from the weights file.
            # Adjust map_location or loading method as needed.
            backbone_model = torch.load(weights_path, map_location='cpu')
            # Optionally remove detection heads if the loaded model contains them.
            if hasattr(backbone_model, 'detection_heads') and hasattr(backbone_model, 'features'):
                backbone_model = backbone_model.features

        # Set the model to evaluation mode to disable dropout, batch norm, etc.
        backbone_model.eval()
        return backbone_model

    except Exception as e:
        print(f"Error loading YOLO Nano backbone: {e}")
        return None


def load_resnet_backbone(existing_backbone=None):
    """
    Load a pre-trained ResNet backbone for feature extraction.
    
    If an existing backbone instance is provided, it will be used;
    otherwise, a new ResNet50 model from torchvision is loaded.
    The final classification layer is removed to obtain a feature extractor.
    
    Parameters:
        existing_backbone (torch.nn.Module, optional): A pre-loaded ResNet model.
        
    Returns:
        backbone_model (torch.nn.Module): A ResNet backbone in evaluation mode for feature extraction.
                                          Returns None if loading fails.
    """
    try:

        if existing_backbone is not None:
            backbone_model = existing_backbone
        else:
            backbone_model = tv.models.resnet50(pretrained=True)
        
        # Remove the final classification layer.
        # In ResNet, the final fc layer is used for classification.
        if hasattr(backbone_model, 'fc'):
            backbone_model.fc = nn.Identity()
        else:
            # Fallback: if the model structure is different, remove the last module.
            modules = list(backbone_model.children())[:-1]
            backbone_model = nn.Sequential(*modules)
            
        backbone_model.eval()
        return backbone_model

    except Exception as e:
        print(f"Error loading ResNet backbone: {e}")
        return None


def load_backbone(model_type, **kwargs):
    """
    Wrapper function that loads the appropriate backbone model based on model_type.
    
    For 'yolo_nano', it calls load_yolo_nano_backbone with additional keyword arguments.
    For 'resnet', it calls load_resnet_backbone with additional keyword arguments.
    
    Parameters:
        model_type (str): The type of backbone to load ('yolo_nano' or 'resnet').
        kwargs: Additional keyword arguments to pass to the respective loading functions 
                (e.g., existing_backbone, weights_path, etc.).
        
    Returns:
        model (torch.nn.Module): The loaded backbone model in evaluation mode, or None if loading fails.
    """
    if model_type == 'yolo_nano':
        print("Loading YOLO Nano backbone...")
        v11n = YOLO('yolo11n.pt')
        model = load_yolo_nano_backbone(existing_backbone=v11n,**kwargs)
    elif model_type == 'resnet':
        print("Loading ResNet backbone...")
        model = load_resnet_backbone(**kwargs)
    else:
        print(f"Unsupported model type: {model_type}")
        model = None

    return model

def extract_feature(image_array, backbone_model):
    """
    Pass the preprocessed image through the chosen backbone.
    Convert the numpy image array to a PyTorch tensor, feed it through the backbone,
    and return a flattened feature vector as a numpy array.

    Parameters:
        image_array (np.array): Preprocessed image array (H, W, C).
        backbone_model (torch.nn.Module): The backbone model in evaluation mode.

    Returns:
        np.array: Flattened feature vector, or None if an error occurs.
    """

    try:
        # Optional: Normalize tensor if required by the backbone.
        # For example:
        # mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        # std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        # tensor = (tensor / 255.0 - mean) / std
        
        if isinstance(backbone_model, YOLO):
            # For YOLO Nano, use the forward method directly.
            # FIXME: the line below prints an empty line in the terminal - disturbing, and should be removed
            # print(backbone_model)
            features = backbone_model.embed(image_array,verbose=False)
            features = features[0]
            if features.dim() > 2:
                features = features.view(features.size(0), -1)
            feature_vector = features.cpu().numpy().squeeze()
            
        else:
            # Convert numpy array (H, W, C) to torch tensor (C, H, W) and add a batch dimension.
            # tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0).float()
            tensor = torch.from_numpy(image_array).permute(2, 0, 1).float()
            # Disable gradient computation for inference.
            with torch.no_grad():
                features = backbone_model(tensor)
            
            # If the output has more than two dimensions (e.g., batch, channels, height, width),
            # flatten it starting from dimension 1 (batch remains the same).
            if features.dim() > 2:
                features = features.view(features.size(0), -1)
            
            # Convert the features tensor to a numpy array and remove the batch dimension.
            feature_vector = features.cpu().numpy().squeeze()
            
        return feature_vector

    except Exception as e:
        print(f"Error extracting feature: {e}")
        return None

def build_feature_database(input_dir, backbone_model, resize_dims=(224, 224)):
    """
    Iterate through images in the input directory, extract features, and store them.
    Return: (list_of_features, list_of_filepaths)
    """
    features = []
    filenames = []

    image_files = glob.glob(os.path.join(input_dir, '*.*'))  # Adjust as needed
    for file_path in image_files:
        img_array = load_and_preprocess_image(file_path, resize_dims=resize_dims)
        if img_array is not None:
            feat = extract_feature(img_array, backbone_model)
            # print(f"Extracted feature for {file_path}")
            if feat is not None:
                features.append(feat)
                filenames.append(file_path)

    features = np.array(features)
    return features, filenames

######################################################
# Dimensionality Reduction Module
######################################################

def apply_pca(features, n_components=None, variance_threshold=0.95):
    """
    Apply PCA to reduce dimensions.
    If n_components is not specified, determine the number of components
    that explain at least variance_threshold of the variance.
    Return (reduced_features, pca_model).
    """
    if features.size == 0:
        return features, None

    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(features)

    if n_components is None:
        # figure out how many components we need
        cum_variance = np.cumsum(pca.explained_variance_ratio_)
        num_components = np.searchsorted(cum_variance, variance_threshold) + 1
        # re-fit with that many components
        pca = PCA(n_components=num_components)
        transformed = pca.fit_transform(features)

    return transformed, pca

######################################################
# Clustering & Diversity Evaluation Module
######################################################

def cluster_features(features, algorithm='kmeans', **kwargs):
    """
    Cluster the feature vectors using the specified algorithm.
    Return the cluster labels and any additional metrics.
    """
    labels = None
    inertia = None

    if algorithm == 'kmeans':
        n_clusters = kwargs.get('n_clusters', 5)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(features)
        labels = kmeans.labels_
        inertia = kmeans.inertia_
        return labels, inertia, kmeans
    
    elif algorithm == 'dbscan':
        eps, mean_smp = kwargs.get('eps', 0.5), kwargs.get('min_samples', 5)
        dbscan = DBSCAN(eps=eps, min_samples=mean_smp)
        labels = dbscan.fit_predict(features)
        return labels, None, None
    elif algorithm == 'agglomerative':
        n_clusters = kwargs.get('n_clusters', 5)
        agg = AgglomerativeClustering(n_clusters=n_clusters)
        labels = agg.fit_predict(features)
        return labels, None, None
    elif algorithm == 'spectral':
        n_clusters = kwargs.get('n_clusters', 5)
        spectral = SpectralClustering(n_clusters=n_clusters)
        labels = spectral.fit_predict(features)
        return labels, None, None
    
    elif algorithm == 'k-center':
        # Placeholder for k-center clustering algorithm
        n_samples = features.shape[0]
        core_set_indices = []
        k = kwargs.get('num_images', 45)
        metric = 'euclidean'
        
        # Initialize with a random index.
        idx = np.random.choice(n_samples)
        core_set_indices.append(idx)
        
        # Compute initial distances from the first center.
        distances = pairwise_distances(features, features[idx].reshape(1, -1), metric=metric).flatten()
        
        # Iteratively add the farthest point to the core set.
        for _ in range(1, k):
            idx = np.argmax(distances)
            core_set_indices.append(idx)
            
            # Update distances: for each point, keep the minimum distance to any selected center.
            new_distances = pairwise_distances(features, features[idx].reshape(1, -1), metric=metric).flatten()
            distances = np.minimum(distances, new_distances)
        return core_set_indices, None, None
    
    elif algorithm == 'torque':
        # Placeholder for torque clustering algorithm
        pass
        return labels

def compute_elbow_curve(features, k_range, clustering_algorithm='kmeans', **kwargs):
    """
    Compute the inertia for each k in k_range.
    Return a list of (k, inertia) pairs.
    """
    elbow_data = []

    if clustering_algorithm == 'kmeans':
        for k in k_range:
        
            _, inertia, _ = cluster_features(features, algorithm='kmeans', n_clusters=k)
            elbow_data.append((k, inertia))
        return elbow_data
    elif clustering_algorithm == 'k-center': 
            # in this case, core_set_indices is chosen directly, so just act as a wrapper function
        core_set_indices, _, _ = cluster_features(features, algorithm='k-center', **kwargs)
        return core_set_indices
    else:          
        # TODO: Support elbow approach for other algorithms if applicable
        pass


def detect_elbow(elbow_data):
    """
    Use a method (like KneeLocator) to find the elbow in the (k, inertia) data.
    Return the suggested optimal k.
    """
    # Example placeholder approach: just return the k for the minimum inertia.
    # In practice, use KneeLocator or a custom method.
    # from kneed import KneeLocator

    # k_values = [item[0] for item in elbow_data]
    # inertias = [item[1] for item in elbow_data]
    # kn = KneeLocator(k_values, inertias, curve='convex', direction='decreasing')
    # optimal_k = kn.knee
    # return optimal_k

    sorted_elbow_data = sorted(elbow_data, key=lambda x: x[1])
    best_k = sorted_elbow_data[0][0] if sorted_elbow_data else None
    return best_k

######################################################
# Optimal Image Selection Module
######################################################

def select_representative_images(features, filenames, num_images, algorithm='kmeans', coreset_idx=None):
    """
    Cluster features into num_images clusters.
    Select the image closest to each cluster centroid.
    Return a list of selected file paths.
    """
    selected_files = []
    if len(features) == 0:
        return selected_files
    if algorithm == 'kmeans':
        # For K-Means, define n_clusters=num_images
        labels, inertia, kmeans_model = cluster_features(
            features, algorithm=algorithm, n_clusters=num_images
        )

        if kmeans_model is None:
            return selected_files

        # Get centroids
        centroids = kmeans_model.cluster_centers_

        # For each cluster, pick the closest image
        for cluster_id in range(num_images):
            # gather indices of this cluster
            cluster_indices = np.where(labels == cluster_id)[0]
            if len(cluster_indices) == 0:
                continue
            # compute distances to centroid
            cluster_features = features[cluster_indices]
            dists = np.linalg.norm(cluster_features - centroids[cluster_id], axis=1)
            closest_index = np.argmin(dists)
            selected_files.append(filenames[cluster_indices[closest_index]])
    elif algorithm == 'k-center':
        # For K-Center, core_set_indices is already the selected indices
        selected_files = [filenames[idx] for idx in coreset_idx]

    return selected_files

######################################################
# Output & Logging Module
######################################################

def save_selection_log(selected_images, log_path="selection_log.csv"):
    """
    Save a CSV log of the selected images.
    """
    df = pd.DataFrame(selected_images, columns=["selected_image"])
    df.to_csv(log_path, index=False)

######################################################
# Main Routine
######################################################

def main():
    # 1. Parse arguments
    args = parse_arguments()

    # 2. Load backbone model
    backbone_model = load_backbone(args.model_type)

    # 3. Build feature database
    features, filenames = build_feature_database(args.input_dir, backbone_model)

    # 4. Apply PCA (optional)
    if args.pca_components is not None:
        features, pca_model = apply_pca(features, n_components=None, variance_threshold=args.pca_variance_threshold)

    k_vals = range(args.k_min, args.k_max + 1)
    # 5. If using K-Means, compute elbow curve
    if args.cluster_algorithm == 'kmeans':
        
        elbow_data = compute_elbow_curve(features, k_vals, clustering_algorithm='kmeans')
        optimal_k = detect_elbow(elbow_data)
        print(f"Elbow curve suggests ~{optimal_k} clusters.")
        # 6. Determine number of images to label
        if args.num_images is not None:
            num_images = args.num_images
        else:
            # fallback to elbow suggestion if user didn't specify
            num_images = optimal_k

    elif args.cluster_algorithm == 'k-center':
        # compute_elbow_curve function works as a wrapper for k-center
        # in this case, so we don't need to do anything here
        coreset_idx = compute_elbow_curve(features, k_vals, clustering_algorithm='k-center', num_images=args.num_images)
        # optimal_k = None
        num_images = len(coreset_idx)
    
    # 7. Select representative images
    selected_images = []
    if num_images:
        selected_images = select_representative_images(
            features,
            filenames,
            num_images=num_images,
            algorithm=args.cluster_algorithm,
            coreset_idx=coreset_idx
        )

    # 8. Copy selected images to output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for img_file in selected_images:
        try:
            shutil.copy(img_file, args.output_dir)
        except Exception as e:
            print(f"Error copying {img_file}: {e}")

    # 9. Save log
    save_selection_log(selected_images, log_path=os.path.join(args.output_dir, "selection_log.csv"))

    # 10. Final summary
    print("========================================")
    print(f"Total images processed: {len(filenames)}")
    print(f"Number of images selected: {len(selected_images)}")
    print("Selected images saved to:", args.output_dir)
    print("Selection log saved as selection_log.csv")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.extend([
            '--input_dir', '/home/tarislada/YOLOprojects/YOLO_custom/Dataset/Walltask/combined',
            '--output_dir', '/home/tarislada/YOLOprojects/YOLO_custom/Dataset/Walltask/represetative',
            '--model_type', 'yolo_nano',
            '--pca_variance_threshold', '0.95',
            '--cluster_algorithm', 'k-center',
            '--k_min', '2',
            '--k_max', '200',
            '--num_images', '100'
        ])

    main()
