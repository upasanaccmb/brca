#This code is used for feature selection using deep learning ResNet50 model and embedded method LASSO



import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from skimage.feature import graycomatrix, graycoprops
import cv2

class HistologyFeatureExtractor:
    def __init__(self, input_shape=(256, 256, 3)):
        self.cnn_model = ResNet50(include_top=False, 
                                weights='imagenet',
                                input_shape=input_shape,
                                pooling='avg')
        self.scaler = StandardScaler()
        
    def extract_features(self, patches):
        features_list = []
        for patch in patches:
            # Deep features
            deep_features = self._get_deep_features(patch)
            
            # Texture features
            texture_features = self._get_texture_features(patch)
            
            # Color features
            color_features = self._get_color_features(patch)
            
            # Combine all features
            combined = np.concatenate([deep_features, texture_features, color_features])
            features_list.append(combined)
            
        return np.array(features_list)
    
    def _get_deep_features(self, patch):
        x = tf.keras.applications.resnet.preprocess_input(patch)
        return self.cnn_model.predict(np.expand_dims(x, axis=0))[0]
    
    def _get_texture_features(self, patch):
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        glcm = graycomatrix(gray, distances=[1], angles=[0, np.pi/4, np.pi/2],
                           levels=256, symmetric=True, normed=True)
        return np.array([
            graycoprops(glcm, prop).ravel()[0]
            for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
        ])
    
    def _get_color_features(self, patch):
        means = np.mean(patch, axis=(0,1))
        stds = np.std(patch, axis=(0,1))
        return np.concatenate([means, stds])

class EmbeddedFeatureSelector:
    def __init__(self, alpha=0.01):
        self.selector = SelectFromModel(Lasso(alpha=alpha))
        
    def fit_transform(self, features, survival_time):
        self.selector.fit(features, survival_time)
        return self.selector.transform(features)
    
    def get_selected_indices(self):
        return np.where(self.selector.get_support())[0]

def run_feature_pipeline(patches, survival_time):
    # Extract features
    extractor = HistologyFeatureExtractor()
    features = extractor.extract_features(patches)
    
    # Select features using LASSO
    selector = EmbeddedFeatureSelector()
    selected_features = selector.fit_transform(features, survival_time)
    
    return selected_features, selector.get_selected_indices()

# Example usage
if __name__ == "__main__":
    # Assuming patches is numpy array of shape (n_patches, height, width, channels)
    patches = np.random.rand(10, 256, 256, 3) * 255
    patches = patches.astype(np.uint8)
    survival_time = np.random.rand(10)
    
    features, selected_indices = run_feature_pipeline(patches, survival_time)
    print(f"Selected features shape: {features.shape}")
    print(f"Selected feature indices: {selected_indices}")