import numpy as np
import os
import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image, ImageOps
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

class EnhancedELM:
    def __init__(self, pca_var=0.95, hidden_units=1000, 
                 activation='relu', reg_param=1e-6, init_mode='he'):
        self.pca_var = pca_var
        self.hidden_units = hidden_units
        self.activation = activation
        self.reg_param = reg_param
        self.init_mode = init_mode
        self.input_shape = (200, 200)
        
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.pca_var, whiten=True)
        self.le = LabelEncoder()
        self.input_weights = None
        self.biases = None
        self.output_weights = None

    def _initialize_weights(self, input_size):
        if self.init_mode == 'he':
            scale = np.sqrt(2.0 / input_size)
        elif self.init_mode == 'xavier':
            scale = np.sqrt(1.0 / (input_size + self.hidden_units))
        else:
            scale = 0.1
            
        self.input_weights = np.random.randn(input_size, self.hidden_units) * scale
        self.biases = np.random.randn(self.hidden_units) * 0.01

    def _apply_activation(self, X):
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-X))
        elif self.activation == 'relu':
            return np.maximum(0, X)
        elif self.activation == 'tanh':
            return np.tanh(X)
        raise ValueError(f"Unsupported activation: {self.activation}")

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        X_reduced = self.pca.fit_transform(X_scaled)
        
        self._initialize_weights(X_reduced.shape[1])
        y_encoded = self.le.fit_transform(y)
        Y_one_hot = np.eye(len(self.le.classes_))[y_encoded]

        H = self._apply_activation(X_reduced @ self.input_weights + self.biases)
        HtH = H.T @ H + self.reg_param * np.eye(self.hidden_units)
        
        try:
            self.output_weights = np.linalg.pinv(HtH) @ H.T @ Y_one_hot
        except np.linalg.LinAlgError:
            self.output_weights = np.linalg.lstsq(HtH, H.T @ Y_one_hot, rcond=None)[0]
        
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        X_reduced = self.pca.transform(X_scaled)
        H = self._apply_activation(X_reduced @ self.input_weights + self.biases)
        return self.le.inverse_transform(np.argmax(H @ self.output_weights, axis=1))

    def evaluate(self, X, y):
        return accuracy_score(y, self.predict(X))

class ELMVisualizer:
    def __init__(self, elm_model):
        self.model = elm_model
        self._process_weights()
        
    def _process_weights(self):
        self.output_weights_abs = np.abs(self.model.output_weights)
        self.node_importance = np.sum(self.output_weights_abs, axis=1)
        self.node_importance = (self.node_importance - self.node_importance.min()) / \
                              (self.node_importance.max() - self.node_importance.min())
    
    def plot_node_network(self, figsize=(12, 8)):
        plt.figure(figsize=figsize)
        G = nx.Graph()
        for i in range(len(self.node_importance)):
            G.add_node(i)
            
        pos = nx.spring_layout(G, seed=42)
        nodes = nx.draw_networkx_nodes(G, pos, 
                                     node_size=800 + 4000 * self.node_importance,
                                     node_color=self.node_importance, 
                                     cmap=plt.cm.plasma, alpha=0.9)
        plt.colorbar(nodes, label='Node Importance')
        plt.title("ELM Hidden Layer Node Importance Network")
        plt.axis('off')
        plt.show()
    
    def plot_weight_heatmap(self, figsize=(12, 6)):
        plt.figure(figsize=figsize)
        plt.imshow(self.output_weights_abs.T, aspect='auto', cmap='viridis')
        plt.colorbar(label='Weight Magnitude')
        plt.xlabel('Hidden Nodes')
        plt.ylabel('Output Classes')
        plt.title('Class-Node Weight Distribution')
        plt.show()
    
    def plot_receptive_fields(self, n_fields=25, figsize=(15, 10)):
        input_weights = self.model.pca.inverse_transform(self.model.input_weights.T)
        input_weights = input_weights.reshape(-1, *self.model.input_shape)
        
        plt.figure(figsize=figsize)
        plt.suptitle("Learned Feature Detectors", y=0.95)
        sorted_nodes = np.argsort(-self.node_importance)[:n_fields]
        
        for i, node in enumerate(sorted_nodes):
            plt.subplot(5, 5, i+1)
            img = input_weights[node]
            img = (img - img.min()) / (img.max() - img.min())
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            plt.title(f'Node {node}\nImp: {self.node_importance[node]:.2f}', fontsize=8)
        
        plt.tight_layout()
        plt.show()

class ImageProcessor:
    def __init__(self, input_shape=(200, 200)):
        self.input_shape = input_shape
        
    def _augment_image(self, img):
        # Geometric transformations
        img = img.rotate(np.random.uniform(-30, 30))
        if np.random.rand() > 0.5:
            img = ImageOps.mirror(img)
        
        # Photometric transformations
        img = np.array(img).astype(np.float32)
        img = img * np.random.uniform(0.5, 1.5)  # Contrast
        img += np.random.normal(0, 25, img.shape)  # Gaussian noise
        img = np.clip(img, 0, 255)
        
        # Random cropping
        h, w = img.shape
        crop_size = int(min(h, w) * np.random.uniform(0.7, 0.9))
        y = np.random.randint(0, h - crop_size)
        x = np.random.randint(0, w - crop_size)
        img = img[y:y+crop_size, x:x+crop_size]
        
        return Image.fromarray(img.astype(np.uint8)).resize(self.input_shape)
    
    def process_image(self, path, augment=False):
        img = Image.open(path).convert('L').resize(self.input_shape)
        if augment:
            img = self._augment_image(img)
        return np.array(img).flatten()

def load_dataset(folder, processor, augment=False):
    paths, labels = [], []
    classes = sorted([d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))])
    
    for cls in classes:
        cls_dir = os.path.join(folder, cls)
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                paths.append(os.path.join(cls_dir, fname))
                labels.append(cls)
    
    images = np.array([processor.process_image(p, augment) for p in paths])
    return images, np.array(labels)

def main():
    # Configuration
    input_shape = (200, 200)
    train_folder = "C:/Users/joaop/Documents/ELM/KTH_train"
    test_folder = "C:/Users/joaop/Documents/ELM/KTH_test"
    
    # Data processing
    processor = ImageProcessor(input_shape)
    X_train, y_train = load_dataset(train_folder, processor, augment=True)
    X_test, y_test = load_dataset(test_folder, processor, augment=False)
    
    # Hyperparameter search
    param_grid = {
        'pca_var': [0.90, 0.95],
        'hidden_units': [1000, 2000],
        'reg_param': [1e-6, 1e-7],
        'activation': ['relu']
    }
    
    best_acc = 0
    best_model = None
    
    for params in ParameterGrid(param_grid):
        print(f"\nTesting {params}")
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train)
        
        try:
            model = EnhancedELM(**params).fit(X_tr, y_tr)
            val_acc = model.evaluate(X_val, y_val)
            test_acc = model.evaluate(X_test, y_test)
            print(f"Val: {val_acc:.3f} | Test: {test_acc:.3f}")
            
            if val_acc > best_acc:
                best_acc = val_acc
                best_model = model
                print("New best model!")
                
        except Exception as e:
            print(f"Error: {str(e)}")
    
    # Visualization and final evaluation
    if best_model:
        print("\n=== Best Model Performance ===")
        print(f"Final Test Accuracy: {best_model.evaluate(X_test, y_test):.4f}")
        
        visualizer = ELMVisualizer(best_model)
        visualizer.plot_node_network()
        visualizer.plot_weight_heatmap()
        visualizer.plot_receptive_fields()
    else:
        print("No valid model found!")

if __name__ == "__main__":
    main()