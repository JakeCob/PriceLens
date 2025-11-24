
import logging
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Generates visual embeddings using a pre-trained ResNet18 model"""

    def __init__(self, device: str = "cpu"):
        self.device = device
        try:
            # Load pre-trained ResNet18
            weights = models.ResNet18_Weights.DEFAULT
            self.model = models.resnet18(weights=weights)
            
            # Remove the final classification layer to get feature embeddings
            self.model.fc = torch.nn.Identity()
            
            self.model.to(device)
            self.model.eval()
            
            # Standard ImageNet normalization
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
            
            logger.info(f"EmbeddingGenerator initialized on {device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize EmbeddingGenerator: {e}")
            raise

    def generate(self, image: np.ndarray) -> np.ndarray:
        """
        Generate embedding for an image
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            Flattened embedding vector (512 dimensions for ResNet18)
        """
        if image is None or image.size == 0:
            raise ValueError("Empty image provided")

        try:
            # Convert BGR (OpenCV) to RGB (PIL)
            image_rgb = image[:, :, ::-1]
            pil_image = Image.fromarray(image_rgb)
            
            # Preprocess
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                embedding = self.model(input_tensor)
                
            # Return as numpy array
            return embedding.cpu().numpy().flatten()
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return np.zeros(512) # Return zero vector on error
