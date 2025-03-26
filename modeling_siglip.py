import torch
from torch import nn
from typing import Optional , Tuple
import torchvision.transforms as transforms
from PIL import Image

class SiglipVisionConfig:
    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers = 12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size= 16,
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_image_tokens:int = None,
        **kwargs,           
        ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.intermediate_size =intermediate_size
        self.num_hidden_layers =num_hidden_layers
        self.num_attention_heads=num_attention_heads
        self.num_channels=num_channels
        self.image_size=image_size
        self.patch_size=patch_size
        self.layer_norm_eps=layer_norm_eps
        self.attention_dropout=attention_dropout
        self.num_image_tokens=num_image_tokens
        

class SiglipVisionEmbeddings(nn.Module):
    def __init__(self,config : SiglipVisionConfig):
        super().__init__()
        
        self.config = config
        self.num_channels = config.num_channels
        self.embed_dim=config.hidden_size
        self.patch_size = config.patch_size
        self.image_size = config.image_size
        
        self.patch_embedding  = nn.Conv2d(
            in_channels = self.num_channels ,
            out_channels = self.embed_dim ,
            kernel_size = self.patch_size , 
            stride = self.patch_size,
            padding=0
            )
        
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand(1,-1),persistent=False)
        
        
    def forward(self,pixel_values):
        
        
        if isinstance(pixel_values, str):
            transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
        ])
        pixel_values = transform(Image.open(pixel_values)).unsqueeze(0)

        if not isinstance(pixel_values, torch.Tensor):
            raise TypeError("Input must be a tensor or a file path")
            
        
        patch_embeds  = self.patch_embedding(pixel_values)
        
        
        embeddings = patch_embeds.flatten(2)

        embeddings = embeddings.transpose(1,2)
        
        embeddings = embeddings + self.position_embedding(self.position_ids)
        
        return embeddings        
    
    

if __name__ == "__main__":
    img = "./dog.jpg"
    cfg = SiglipVisionConfig()
    embd = SiglipVisionEmbeddings(cfg)
    

    
    print(embd(img).shape)