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
                
        
        patch_embeds  = self.patch_embedding(pixel_values)
        
        
        embeddings = patch_embeds.flatten(2)

        embeddings = embeddings.transpose(1,2)
        
        embeddings = embeddings + self.position_embedding(self.position_ids)
        
        return embeddings        
    
    
    
class SiglipEncoderLayer(nn.Module):
    def __init__(self,config :SiglipVisionConfig):
        super().__init__()
        
        self.config= config
        # MHA
        self.self_atten =  SiglipAttention(config)
        
        #MLP
        self.mlp = SiglipMLP(config)
        
        # layer normalization
        self.layer_norm1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size)
    

    def forward(self,hidden_states: torch.Tensor):
        
        residual = hidden_states
        
        hidden_states = self.layer_norm1(hidden_states)
        
        hidden_states = self.self_atten(hidden_states)
        
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        
        hidden_states = self.layer_norm2(hidden_states)
        
        hidden_states = self.mlp(hidden_states)
        
        hidden_states = residual + hidden_states
        
        return hidden_states
    
    
class  SiglipEncoder(nn.Module):
        def __init__(self, config:SiglipVisionConfig):
            super(),__init__()
            
            self.config = config
            
            self.layers = nn.ModuleList([SiglipEncoderLayer(self.config) for _ in range(self.config.num_hidden_layers)])
            
        def forward(self,inputs_embeds:torch.Tensor):
            
            hidden_states = inputs_embeds
            
            for layer in self.layers:
                
                hidden_states = layer(inputs_embeds)
                
            return hidden_states
        
        
        
class SiglipVisionTransformer(nn.Module):
    
    def __init__(self,config:SiglipVisionConfig):
        super().__init__()
        
        self.config = config
        
        self.embeddings  = SiglipVisionEmbeddings(self.config)
        
        self.encoder = SiglipEncoder(self.config)
        
        self.post_layernorm = nn.LayerNorm(self.config.hidden_size , eps = self.config.layer_norm_eps)
        
        
    def forward(self, pixel_values : torch.Tensor):
        
        hidden_states =self.embeddings(pixel_values)
        
        last_hidden_states = self.encoder(hidden_states)
        
        last_hidden_states = self.post_layernorm(last_hidden_states)
        
        return last_hidden_states
        


class SiglipVisionModel(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values) -> Tuple:
        return self.vision_model(pixel_values=pixel_values) 





if __name__ == "__main__":
    img = "./dog.jpg"
    
    transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
    ])
    pixel_values = transform(Image.open(img)).unsqueeze(0)

    cfg = SiglipVisionConfig()
    embd = SiglipVisionEmbeddings(cfg)
    

    
    print(embd(pixel_values).shape)