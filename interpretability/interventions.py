import torch
class Casual_Intervention:
    def __init__(self,activations):
        self.activations = activations
    
    def latent_direction(self):

        U, _, Vt = torch.linalg.svd(self.activations)
        direction = U[:, 0]
        return direction
    

    def ablation(self):
        return self.activations * (self.activations > 0)

    def swapping(self):
        return self.activations[torch.randperm(self.activations.size(0))]

    def projection_out(self,activations):
        U, _, Vt = torch.linalg.svd(self.activations)
        direction = U[:, 0]
        return self.activations - activations 

    def amplification(self):
        return self.activations * 2

    def matched_random_controls(self):
        random_direction = torch.randn_like(self.activations)
        random_direction = random_direction / torch.norm(random_direction, dim=1, keepdim=True)
        return random_direction

        
