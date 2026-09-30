
def activation_hook():
    activations = [] 
    
    def hook(model, input, output):
        activations.append(output.detach().clone().cpu())
        
    return hook, activations
