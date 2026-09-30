from jepa.model import JEPA
from interpretability.hook import activation_hook
from interpretability.interventions import CausalIntervention
import torch

model = JEPA()

hook_fn, activations = activation_hook()

model.context_encoder.register_forward_hook(hook_fn)

intervener = CausalIntervention(activations)
feature_dir = intervener.extract_principal_direction()
random_dir = intervener.get_matched_random_control(feature_dir)

ablated_activations = intervener.projection_out(feature_dir)

random_control_activations = intervener.projection_out(random_dir)
