import torch


class CausalIntervention:

    def __init__(self, activations: torch.Tensor, seed: int = 42) -> None:
        self.activations = activations
        self.seed = seed

    def extract_principal_direction(self) -> torch.Tensor:
        dirs, _, _ = self.extract_top_k_directions(k=1)
        return dirs[0]

    def extract_top_k_directions(
        self, k: int = 5
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        centered = self.activations - self.activations.mean(dim=0, keepdim=True)
        _, S, Vt = torch.linalg.svd(centered, full_matrices=False)
        k_clamped = min(k, Vt.size(0))
        directions = Vt[:k_clamped, :]
        singular_values = S[:k_clamped]
        total_variance = torch.sum(S**2)
        explained_variance = (singular_values**2) / (total_variance + 1e-12)
        return directions, singular_values, explained_variance

    def _get_projection(
        self, direction: torch.Tensor, activations: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        acts = self.activations if activations is None else activations
        direction = direction.to(acts.device)
        direction = direction / torch.norm(direction)
        magnitudes = acts @ direction  # [batch]
        projections = magnitudes.unsqueeze(1) * direction  # [batch, hidden_dim]
        return magnitudes, projections

    def projection_out(
        self, direction: torch.Tensor, activations: torch.Tensor | None = None
    ) -> torch.Tensor:
        acts = self.activations if activations is None else activations
        _, projections = self._get_projection(direction, acts)
        return acts - projections

    def amplification(
        self,
        direction: torch.Tensor,
        multiplier: float = 2.0,
        activations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        acts = self.activations if activations is None else activations
        _, projections = self._get_projection(direction, acts)
        return acts + (multiplier - 1.0) * projections

    def clamping_to_mean(
        self, direction: torch.Tensor, activations: torch.Tensor | None = None
    ) -> torch.Tensor:
        acts = self.activations if activations is None else activations
        direction = direction.to(acts.device)
        direction = direction / torch.norm(direction)
        magnitudes = acts @ direction
        mean_mag = magnitudes.mean()
        delta = (mean_mag - magnitudes).unsqueeze(1) * direction
        return acts + delta

    def swapping(
        self,
        direction: torch.Tensor,
        activations: torch.Tensor | None = None,
        seed: int | None = None,
    ) -> torch.Tensor:
        acts = self.activations if activations is None else activations
        magnitudes, projections = self._get_projection(direction, acts)

        shuffle_seed = self.seed if seed is None else seed
        generator = torch.Generator(device="cpu")
        generator.manual_seed(shuffle_seed)
        shuffled_indices = torch.randperm(acts.size(0), generator=generator).to(acts.device)
        shuffled_magnitudes = magnitudes[shuffled_indices]

        direction = direction.to(acts.device)
        direction_unit = direction / torch.norm(direction)
        swapped_projections = shuffled_magnitudes.unsqueeze(1) * direction_unit

        return (acts - projections) + swapped_projections

    def get_matched_random_control(
        self, reference_direction: torch.Tensor, seed: int | None = None
    ) -> torch.Tensor:
        control_seed = (self.seed + 1) if seed is None else seed
        generator = torch.Generator(device="cpu")
        generator.manual_seed(control_seed)
        random_dir = torch.randn(reference_direction.shape, generator=generator).to(
            reference_direction.device
        )
        return random_dir / torch.norm(random_dir)