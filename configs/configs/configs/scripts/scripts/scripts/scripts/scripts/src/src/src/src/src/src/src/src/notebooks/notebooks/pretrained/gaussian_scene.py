import numpy as np

class GaussianSplatScene:
    def __init__(self, num_splats=150):
        self.num_splats = num_splats
        self.splats = self._generate_scene()

    def _generate_scene(self):
        means = np.random.uniform(-1.0, 1.0, (self.num_splats, 3))
        covariances = np.random.uniform(0.005, 0.03, (self.num_splats, 3))
        colors = np.random.uniform(0, 1, (self.num_splats, 3))
        return {"means": means, "covs": covariances, "colors": colors}

    def sample_view(self, view_direction):
        view_direction = view_direction / np.linalg.norm(view_direction)

        projected = np.dot(self.splats["means"], view_direction)

        sorted_idx = np.argsort(projected)
        means = self.splats["means"][sorted_idx]
        covs = self.splats["covs"][sorted_idx]
        colors = self.splats["colors"][sorted_idx]

        features = np.concatenate([means, covs, colors], axis=1)
        return features.astype(np.float32)
