from __future__ import annotations

import torch
import torch.nn as nn

from modules.visual_projection.projection import VisualProjection
from modules.class_embedding.class_embedding import ClassEmbedding
from modules.spatial_embedding.spatial_embedding import SpatialEmbedding
from modules.text_projection.text_projection import TextProjection
from modules.context_reasoning.context import ResidualContextPropagation
from modules.context_reasoning.task_conditioning import TaskConditioning
from modules.similarity.similarity import SimilarityCalculation
from modules.ranking.ranking import RankingLayer
from modules.selection.selection import ObjectSelector
from modules.task_encoder.task_encoder import TASK_DESCRIPTIONS


class ReasoningNetwork(nn.Module):
    """
    Unified task-aware reasoning network that processes visual, spatial, class, and task prompt
    embeddings to rank and select the best candidate object proposal.
    """

    def __init__(
        self,
        default_threshold: float = 0.0,
        prior_tensor: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.visual_projection = VisualProjection()
        self.spatial_embedding = SpatialEmbedding()
        self.class_embedding = ClassEmbedding()
        self.text_projection = TextProjection()
        self.task_conditioning = TaskConditioning()
        self.context_module = ResidualContextPropagation()
        self.similarity_module = SimilarityCalculation()
        self.ranking_module = RankingLayer()
        self.selector = ObjectSelector(default_threshold=default_threshold)

        if prior_tensor is None:
            prior_tensor = torch.zeros((len(TASK_DESCRIPTIONS), 91), dtype=torch.float32)

        self.register_buffer("category_prior_tensor", prior_tensor)

    def forward(
        self,
        roi_features: torch.Tensor,
        boxes: torch.Tensor,
        class_ids: torch.Tensor,
        scores: torch.Tensor,
        task_raw_emb: torch.Tensor,
        task_id: int,
        img_size: tuple[int, int],
        threshold: float | None = None,
    ) -> tuple[torch.Tensor, int, float]:
        """
        Runs the full task-aware reasoning pipeline.
        
        Args:
            roi_features: Tensor of shape [M, 256, 5, 5]
            boxes: Tensor of shape [M, 4] (absolute coordinates [x1, y1, x2, y2])
            class_ids: LongTensor of shape [M] containing category IDs
            scores: Tensor of shape [M] containing detection confidences
            task_raw_emb: Tensor of shape [384] containing SentenceTransformer prompt embedding
            img_size: Tuple containing (height, width) of the image
            threshold: Optional threshold override for selection
            
        Returns:
            A tuple containing:
                logits: Tensor of shape [M] containing preference scores
                best_idx: Index of the selected object, or -1 if none exceed threshold
                best_score: The preference score of the selected object
        """
        # If no boxes are present, return default values
        if len(boxes) == 0:
            return torch.zeros((0,), dtype=torch.float32), -1, float("-inf")

        # 1. Project visual features to 128-D
        e_v = self.visual_projection(roi_features)  # Shape [M, 128]

        # 2. Project prompt embedding to 192-D
        t = self.text_projection(task_raw_emb)  # Shape [192]

        # 3. Modulate visual features via task conditioning
        e_v = self.task_conditioning(e_v, t)

        # 4. Project spatial coordinates to 32-D
        e_s = self.spatial_embedding(boxes, img_size)  # Shape [M, 32]

        # 5. Lookup category ID embeddings (32-D)
        e_c = self.class_embedding(class_ids)  # Shape [M, 32]

        # 6. Concatenate visual, spatial, and class embeddings to 192-D
        f = torch.cat([e_v, e_s, e_c], dim=1)  # Shape [M, 192]

        # 7. Apply object-specific residual context propagation
        f_updated = self.context_module(f, scores)  # Shape [M, 192]

        # 8. Compute task-object similarity scores
        sim = self.similarity_module(f_updated, t)  # Shape [M]

        # 9. Retrieve semantic category priors for the current task
        category_priors = self.category_prior_tensor[task_id - 1, class_ids].to(sim.device).to(sim.dtype)

        # 10. Compute preference logits with tuned combination of similarity, confidence, and priors
        logits = self.ranking_module(sim, scores, category_priors)

        # 11. Filter and select the best index using thresholding
        best_idx, best_score = self.selector(logits, threshold=threshold)

        return logits, best_idx, best_score
