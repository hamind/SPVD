from open_clip.loss import SigLipLoss, neighbour_exchange_bidir_with_grad, neighbour_exchange_with_grad
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MultiCaptionLoss(nn.Module):
    def __init__(self,
                mode: str = "ignore",
                cache_labels: bool = False,
                rank: int = 0,
                world_size: int = 1,
                dist_impl: Optional[str] = None):
        super().__init__()
        self.mode = mode
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.dist_impl = dist_impl or 'bidir'  # default to bidir exchange for now, this will likely change
        assert self.dist_impl in ('bidir', 'shift', 'reduce', 'gather')

        # cache state FIXME cache not currently used, worthwhile?
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, map, B, K, device, dtype, num_logits, negative_only: bool = False,
        intra_label_mode: str = "ignore",
        pos_iou_thr: float = 0.5,
        neg_iou_thr: float = 0.2,
    ) -> torch.Tensor:
        assert num_logits == B * K
        assert intra_label_mode in ["ignore", "only_pos", "pos_and_neg"]
        labels = -torch.ones(
            (num_logits, num_logits),
            device=device,
            dtype=dtype,
        )

        if not negative_only:
            labels.fill_(-1)
            labels.diagonal().fill_(1)
        map = torch.mean(map, dim=-2).detach()

        # map = (map > 0.5).to(dtype)
        map = map.to(dtype)
        inter = torch.einsum("b i p, b j p -> b i j", map, map)
        area = map.sum(dim=-1)  # (B, K)
        union = (area[:, :, None]+ area[:, None, :]- inter).clamp_min(1e-6)

        map_logits = inter / union  # (B, K, K)

        block_labels = torch.zeros((B, K, K),device=device,dtype=dtype,)

        eye = torch.eye(K, device=device, dtype=torch.bool)
        off_diag = ~eye

        if intra_label_mode == "ignore":
            block_labels[:, off_diag] = 0

        elif intra_label_mode == "only_pos":
            pos_mask = (map_logits >= pos_iou_thr) & off_diag[None, :, :]

            block_labels[:, off_diag] = 0
            block_labels[pos_mask] = 1

        elif intra_label_mode == "pos_and_neg":
            pos_mask = (map_logits >= pos_iou_thr) & off_diag[None, :, :]
            neg_mask = (map_logits <= neg_iou_thr) & off_diag[None, :, :]

            block_labels[:, off_diag] = 0
            block_labels[pos_mask] = 1
            block_labels[neg_mask] = -1

        if not negative_only:
            block_labels[:, eye] = 1
        else:
            block_labels[:, eye] = -1

        labels_4d = labels.view(B, K, B, K)

        idx = torch.arange(B, device=device)
        labels_4d[idx, :, idx, :] = block_labels

        labels = labels_4d.view(num_logits, num_logits)

        return labels

    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        logits = logit_scale * image_features @ text_features.T
        if logit_bias is not None:
            logits += logit_bias
        return logits

    def _loss(self, image_features, text_features, B, K, semantic_map, logit_scale, logit_bias=None, negative_only=False):
        logits = self.get_logits(image_features, text_features, logit_scale, logit_bias)
        labels = self.get_ground_truth(
            semantic_map,
            B,
            K,
            image_features.device,
            image_features.dtype,
            image_features.shape[0],
            negative_only=negative_only,
            intra_label_mode=self.mode,
        )
        valid = labels != 0
        if valid.any():
            loss = -F.logsigmoid(labels[valid] * logits[valid]).sum() / image_features.shape[0]
        else:
            loss = logits.sum() * 0.0

        return loss

    def forward(self, image_features, text_features, semantic_map, logit_scale, logit_bias, output_dict=False):
        B, K, D = text_features.shape
        image_features = image_features.view(B * K, D)
        text_features = text_features.view(B * K, D)

        image_features = F.normalize(image_features , dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        loss = self._loss(image_features, text_features, B, K, semantic_map, logit_scale, logit_bias)

        if self.world_size > 1:
            if self.dist_impl == 'bidir':
                right_rank = (self.rank + 1) % self.world_size
                left_rank = (self.rank - 1 + self.world_size) % self.world_size
                text_features_to_right = text_features_to_left = text_features
                num_bidir, remainder = divmod(self.world_size - 1, 2)
                for i in range(num_bidir):
                    text_features_recv = neighbour_exchange_bidir_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_left,
                        text_features_to_right,
                    )
                    for f in text_features_recv:
                        loss += self._loss(
                            image_features,
                            f,
                            B,
                            K,
                            semantic_map,
                            logit_scale,
                            logit_bias,
                            negative_only=True,
                        )
                    text_features_to_left, text_features_to_right = text_features_recv

                if remainder:
                    text_features_recv = neighbour_exchange_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_right
                    )
                    loss += self._loss(
                        image_features,
                        text_features_recv,
                        B,
                        K,
                        semantic_map,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            elif self.dist_impl == "shift":
                right_rank = (self.rank + 1) % self.world_size
                left_rank = (self.rank - 1 + self.world_size) % self.world_size
                text_features_to_right = text_features
                for i in range(self.world_size - 1):
                    text_features_from_left = neighbour_exchange_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_right,
                    )
                    loss += self._loss(
                        image_features,
                        text_features_from_left,
                        B,
                        K,
                        semantic_map,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
                    text_features_to_right = text_features_from_left
            elif self.dist_impl == "reduce":
                for i in range(self.world_size):
                    text_from_other = torch.distributed.nn.all_reduce(
                        text_features * (self.rank == i),
                        torch.distributed.ReduceOp.SUM,
                    )
                    loss += float(i != self.rank) * self._loss(
                        image_features,
                        text_from_other,
                        B,
                        K,
                        semantic_map,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            elif self.dist_impl == "gather":
                all_text = torch.distributed.nn.all_gather(text_features)
                for i in range(self.world_size):
                    loss += float(i != self.rank) * self._loss(
                        image_features,
                        all_text[i],
                        B,
                        K,
                        semantic_map,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            else:
                assert False

        if output_dict:
            return {"loss_align": loss}
        return loss


class ResidualVarianceLoss(nn.Module):
    """Variance floor for the residual branch."""

    def __init__(self, gamma: float = 1.0, eps: float = 1.0e-4) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.eps = float(eps)

    def forward(self, residual_features):
        residual = residual_features.float().reshape(-1, residual_features.shape[-1])
        std = torch.sqrt(residual.var(dim=0, unbiased=False) + self.eps)
        return F.relu(self.gamma - std).mean()


class BranchBCELoss(nn.Module):
    """Semantic-text positive BCE plus residual-text negative BCE."""

    def __init__(
        self,
        logit_scale: float = 5.0,
        residual_negative_weight: float = 0.25,
        detach_text_for_residual: bool = True,
    ) -> None:
        super().__init__()
        self.logit_scale = float(logit_scale)
        self.residual_negative_weight = float(residual_negative_weight)
        self.detach_text_for_residual = bool(detach_text_for_residual)

    def forward(self, shared_features: Tensor, residual_features: Tensor, text_features: Tensor) -> dict[str, Tensor]:
        if shared_features.shape[:-1] != residual_features.shape[:-1]:
            raise ValueError(
                "shared_features and residual_features must have matching leading dimensions, "
                f"got {tuple(shared_features.shape)} and {tuple(residual_features.shape)}."
            )
        if shared_features.shape[:-1] != text_features.shape[:-1]:
            raise ValueError(
                "shared_features and text_features must have matching leading dimensions for branch BCE, "
                f"got {tuple(shared_features.shape)} and {tuple(text_features.shape)}."
            )
        shared = F.normalize(shared_features.float().reshape(-1, shared_features.shape[-1]), dim=-1)
        residual = F.normalize(residual_features.float().reshape(-1, residual_features.shape[-1]), dim=-1)
        text = F.normalize(text_features.float().reshape(-1, text_features.shape[-1]), dim=-1)
        residual_text = text.detach() if self.detach_text_for_residual else text

        sim_s_text = (shared * text).sum(dim=-1)
        sim_r_text = (residual * residual_text).sum(dim=-1)
        logits_s = self.logit_scale * sim_s_text
        logits_r = self.logit_scale * sim_r_text
        loss_s_text = F.binary_cross_entropy_with_logits(logits_s, torch.ones_like(logits_s))
        loss_r_text = F.binary_cross_entropy_with_logits(logits_r, torch.zeros_like(logits_r))
        loss_branch = loss_s_text + self.residual_negative_weight * loss_r_text

        return {
            "loss_branch": loss_branch,
            "loss_branch_s_text": loss_s_text,
            "loss_branch_r_text": loss_r_text,
            "branch_sim_s_text": sim_s_text.detach().mean(),
            "branch_sim_r_text": sim_r_text.detach().mean(),
            "branch_gap_s_minus_r": (sim_s_text.detach() - sim_r_text.detach()).mean(),
        }

class SPVDLoss(nn.Module):
    expects_output_dict = True

    def __init__(
        self,
        local_loss: bool = False,
        gather_with_grad: bool = False,
        cache_labels: bool = False,
        rank: int = 0,
        world_size: int = 1,
        branch_bce_weight: float = 0.0,
        branch_logit_scale: float = 5.0,
        residual_negative_weight: float = 0.25,
        detach_text_for_residual: bool = True,
        residual_variance_weight: float = 0.0,
        residual_variance_gamma: float = 1.0,
        caption_same_image_mode: str = "ignore",
        align_weight: float = 1.0,
        global_align_weight: float = 1.0,
        caption_align_weight: float = 1.0,
        loss_dist_impl: str | None = None,
        debug_finite_checks: bool = False,
    ) -> None:
        super().__init__()
        self.multi_align_loss = MultiCaptionLoss(
            mode=caption_same_image_mode,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            dist_impl=loss_dist_impl,
        )
        self.residual_variance_loss = ResidualVarianceLoss(gamma=residual_variance_gamma)
        self.branch_bce_loss = BranchBCELoss(
            logit_scale=branch_logit_scale,
            residual_negative_weight=residual_negative_weight,
            detach_text_for_residual=detach_text_for_residual,
        )
        self.align_weight = float(align_weight)
        self.branch_bce_weight = float(branch_bce_weight)
        self.residual_variance_weight = float(residual_variance_weight)
        self.debug_finite_checks = bool(debug_finite_checks)

    def forward(self, outputs=None, output_dict: bool = False, **kwargs):
        if outputs is None:
            outputs = kwargs
        if not isinstance(outputs, dict):
            raise TypeError("SPVDLoss expects a model output dict.")

        semantic = outputs["semantic"]
        residual = outputs["residual"]
        text_features = outputs["text_features"]
        semantic_map = outputs["semantic_map"]
        logit_scale = outputs["logit_scale"]
        logit_bias = outputs.get("logit_bias")

        if self.debug_finite_checks:
            for name, tensor in (
                ("semantic", semantic),
                ("residual", residual),
                ("text_features", text_features),
                ("semantic_map", semantic_map),
                ("logit_scale", logit_scale),
            ):
                if torch.is_tensor(tensor) and not torch.isfinite(tensor).all():
                    raise FloatingPointError(f"{name} contains non-finite values")

        loss_align = self.multi_align_loss(
            semantic,
            text_features,
            semantic_map,
            logit_scale,
            logit_bias,
            output_dict=False,
        )
        branch_terms = self.branch_bce_loss(semantic, residual, text_features)
        loss_residual_variance = self.residual_variance_loss(residual)
        total_loss = (
            self.align_weight * loss_align
            + self.branch_bce_weight * branch_terms["loss_branch"]
            + self.residual_variance_weight * loss_residual_variance
        )

        if output_dict:
            return {
                "loss_align": self.align_weight * loss_align,
                "loss_branch": self.branch_bce_weight * branch_terms["loss_branch"],
                "loss_residual_variance": self.residual_variance_weight * loss_residual_variance,
            }
        losses = {
            "loss_align": loss_align,
            "loss_sigmoid": loss_align.detach(),
            "loss_branch": branch_terms["loss_branch"],
            "loss_branch_s_text": branch_terms["loss_branch_s_text"].detach(),
            "loss_branch_r_text": branch_terms["loss_branch_r_text"].detach(),
            "branch_sim_s_text": branch_terms["branch_sim_s_text"].detach(),
            "branch_sim_r_text": branch_terms["branch_sim_r_text"].detach(),
            "branch_gap_s_minus_r": branch_terms["branch_gap_s_minus_r"].detach(),
            "loss_residual_variance": loss_residual_variance,
            "logit_scale": logit_scale.detach(),
        }
        return total_loss, {key: value.detach() if torch.is_tensor(value) else value for key, value in losses.items()}
