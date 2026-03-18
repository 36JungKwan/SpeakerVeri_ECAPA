from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from app.feature_extractor import DemoFeatureExtractor
from train.model import get_model


@dataclass
class RuntimeConfig:
    mode: int = 3
    fusion_method: str = "concat"
    feature_mode: str = "mfbe_pitch"


class SpeakerDemoRuntime:
    def __init__(self, checkpoint_path: str, device: str = "cpu", config: RuntimeConfig | None = None) -> None:
        self.device = torch.device(device)
        self.config = config or RuntimeConfig()
        self.extractor = DemoFeatureExtractor(device=str(self.device))

        self.model = get_model(
            num_speakers=1,
            device=str(self.device),
            mode=self.config.mode,
            fusion_method=self.config.fusion_method,
            feature_mode=self.config.feature_mode,
        )
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

    @torch.inference_mode()
    def embedding_from_audio(self, audio_path: str) -> torch.Tensor:
        fbank, handcrafted = self.extractor.extract_for_mode3_concat_mfbe_pitch(audio_path)
        _, emb = self.model(
            fbank=fbank.unsqueeze(0).to(self.device),
            handcrafted=handcrafted.unsqueeze(0).to(self.device),
        )
        emb = F.normalize(emb, p=2, dim=1)
        return emb.squeeze(0).cpu()

    @torch.inference_mode()
    def compare_two_audio(self, audio_a: str, audio_b: str) -> float:
        emb_a = self.embedding_from_audio(audio_a)
        emb_b = self.embedding_from_audio(audio_b)
        score = torch.sum(emb_a * emb_b).item()
        return float(score)

    @staticmethod
    def identify(embedding: torch.Tensor, centroids: Dict[str, torch.Tensor]) -> Tuple[str | None, Dict[str, float]]:
        if not centroids:
            return None, {}

        scores: Dict[str, float] = {}
        for speaker_id, centroid in centroids.items():
            scores[speaker_id] = float(torch.sum(embedding * centroid).item())

        best_speaker = max(scores.items(), key=lambda x: x[1])[0]
        return best_speaker, scores
