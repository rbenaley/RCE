"""
Implémentation monolithique de RCE-LLM (Relational Coherence Engine)
compatible avec une API OpenAI-like (ex: LM Studio).

Dépendances principales :
    pip install torch openai numpy

Ce fichier contient :
    - Client LM Studio (API OpenAI-compatible)
    - Graphizer piloté par LLM (construction du graphe candidat)
    - Encodeur de contexte par embeddings LLM
    - Moteur RCE : modules de cohérence µ_k, µ(Ω|C), actualisation Ω*
    - Objectif d'entraînement L_coherence avec relaxation Gumbel-Softmax
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Callable, Any

import json
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from openai import OpenAI

import os


# ---------------------------------------------------------------------------
# 1. Client LM Studio (OpenAI-compatible)
# ---------------------------------------------------------------------------

def get_lmstudio_client(
    base_url: str = "http://localhost:1234/v1",
    api_key: str = "lm-studio",
) -> OpenAI:
    """
    Retourne un client OpenAI-compatible pointant vers LM Studio.
    """
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )
    return client


# ---------------------------------------------------------------------------
# 2. Structures de graphe relationnel
# ---------------------------------------------------------------------------

@dataclass
class Node:
    id: int
    label: str
    type: str = "entity"
    attrs: Optional[Dict[str, Any]] = None


@dataclass
class Relation:
    id: int
    head: int
    tail: int
    rtype: str
    attrs: Optional[Dict[str, Any]] = None


@dataclass
class RelationalGraph:
    nodes: List[Node]
    relations: List[Relation]
    rel_features: torch.Tensor  # [R, d_rel]

    def to_device(self, device: torch.device) -> "RelationalGraph":
        self.rel_features = self.rel_features.to(device)
        return self


# ---------------------------------------------------------------------------
# 3. Encodage par LLM : graphizer + contexte
# ---------------------------------------------------------------------------

class GraphizerLLM:
    """
    Utilise un LLM de chat pour extraire un graphe :
        - nodes: [ {id, label, type}, ... ]
        - relations: [ {id, head, tail, rtype}, ... ]

    On ajoute une validation pour éviter les relations invalides (head/tail hors bornes).
    """

    def __init__(self, client: OpenAI, model_name: str):
        self.client = client
        self.model_name = model_name

    def _validate_graph(self, data: Dict[str, Any]) -> Dict[str, Any]:
        nodes = data.get("nodes", [])
        relations = data.get("relations", [])

        if not isinstance(nodes, list) or not isinstance(relations, list):
            raise RuntimeError(
                f"nodes/relations ne sont pas des listes : {data}"
            )

        max_id = len(nodes) - 1
        valid_relations: List[Dict[str, Any]] = []
        invalid_relations: List[Dict[str, Any]] = []

        for r in relations:
            h = r.get("head")
            t = r.get("tail")
            if (
                isinstance(h, int)
                and isinstance(t, int)
                and 0 <= h <= max_id
                and 0 <= t <= max_id
            ):
                valid_relations.append(r)
            else:
                invalid_relations.append(r)

        if invalid_relations:
            print("⚠️  Relations invalides supprimées du graphe :")
            for r in invalid_relations:
                print("   →", r)

        data["relations"] = valid_relations
        return data

    def graphize(self, text: str) -> Dict[str, Any]:
        system_msg = (
            "Tu es un extracteur de graphes. "
            "Tu dois retourner STRICTEMENT un JSON sérialisé valide, sans explication."
        )

        user_prompt = f"""
Analyse le texte suivant et construis un graphe relationnel.

Retourne STRICTEMENT un JSON avec cette structure exacte :

{{
  "nodes": [
    {{ "id": int, "label": str, "type": str }}
  ],
  "relations": [
    {{ "id": int, "head": int, "tail": int, "rtype": str }}
  ]
}}

Le JSON doit être bien formé, parseable, et les ids de head/tail doivent
correspondre à des indices de la liste "nodes".

Texte :
\"\"\"{text}\"\"\"
"""

        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )

        content = resp.choices[0].message.content.strip()

        # Debug facultatif :
        # print("\n=== RAW GRAPHIZER OUTPUT ===")
        # print(content)
        # print("================================\n")

        # Parse JSON
        try:
            data = json.loads(content)
        except Exception:
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                snippet = content[start : end + 1]
                data = json.loads(snippet)
            else:
                raise RuntimeError(
                    "Le LLM graphizer n'a pas renvoyé de JSON parseable. "
                    f"Contenu brut : {content}"
                )

        if "nodes" not in data or "relations" not in data:
            raise RuntimeError(
                "Le LLM graphizer doit renvoyer un JSON avec les clés "
                f"'nodes' et 'relations'. JSON reçu : {data}"
            )

        if not isinstance(data["nodes"], list) or not isinstance(data["relations"], list):
            raise RuntimeError(
                "Les champs 'nodes' et 'relations' doivent être des listes. "
                f"JSON reçu : {data}"
            )

        if len(data["nodes"]) == 0:
            raise RuntimeError(
                "Le LLM graphizer a renvoyé 0 nœud. "
                "Impossible de construire un graphe utile."
            )

        # Nettoyage / validation des relations
        data = self._validate_graph(data)

        if len(data["relations"]) == 0:
            print("⚠️  Le LLM graphizer a renvoyé 0 relation après validation.")

        return data


class ContextEncoderLLM:
    """
    Génère un vecteur de contexte C via l'API /embeddings (OpenAI-compatible).
    """

    def __init__(self, client: OpenAI, embedding_model: str):
        self.client = client
        self.embedding_model = embedding_model

    def encode(self, text: str) -> torch.Tensor:
        resp = self.client.embeddings.create(
            model=self.embedding_model,
            input=text,
        )

        # Debug facultatif :
        # print("=== EMBEDDING RAW RESPONSE ===")
        # print(resp)
        # print("================================")

        if not hasattr(resp, "data") or resp.data is None or len(resp.data) == 0:
            raise RuntimeError(
                f"Le modèle d'embedding '{self.embedding_model}' "
                f"n'a renvoyé aucune donnée (data est vide). "
                f"Vérifie qu'il supporte bien /embeddings côté LM Studio."
            )

        if not hasattr(resp.data[0], "embedding"):
            raise RuntimeError(
                f"Le modèle d'embedding '{self.embedding_model}' "
                f"n'a pas de champ 'embedding' dans resp.data[0]. "
                f"Sortie inattendue de l'API."
            )

        emb = resp.data[0].embedding
        return torch.tensor(emb, dtype=torch.float32)


# ---------------------------------------------------------------------------
# 4. Modules de cohérence µ_k et config RCE
# ---------------------------------------------------------------------------

class CoherenceModule(nn.Module):
    """
    µ_k({r} | C) ∈ [0,1] pour chaque relation, implémenté par un MLP.

    Entrée :
        concat([features_relation, contexte]) -> score dans [0,1].
    """

    def __init__(self, rel_dim: int, ctx_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(rel_dim + ctx_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, rel_feats: torch.Tensor, ctx_vec: torch.Tensor) -> torch.Tensor:
        """
        rel_feats : [R, d_rel]
        ctx_vec   : [d_ctx] ou [1, d_ctx]
        Retour :   [R]
        """
        if ctx_vec.dim() == 1:
            ctx = ctx_vec.unsqueeze(0).expand(rel_feats.size(0), -1)
        else:
            assert ctx_vec.size(0) == 1
            ctx = ctx_vec.expand(rel_feats.size(0), -1)

        x = torch.cat([rel_feats, ctx], dim=-1)
        scores = self.net(x).squeeze(-1)
        return scores


class UnitsCoherence(CoherenceModule):
    """µ_units : cohérence des unités, dimensions, conversions."""
    pass


class TimeCoherence(CoherenceModule):
    """µ_time : cohérence temporelle / chronologie."""
    pass


class ArithmeticCoherence(CoherenceModule):
    """µ_arith : cohérence arithmétique / numérique."""
    pass


class CorefCoherence(CoherenceModule):
    """µ_coref : coreference / stabilité des entités."""
    pass


class EntailmentCoherence(CoherenceModule):
    """µ_entail : ancrage factuel / entailment."""
    pass


@dataclass
class RCEConfig:
    rel_dim: int              # dimension des features de relations
    ctx_dim: int              # dimension du vecteur de contexte C
    num_modules: int = 5      # K (units, time, arith, coref, entail)
    reduction: str = "mean"
    beam_size: int = 10
    max_relations_in_subgraph: Optional[int] = None
    margin_delta: float = 0.1
    lambda_margin: float = 1.0
    gumbel_tau: float = 0.5
    use_gpu: bool = True


# ---------------------------------------------------------------------------
# 5. Relational Coherence Engine
# ---------------------------------------------------------------------------

class RelationalCoherenceEngine(nn.Module):
    """
    Implémentation d'un RCE-LLM :

        µ(Ω | C) = Σ_k w_k(C) µ_k(Ω | C)

    µ_k(Ω | C) = Σ_{r∈Ω} µ_k({r} | C)
    µ_local(r | C) = Σ_k w_k(C) µ_k({r} | C)

    Actualisation approx. Ω* via beam search.
    """

    def __init__(self, cfg: RCEConfig):
        super().__init__()
        self.cfg = cfg

        # Device
        if self.cfg.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Poids contextuels w_k(C) (softmax)
        self.ctx_to_weights = nn.Sequential(
            nn.Linear(self.cfg.ctx_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.cfg.num_modules),
            nn.Softmax(dim=-1),
        )

        rel_dim = self.cfg.rel_dim
        ctx_dim = self.cfg.ctx_dim

        # IMPORTANT : utiliser un nom différent de "modules" pour éviter de
        # masquer la méthode .modules() de nn.Module.
        self.mu_modules = nn.ModuleList([
            UnitsCoherence(rel_dim, ctx_dim),
            TimeCoherence(rel_dim, ctx_dim),
            ArithmeticCoherence(rel_dim, ctx_dim),
            CorefCoherence(rel_dim, ctx_dim),
            EntailmentCoherence(rel_dim, ctx_dim),
        ])
        assert len(self.mu_modules) == self.cfg.num_modules

        self.to(self.device)

    # -------------------------
    # 5.1 Scores µ_k({r} | C) et µ_local
    # -------------------------

    def compute_module_scores(
        self,
        graph: RelationalGraph,
        ctx_vec: torch.Tensor,
    ) -> torch.Tensor:
        """
        Retourne [K, R] : scores[k, r] = µ_k({r} | C)
        """
        rel_feats = graph.rel_features.to(self.device)
        ctx_vec = ctx_vec.to(self.device)

        scores = []
        for module in self.mu_modules:
            s_k = module(rel_feats, ctx_vec)
            scores.append(s_k)
        scores = torch.stack(scores, dim=0)  # [K, R]
        return scores

    def compute_local_scores(
        self,
        graph: RelationalGraph,
        ctx_vec: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        module_scores : [K, R]
        local_scores  : [R] (µ_local(r | C))
        """
        module_scores = self.compute_module_scores(graph, ctx_vec)  # [K, R]
        w = self.ctx_to_weights(ctx_vec.to(self.device))            # [K]
        w_exp = w.unsqueeze(-1)                                    # [K, 1]
        local_scores = (w_exp * module_scores).sum(dim=0)          # [R]
        return module_scores, local_scores

    # -------------------------
    # 5.2 Coherence d'un sous-graphe Ω
    # -------------------------

    @staticmethod
    def _subset_to_mask(R: int, subset: List[int]) -> torch.Tensor:
        """Crée un masque booléen de taille R à partir d'une liste d'indices.

        On clippe automatiquement les indices hors bornes pour éviter les
        IndexError pendant l'entraînement ou l'inférence.
        """
        mask = torch.zeros(R, dtype=torch.bool)
        if not subset:
            return mask

        # On ne garde que les indices valides dans [0, R-1]
        valid = [i for i in subset if 0 <= i < R]
        if not valid:
            return mask

        idx = torch.tensor(valid, dtype=torch.long)
        mask[idx] = True
        return mask

    def coherence_of_subgraph(
        self,
        local_scores: torch.Tensor,   # [R]
        subset_mask: torch.Tensor,    # [R] bool
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        µ(Ω | C) = Σ_{r∈Ω} µ_local(r | C) (optionnellement normalisé).
        """
        mask = subset_mask.bool()
        if mask.sum() == 0:
            return torch.zeros((), device=local_scores.device)

        s = (local_scores * mask.float()).sum()
        if normalize:
            s = s / mask.sum().clamp(min=1.0)
        return s

    # -------------------------
    # 5.3 Actualisation via beam search
    # -------------------------

    def actualize_beam_search(
        self,
        graph: RelationalGraph,
        ctx_vec: torch.Tensor,
        constraint_fn: Optional[Callable[[List[int], int, RelationalGraph], bool]] = None,
    ) -> Tuple[List[int], torch.Tensor]:
        """
        Approximation de Ω* = arg max_Ω⊆G µ(Ω | C) via beam search.
        """
        graph = graph.to_device(self.device)
        _, local_scores = self.compute_local_scores(graph, ctx_vec)

        R = len(graph.relations)
        if R == 0:
            # Pas de relation → cohérence triviale nulle
            return [], torch.tensor(0.0, device=self.device)

        local_scores_detached = local_scores.detach()
        order = torch.argsort(local_scores_detached, descending=True).tolist()

        beam: List[Tuple[List[int], torch.Tensor]] = [
            ([], torch.tensor(0.0, device=self.device))
        ]
        best_subset: List[int] = []
        best_score = torch.tensor(0.0, device=self.device)

        max_len = self.cfg.max_relations_in_subgraph or R
        B = self.cfg.beam_size

        for r_idx in order:
            new_beam: List[Tuple[List[int], torch.Tensor]] = []

            for subset, _ in beam:
                # Option 1 : garder subset
                mask_subset = self._subset_to_mask(R, subset).to(self.device)
                score_subset = self.coherence_of_subgraph(local_scores, mask_subset, normalize=True)
                new_beam.append((subset, score_subset))

                # Option 2 : ajouter r_idx
                if len(subset) < max_len:
                    if constraint_fn is None or constraint_fn(subset, r_idx, graph):
                        cand_subset = subset + [r_idx]
                        mask_cand = self._subset_to_mask(R, cand_subset).to(self.device)
                        score_cand = self.coherence_of_subgraph(local_scores, mask_cand, normalize=True)
                        new_beam.append((cand_subset, score_cand))

                        if score_cand > best_score:
                            best_score = score_cand
                            best_subset = cand_subset

            new_beam.sort(key=lambda x: x[1].item(), reverse=True)
            beam = new_beam[:B]

        return best_subset, best_score

    # -------------------------
    # 5.4 Relaxation Gumbel-Softmax
    # -------------------------

    def _gumbel_binary_sample(self, logits: torch.Tensor, tau: float) -> torch.Tensor:
        """
        Logistic-Concrete (relaxed Bernoulli) pour chaque relation.
        """
        eps = 1e-9
        u = torch.rand_like(logits)
        g = -torch.log(-torch.log(u + eps) + eps)
        z = torch.sigmoid((logits + g) / tau)
        return z

    def soft_coherence(
        self,
        graph: RelationalGraph,
        ctx_vec: torch.Tensor,
        tau: Optional[float] = None,
    ) -> torch.Tensor:
        """
        µ_soft(Ω~ | C) = Σ_r z~_r µ_local(r | C),
        z~_r échantillonné via logistic-concrete.
        """
        graph = graph.to_device(self.device)
        _, local_scores = self.compute_local_scores(graph, ctx_vec)
        if len(graph.relations) == 0:
            return torch.tensor(0.0, device=self.device)

        tau = tau or self.cfg.gumbel_tau
        z_tilde = self._gumbel_binary_sample(local_scores, tau)
        mu_soft = (z_tilde * local_scores).sum() / (z_tilde.sum().clamp(min=1.0))
        return mu_soft

    # -------------------------
    # 5.5 Objectif d'entraînement L_coherence
    # -------------------------

    def coherence_loss(
        self,
        graph: RelationalGraph,
        ctx_vec: torch.Tensor,
        gold_subset: List[int],
        negative_subset: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Objectif de cohérence avec contrôle robuste des indices de relations.

        L_coherence = L_maximize + λ L_margin
        L_maximize  = 1 - µ(Ω_gold | C)
        L_margin    = max(0, µ(Ω~ | C) - µ(Ω* | C) + δ)

        Les indices de gold_subset et negative_subset sont automatiquement
        tronqués à [0, R-1] pour éviter les IndexError si le graphizer
        retourne moins de relations que prévu.
        """
        graph = graph.to_device(self.device)
        _, local_scores = self.compute_local_scores(graph, ctx_vec)
        R = len(graph.relations)

        if R == 0:
            # Pas de relations -> pas de perte utile
            return torch.tensor(0.0, device=self.device)

        # Filtrage des indices gold hors bornes
        gold_subset_filtered = [i for i in gold_subset if 0 <= i < R]
        if not gold_subset_filtered:
            # Si aucun indice gold valide, on ne peut pas définir une cible claire.
            # On renvoie une perte nulle pour ignorer cet exemple.
            print("⚠️  Aucun indice gold valide pour ce graphe, sample ignoré dans coherence_loss.")
            return torch.tensor(0.0, device=self.device)

        mask_gold = self._subset_to_mask(R, gold_subset_filtered).to(self.device)
        mu_gold = self.coherence_of_subgraph(local_scores, mask_gold, normalize=True)

        # Sous-graphe actualisé Ω*
        pred_subset, mu_pred = self.actualize_beam_search(graph, ctx_vec)

        # Sous-graphe négatif Ω~
        if negative_subset is None:
            perm = list(range(R))
            random.shuffle(perm)
            length = max(1, R // 2)
            negative_subset = perm[:length]

        neg_subset_filtered = [i for i in negative_subset if 0 <= i < R]
        if not neg_subset_filtered:
            # Si on ne parvient pas à définir un négatif valide, on en tire un trivial
            neg_subset_filtered = [i for i in range(R) if i not in gold_subset_filtered]
            if not neg_subset_filtered:
                # Cas extrême : tout est gold → pas de marge informative
                neg_subset_filtered = [0]

        mask_neg = self._subset_to_mask(R, neg_subset_filtered).to(self.device)
        mu_neg = self.coherence_of_subgraph(local_scores, mask_neg, normalize=True)

        L_max = 1.0 - mu_gold
        margin = mu_neg - mu_pred + self.cfg.margin_delta
        L_margin = torch.clamp(margin, min=0.0)

        loss = L_max + self.cfg.lambda_margin * L_margin
        return loss


# ---------------------------------------------------------------------------
# 6. Pipeline complet RCE-LLM + LM Studio
# ---------------------------------------------------------------------------

class RCELLMPipeline:
    """
    Pipeline complet :

        x (texte)
          ├─ Graphizer LLM  -> G(x)
          ├─ Embedding LLM  -> C
          └─ RCE            -> µ(Ω* | C), Ω*, explication
    """

    def __init__(
        self,
        lm_base_url: str,
        lm_api_key: str,
        graphizer_model: str,
        embedding_model: str,
        rce_config: RCEConfig,
    ):
        self.client = get_lmstudio_client(base_url=lm_base_url, api_key=lm_api_key)
        self.graphizer = GraphizerLLM(self.client, graphizer_model)
        self.context_encoder = ContextEncoderLLM(self.client, embedding_model)
        self.rce = RelationalCoherenceEngine(rce_config)

        if os.path.exists("rce_weights.pt"):
            self.rce.load_state_dict(torch.load("rce_weights.pt", map_location=self.rce.device))
            print("✔️ RCE weights loaded from rce_weights.pt")
        else:
            print("⚠️ Aucun fichier rce_weights.pt trouvé — RCE non entraîné.")

    def _build_graph_from_llm(self, text: str) -> RelationalGraph:
        data = self.graphizer.graphize(text)
        nodes = [Node(**n) for n in data["nodes"]]
        relations = [Relation(**r) for r in data["relations"]]

        if len(relations) == 0:
            # Graphe sans relation -> features vide (0, d_rel)
            rel_feats = torch.zeros(0, self.rce.cfg.rel_dim)
            return RelationalGraph(nodes, relations, rel_feats)

        triplets = [
            f"{nodes[r.head].label} {r.rtype} {nodes[r.tail].label}"
            for r in relations
        ]

        feats = []
        for t in triplets:
            emb = self.context_encoder.encode(t)  # [d_ctx]
            feats.append(emb)
        rel_feats = torch.stack(feats, dim=0)  # [R, d_ctx]

        return RelationalGraph(nodes, relations, rel_feats)

    def score_text(self, text: str) -> float:
        graph = self._build_graph_from_llm(text)
        if len(graph.relations) == 0:
            print("⚠️  Aucune relation dans le graphe -> score 0.0")
            return 0.0

        ctx = self.context_encoder.encode(text)
        subset, score = self.rce.actualize_beam_search(graph, ctx)
        print("Sous-graphe actualisé (indices de relations) :", subset)
        return float(score.item())

    def explain_relations(self, text: str) -> None:
        graph = self._build_graph_from_llm(text)
        if len(graph.relations) == 0:
            print("⚠️  Aucune relation à expliquer (graphe vide).")
            return

        ctx = self.context_encoder.encode(text)
        _, local_scores = self.rce.compute_local_scores(graph, ctx)

        print("\nRelations & scores µ_local(r | C) :")
        for rel, s in zip(graph.relations, local_scores):
            print(
                f"[Rel {rel.id}] {rel.rtype}("
                f"{rel.head}->{rel.tail}) = {s.item():.4f}"
            )

    def train_step(
        self,
        text: str,
        gold_subset: List[int],
        optimizer: torch.optim.Optimizer,
    ) -> float:
        self.rce.train()
        graph = self._build_graph_from_llm(text)
        if len(graph.relations) == 0:
            print("⚠️  Graphe sans relations, train_step ignoré.")
            return 0.0

        ctx = self.context_encoder.encode(text)

        optimizer.zero_grad()
        loss = self.rce.coherence_loss(graph, ctx, gold_subset)
        loss.backward()
        optimizer.step()
        return float(loss.item())


# ---------------------------------------------------------------------------
# 7. Main de test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ➤ Paramètres adaptés à ta config
    LM_BASE_URL = "http://localhost:1234/v1"
    LM_API_KEY = "lm-studio"

    GRAPHIZER_MODEL = "openai/gpt-oss-120b"
    EMBEDDING_MODEL = "bge-m3"
    EMBEDDING_DIM = 1024  # bge-m3 -> 1024

    text = (
        "Alice court 3 km, ce qui équivaut à 3000 m. "
        "Bob court 5 km, donc plus que 3000 m."
    )

    RCE_CFG = RCEConfig(
        rel_dim=EMBEDDING_DIM,
        ctx_dim=EMBEDDING_DIM,
        num_modules=5,
        beam_size=8,
        max_relations_in_subgraph=None,
        margin_delta=0.1,
        lambda_margin=1.0,
        gumbel_tau=0.5,
        use_gpu=True,
    )

    pipeline = RCELLMPipeline(
        lm_base_url=LM_BASE_URL,
        lm_api_key=LM_API_KEY,
        graphizer_model=GRAPHIZER_MODEL,
        embedding_model=EMBEDDING_MODEL,
        rce_config=RCE_CFG,
    )

    # Inference simple
    score = pipeline.score_text(text)
    print(f"\nScore de cohérence RCE-LLM : {score:.4f}")

    pipeline.explain_relations(text)
