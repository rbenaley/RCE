# train_rce_offline.py

import json
import random
import torch
from torch.optim import Adam

from rce import (
    RelationalCoherenceEngine,
    RelationalGraph,
    Node,
    Relation,
    RCEConfig,
)


EMBEDDING_DIM = 1024  # bge-m3
DATASET_FILE = "graphs_dataset.jsonl"


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


def load_graph_dataset(path: str):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            data.append(sample)
    return data


def sample_to_graph_and_ctx(sample) -> (RelationalGraph, torch.Tensor, list):
    # Reconstruire Node / Relation
    nodes = [Node(**n) for n in sample["nodes"]]
    relations = [Relation(**r) for r in sample["relations"]]

    rel_features = torch.tensor(sample["rel_features"], dtype=torch.float32)
    ctx_vec = torch.tensor(sample["ctx_vec"], dtype=torch.float32)

    graph = RelationalGraph(nodes=nodes, relations=relations, rel_features=rel_features)
    gold_subset = sample.get("gold_subset", [])

    return graph, ctx_vec, gold_subset


def main(epochs: int = 10, lr: float = 1e-4):
    dataset = load_graph_dataset(DATASET_FILE)
    if not dataset:
        print(f"⚠️  Dataset vide : {DATASET_FILE}")
        return

    rce = RelationalCoherenceEngine(RCE_CFG)
    optimizer = Adam(rce.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        random.shuffle(dataset)
        total_loss = 0.0
        count = 0

        for sample in dataset:
            graph, ctx_vec, gold_subset = sample_to_graph_and_ctx(sample)

            if len(graph.relations) == 0:
                print("⚠️  Graphe sans relations, sample ignoré.")
                continue

            optimizer.zero_grad()
            loss = rce.coherence_loss(graph, ctx_vec, gold_subset)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

        if count == 0:
            print("⚠️  Aucun sample utilisé à l'epoch", epoch)
            continue

        avg_loss = total_loss / count
        print(f"=== Epoch {epoch} terminé - loss moyenne={avg_loss:.4f} ===")

    torch.save(rce.state_dict(), "rce_weights.pt")
    print("✔️ Poids sauvegardés dans rce_weights.pt")

    # Test sur le premier sample
    print("\n=== Test offline sur le premier sample ===")
    graph, ctx_vec, gold_subset = sample_to_graph_and_ctx(dataset[0])

    # Scores µ_local
    _, local_scores = rce.compute_local_scores(graph, ctx_vec)
    for rel, s in zip(graph.relations, local_scores):
        print(
            f"[Rel {rel.id}] {rel.rtype}({rel.head}->{rel.tail}) "
            f"µ_local={s.item():.4f}"
        )

    # Coherence du sous-graphe gold
    R = len(graph.relations)
    mask_gold = rce._subset_to_mask(R, gold_subset).to(rce.device)
    mu_gold = rce.coherence_of_subgraph(local_scores.to(rce.device), mask_gold, normalize=True)
    print(f"\nµ(Ω_gold | C) = {mu_gold.item():.4f}")


if __name__ == "__main__":
    main()



