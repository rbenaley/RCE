import json
from rce import RCELLMPipeline, RCEConfig

LM_BASE_URL = "http://localhost:1234/v1"
LM_API_KEY = "lm-studio"

GRAPHIZER_MODEL = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "bge-m3"
EMBEDDING_DIM = 1024

RCE_CFG = RCEConfig(
    rel_dim=EMBEDDING_DIM,
    ctx_dim=EMBEDDING_DIM,
    num_modules=5,
    beam_size=8,
    use_gpu=True,
)

pipeline = RCELLMPipeline(
    lm_base_url=LM_BASE_URL,
    lm_api_key=LM_API_KEY,
    graphizer_model=GRAPHIZER_MODEL,
    embedding_model=EMBEDDING_MODEL,
    rce_config=RCE_CFG,
)

DATASET_FILE = "rce_dataset.jsonl"


def annotate(text):
    graph = pipeline._build_graph_from_llm(text)

    print("\n=== TEXTE ===")
    print(text)

    print("\n=== NODES ===")
    for n in graph.nodes:
        print(f"[Node {n.id}] {n.label}")

    print("\n=== RELATIONS ===")
    for r in graph.relations:
        print(f"[Rel {r.id}] {graph.nodes[r.head].label} --{r.rtype}--> {graph.nodes[r.tail].label}")

    print("\nIndique les relations cohérentes (gold), ex: 0,1,3")
    gold = input("gold_subset = ").strip()

    if gold == "":
        gold_subset = []
    else:
        gold_subset = [int(x) for x in gold.split(",")]

    sample = {
        "text": text,
        "gold_subset": gold_subset,
    }

    with open(DATASET_FILE, "a") as f:
        f.write(json.dumps(sample) + "\n")

    print("\n✔️ Ajouté au dataset.")


if __name__ == "__main__":
    # Exemple : tu peux enchaîner plusieurs textes ici :
    annotate("Alice court 3 km, ce qui équivaut à 3000 m. Bob court 5 km, donc plus que 3000 m.")
