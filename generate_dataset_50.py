# generate_dataset_50.py

import json

from rce import RCELLMPipeline, RCEConfig

# -----------------------------
# Config LM Studio + modèles
# -----------------------------
LM_BASE_URL = "http://localhost:1234/v1"
LM_API_KEY = "lm-studio"

GRAPHIZER_MODEL = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "bge-m3"
EMBEDDING_DIM = 1024  # bge-m3

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

OUTPUT_FILE = "graphs_dataset.jsonl"


TEXTS = [
    # 1–10 : unités, conversions, arithmétique simple
    "Alice marche 2 km, ce qui correspond à 2000 mètres.",
    "Un litre contient 1000 millilitres.",
    "Une heure dure 60 minutes.",
    "Un jour dure 24 heures.",
    "Un mètre correspond à 100 centimètres.",
    "Un kilogramme équivaut à 1000 grammes.",
    "Un euro vaut 100 centimes.",
    "Une semaine contient 7 jours.",
    "Une année contient 12 mois.",
    "Un demi-litre correspond à 500 millilitres.",

    # 11–20 : comparaisons et quantités
    "Bob a 3 pommes et Jean en a 5, donc Jean en a plus que Bob.",
    "Sarah possède 4 livres, alors que Tom en possède 4 aussi.",
    "Claire a 10 bonbons et en donne 3 à Paul, il lui en reste 7.",
    "Le sac rouge est plus lourd que le sac bleu.",
    "La tour A est plus haute que la tour B.",
    "La température est plus élevée aujourd'hui qu'hier.",
    "Ce mois-ci, l'entreprise a vendu plus de produits que le mois dernier.",
    "La voiture de Marc consomme moins de carburant que celle de Julie.",
    "L'écran de 27 pouces est plus grand que l'écran de 24 pouces.",
    "L'année 2024 a connu plus de jours de pluie que 2023 dans cette ville.",

    # 21–30 : temps / chronologie
    "Paul s'est réveillé avant de prendre son petit-déjeuner.",
    "Après avoir fini son travail, Léa est rentrée chez elle.",
    "Le train est parti puis il a commencé à pleuvoir.",
    "D'abord il a préparé le café, ensuite il a allumé son ordinateur.",
    "Avant de partir en vacances, ils ont réservé un hôtel.",
    "Le film a commencé après les publicités.",
    "La réunion s'est terminée avant le déjeuner.",
    "Il a lu un livre puis il s'est endormi.",
    "Elle a d'abord rangé la cuisine, puis elle a lancé une lessive.",
    "Le soleil s'est couché après la fin du match.",

    # 31–40 : coref / entailment simple
    "Marie a acheté une voiture. Elle la conduit tous les jours.",
    "Lucas a adopté un chien, et l'animal dort dans sa chambre.",
    "Sophie a perdu ses clés, elle les cherche depuis ce matin.",
    "Pierre a cassé un verre, il l'a fait tomber par terre.",
    "L'entreprise a embauché un nouveau développeur, il commence lundi.",
    "Julie a acheté un ordinateur portable, il est plus léger que son ancien PC.",
    "Thomas a écrit un livre, et son ouvrage a été publié en mai.",
    "Le professeur a donné un exercice, les élèves doivent le rendre demain.",
    "Marc a prêté sa voiture à son frère, il la récupérera ce soir.",
    "Emma a acheté un nouveau téléphone, elle l'utilise pour prendre des photos.",

    # 41–50 : causalité, logique, un peu de pièges doux
    "Il pleut, donc la route est mouillée.",
    "Le verre est tombé par terre, alors il s'est brisé.",
    "Il a oublié son parapluie et s'est retrouvé trempé.",
    "La batterie est vide, donc le téléphone ne s'allume plus.",
    "Il a beaucoup trop mangé, c'est pour cela qu'il a mal au ventre.",
    "La fenêtre était ouverte, donc la pièce était froide.",
    "Le serveur est en panne, c'est pourquoi le site est indisponible.",
    "La lumière est allumée parce que quelqu'un est dans la pièce.",
    "Le moteur n'a pas été entretenu, alors la voiture est tombée en panne.",
    "La clé ne tourne pas dans la serrure, car ce n'est pas la bonne clé.",
]


def build_pipeline() -> RCELLMPipeline:
    pipeline = RCELLMPipeline(
        lm_base_url=LM_BASE_URL,
        lm_api_key=LM_API_KEY,
        graphizer_model=GRAPHIZER_MODEL,
        embedding_model=EMBEDDING_MODEL,
        rce_config=RCE_CFG,
    )
    return pipeline


def main():
    pipeline = build_pipeline()

    print(f"Génération d'un dataset à partir de {len(TEXTS)} textes...")
    count_ok = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        for idx, text in enumerate(TEXTS):
            print(f"\n[{idx+1}/{len(TEXTS)}] Traitement du texte :")
            print(text)

            # Construction graphe + contexte via LM Studio
            graph = pipeline._build_graph_from_llm(text)
            ctx_vec = pipeline.context_encoder.encode(text)

            if len(graph.relations) == 0:
                print("⚠️  Aucune relation trouvée, sample ignoré.")
                continue

            # Supervision automatique : on considère toutes les relations comme gold
            R = len(graph.relations)
            gold_subset = list(range(R))

            # Conversion des données en formes sérialisables
            rel_features = graph.rel_features.cpu().tolist()
            ctx_list = ctx_vec.cpu().tolist()

            sample = {
                "text": text,
                "nodes": [
                    {"id": n.id, "label": n.label, "type": n.type}
                    for n in graph.nodes
                ],
                "relations": [
                    {"id": r.id, "head": r.head, "tail": r.tail, "rtype": r.rtype}
                    for r in graph.relations
                ],
                "rel_features": rel_features,  # [[...], ...]
                "ctx_vec": ctx_list,           # [...]
                "gold_subset": gold_subset,
            }

            f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")
            count_ok += 1
            print(f"✔️ Sample ajouté (relations: {R})")

    print(f"\nTerminé. {count_ok} samples écrits dans {OUTPUT_FILE}.")


if __name__ == "__main__":
    main()

