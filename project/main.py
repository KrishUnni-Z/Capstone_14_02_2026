from system1.llm_parser import parse_user_input
from system1.llm_generator import generate_output

from system2.preprocess import clean_data
from system2.features import create_features
from system2.rules import score_rules
from system2.pipeline import prepare_model_input

from system3.etl import load_and_transform
from system3.embeddings import create_embeddings
from system3.faiss_index import build_index, search

from system3_model.predict import predict_scores


def main(user_text):

    # 1. System 1 → parse input
    structured = parse_user_input(user_text)

    # 2. System 3 → validate + structure
    structured = load_and_transform()

    # 3. System 2 → clean + features
    clean = clean_data(structured)
    features = create_features(clean)

    # 4. Optional FAISS (context retrieval)
    vectors = create_embeddings(features)
    index = build_index(vectors)
    similar = search(index, vectors[:1])   # example

    # 5. Rule-based scoring
    rule_scores = score_rules(features)

    # 6. ML scoring
    model_scores = predict_scores(features)

    # 7. Combine scores
    final_scores = {**model_scores, **rule_scores}

    # 8. System 1 → explanation
    output = generate_output(final_scores, context=similar)

    return output


if __name__ == "__main__":
    print(main("Which goals are at risk?"))
