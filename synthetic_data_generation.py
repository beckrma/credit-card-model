import pandas as pd
import random
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.evaluation import run_diagnostic
from sdv.evaluation import evaluate_quality
from sdv.metadata import Metadata
from sdv.sampling import Condition

def load_data():
    df = pd.read_csv("archive/creditcard.csv")
    metadata = Metadata.load_from_json("archive/creditcard_metadata.json")
    return df, metadata

def synthesize_data(real_data, metadata, demo_condition=None, more_non_fraud=None):
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(real_data)
    if demo_condition and more_non_fraud:
        synthetic_data = synthesizer.sample_from_conditions(conditions=[demo_condition, more_non_fraud])
    else:
        synthetic_data = synthesizer.sample(num_rows=100000)
    return synthetic_data

def validating_data(real_data, synthetic_data, metadata):
    diagnostic = run_diagnostic(
        real_data=real_data,
        synthetic_data=synthetic_data,
        metadata=metadata
    )

    quality_report = evaluate_quality(
        real_data,
        synthetic_data,
        metadata
    )

def get_demo_condition():
    more_fraud = Condition(
        num_rows=500,
        column_values={"Class": 1}
    )

    more_non_fraud = Condition(
        num_rows = 9500,
        column_values={"Class": 0}
    )
    return more_fraud, more_non_fraud

def modify_demo_data(synthetic_data):
    demo_data = synthetic_data

    fraud = demo_data["Class"] == 1

    for index in demo_data.index[fraud]:
        features = random.sample(range(1, 29), 3)

        for feature in features:
            column = f"V{feature}"
            demo_data.loc[index, column] += random.uniform(-5, 5)

        demo_data.loc[index, "Amount"] *= random.uniform(3, 10)

    return demo_data

if __name__ == "__main__":
    choice_input = input("Demo condition (1) or Full synthesis (2)? ")
    if choice_input == "1":
        demo_condition, more_non_fraud = get_demo_condition()
    else:
        demo_condition = None
    real_data, metadata = load_data()
    synthetic_data = synthesize_data(real_data, metadata, demo_condition, more_non_fraud)
    validating_data(real_data, synthetic_data, metadata)
    if choice_input == "1":
        synthetic_data = modify_demo_data(synthetic_data)
        synthetic_data.to_csv("archive/synthetic_creditcard_demo.csv", index=False)
    else:
        synthetic_data.to_csv("archive/synthetic_creditcard_realistic.csv", index=False)