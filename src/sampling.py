# src/sampling.py

import pandas as pd

TARGET_SIZE = 12000
RANDOM_STATE = 42


def create_stratified_sample(
    input_path="data/filtered_complaints.csv",
    output_path="data/sample_complaints.csv",
):
    df = pd.read_csv(input_path)

    sample_df = (
        df.groupby("product", group_keys=False)
        .apply(
            lambda x: x.sample(
                n=max(
                    1,
                    round(len(x) / len(df) * TARGET_SIZE)
                ),
                random_state=RANDOM_STATE,
            )
        )
        .reset_index(drop=True)
    )

    sample_df.to_csv(output_path, index=False)

    print(f"Sample size: {len(sample_df)}")

    return sample_df


if __name__ == "__main__":
    create_stratified_sample()