import pandas as pd

TARGET_SIZE = 12000
RANDOM_STATE = 42


def create_stratified_sample(
    input_path="data/processed/filtered_complaints.csv",
    output_path="data/processed/sample_complaints.csv",
):

    df = pd.read_csv(input_path)

    print("Original shape:", df.shape)

    sample_df = (
        df.groupby("Product", group_keys=False)
        .apply(
            lambda x: x.sample(
                n=max(
                    1,
                    round(
                        len(x)
                        / len(df)
                        * TARGET_SIZE
                    )
                ),
                random_state=RANDOM_STATE,
            )
        )
        .reset_index(drop=True)
    )

    sample_df.to_csv(output_path, index=False)

    print("Sample shape:", sample_df.shape)

    return sample_df


if __name__ == "__main__":
    create_stratified_sample()