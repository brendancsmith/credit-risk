import pandas as pd


def download_dataset():
    # Load dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
    columns = [
        "Status_of_existing_checking_account",
        "Duration_in_month",
        "Credit_history",
        "Purpose",
        "Credit_amount",
        "Savings_account/bonds",
        "Present_employment_since",
        "Installment_rate_in_percentage_of_disposable_income",
        "Personal_status_and_sex",
        "Other_debtors/guarantors",
        "Present_residence_since",
        "Property",
        "Age_in_years",
        "Other_installment_plans",
        "Housing",
        "Number_of_existing_credits_at_this_bank",
        "Job",
        "Number_of_people_being_liable_to_provide_maintenance_for",
        "Telephone",
        "Foreign_worker",
        "Target",
    ]
    df = pd.read_csv(url, delimiter=" ", header=None, names=columns)

    return df
