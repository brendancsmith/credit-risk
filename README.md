# Credit Risk Modeling using XGBoost

## Overview

This project uses the **Lending Club Loan Dataset** from Kaggle, which contains detailed information about loans issued through the Lending Club platform, including borrower details, loan attributes, and payment status.

## Installation

1. Clone the repository

```bash
git clone https://github.com/brendancsmith/credit-risk-modeling.git
```

2. Install Python dependencies

```bash
poetry init
poetry install
```

3. Download the [dataset](https://www.kaggle.com/datasets/wordsforthewise/lending-club) from Kaggle

```bash
brew install kaggle

cd data
kaggle datasets download -d wordsforthewise/lending-club --unzip
mv accepted_2007_to_2018Q4.csv.gz raw/
rm -r *_2007_to_2018*
cd -
```

## Usage

See the `notebooks` folder for a detailed analysis of the dataset and the modeling process.

## Development

```bash
pre-commit install
pre-commit install-hooks
```

```bash
nbstripout --install
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
