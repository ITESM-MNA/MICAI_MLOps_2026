# Wine Quality (white wine)

| File | Content |
|---|---|
| `original.csv` | The UCI *Wine Quality* white-wine data (4,898 rows, 12 columns). Used only as a baseline in notebook 1. |
| `modified.csv` | A deliberately corrupted copy (4,995 rows, 13 columns): padded spaces, placeholder words, impossible values, a noise column and injected duplicates. The input of the cleaning pipeline. |
| `metadata.json` | How `modified.csv` was produced (seed, corruption rates, file hashes). |
| `official/` | The processed official versions (`v1_clean_drop.csv`, `v2_clean_impute.csv`, `holdout.csv`) published to MLflow by notebook 00. Read by notebooks 02 and 03 when MLflow has no official run. |
| `official_manifest.json` | Rows and SHA-256 fingerprints of the official dataset versions used by notebooks 2 and 3. |

Source: P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis (2009). *Modeling wine preferences by data mining
from physicochemical properties.* Decision Support Systems 47(4), 547–553. UCI Machine Learning Repository.

**License:** these files are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
([doi:10.24432/C56S3T](https://doi.org/10.24432/C56S3T)), not under the repository's MIT License. `modified.csv`
and `official/` are modified versions prepared for this workshop.
