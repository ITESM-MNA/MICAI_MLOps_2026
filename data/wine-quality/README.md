# Wine Quality (white wine)

| File | Content |
|---|---|
| `original.csv` | The UCI *Wine Quality* white-wine data (4,898 rows, 12 columns). Used only as a baseline in notebook 1. |
| `modified.csv` | A deliberately corrupted copy (4,995 rows, 13 columns): padded spaces, placeholder words, impossible values, a noise column and injected duplicates. The input of the cleaning pipeline. |
| `metadata.json` | How `modified.csv` was produced (seed, corruption rates, file hashes). |
| `official_manifest.json` | Rows and SHA-256 fingerprints of the official dataset versions used by notebooks 2 and 3. |

Source: P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis (2009). *Modeling wine preferences by data mining
from physicochemical properties.* Decision Support Systems 47(4), 547–553. UCI Machine Learning Repository.
