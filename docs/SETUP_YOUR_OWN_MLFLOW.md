# Run the notebooks on your own Databricks MLflow

During the MICAI 2026 workshop everyone logged to a shared MLflow server prepared by the instructor. This guide sets
up **your own** free MLflow server on Databricks, so the notebooks work exactly as they did in the session.

**Time:** about 15 minutes, once. **You need:** a Google account (for Colab) and an email address (for Databricks).

| Step | What you do | Where |
|---|---|---|
| 1 | Create a free Databricks account | Databricks website |
| 2 | Note your workspace address | Databricks |
| 3 | Create a personal access token | Databricks → Settings |
| 4 | Fill in your config file | Your computer |
| 5 | Give the config to Colab as a secret | Google Colab |
| 6 | Run notebook 00 once | Google Colab |
| 7 | Run notebooks 01, 02 and 03 | Google Colab |

> Prefer to try first without any account? Run the notebooks and leave **CONFIG_LINK** empty. With no config they
> work **offline** with a local MLflow store inside Colab; nothing is saved after the session ends.

---

## 1. Create a free Databricks account

1. Go to the **Databricks Free Edition** sign-up page (search for "Databricks Free Edition").
2. Sign up with Google, Microsoft or an email code. No credit card is needed.
3. Wait until your workspace opens (one or two minutes).

## 2. Note your workspace address

Copy the address from your browser's address bar, **up to `.com` only**, for example:

```text
https://dbc-a1b2c3d4-e5f6.cloud.databricks.com
```

This is your `databricks_host`.

## 3. Create a personal access token

The token is the password the notebooks use to log runs to your workspace. Treat it like a password.

1. In Databricks, click your **user icon** (top right) → **Settings** → **Developer**.
2. Next to **Access tokens**, click **Manage** → **Generate new token**.
3. **Comment:** `MICAI MLOps notebooks`. **Lifetime:** for example **90** days.
4. Click **Generate** and **copy the token now**: it is shown only once. Then click **Done**.

This is your `databricks_token`. If you lose it, delete it in the same page and generate a new one.

## 4. Fill in your config file

The repository contains a template, `config.template.json`. **Never write your token in the template.** Make a
copy called `config.json` and edit only the copy: `config.json` is listed in `.gitignore`, so Git never adds it to a
commit and your token stays on your computer.

**If you cloned the repository** (from its folder):

```bash
# macOS / Linux / Git Bash
cp config.template.json config.json
# Windows PowerShell
Copy-Item config.template.json config.json
```

**If you did not clone it**, download the template from GitHub (**config.template.json → Download raw file**) and
save it as `config.json` anywhere outside a Git folder.

Open `config.json` in a text editor and fill in:

| Field | Value |
|---|---|
| `schema_version` | Keep `1` |
| `workshop` | Any label; not used by the code |
| `databricks_host` | Your workspace address from step 2 |
| `databricks_token` | Your token from step 3 |
| `experiment_root` | Keep `/Shared/MICAI_MLOps_2026` (the folder notebook 00 creates) |
| `official_data_run_id` | Leave empty `""`: the notebooks find the official datasets by themselves |
| `expires` | The token's expiry date, for example `2027-02-03`, or leave empty `""` |

Example (the token is a placeholder):

```json
{
  "schema_version": 1,
  "workshop": "MICAI 2026 MLOps Workshop",
  "databricks_host": "https://dbc-a1b2c3d4-e5f6.cloud.databricks.com",
  "databricks_token": "dapi-paste-your-own-token-here",
  "experiment_root": "/Shared/MICAI_MLOps_2026",
  "official_data_run_id": "",
  "expires": "2027-02-03"
}
```

**Check that Git ignores your copy** (in a clone):

```bash
git check-ignore -v config.json      # prints the .gitignore line that ignores it
git status                           # config.json must NOT be listed
```

If you edited `config.template.json` by mistake, undo it with `git restore config.template.json`. If you really
want to keep a modified template locally, tell Git to stop tracking your changes to it with
`git update-index --skip-worktree config.template.json` (undo with `--no-skip-worktree`). Never commit, push or
share a file that contains your token.

## 5. Give the config to Colab as a secret

Colab Secrets keep the token in your Google account instead of in a file or a link. Each notebook only reads a secret
after you allow it.

1. Open any of the notebooks in Colab (links in the README).
2. Click the **key icon** 🔑 in the left sidebar → **Add new secret**.
3. **Name:** `MICAI_CONFIG`. **Value:** paste the **whole content** of your `config.json`, from `{` to `}`.
4. Switch on **Notebook access** for this notebook. Repeat this switch once in each notebook you open (00 to 03);
   Colab may also ask "Grant access?" when the cell runs: choose **Grant access**.

In every notebook, leave **CONFIG_LINK empty**: the setup cell reads the secret. The token is never printed.

**Alternatives** (no secret):
- **Upload the file:** in Colab, **Files** (folder icon) → **Upload** `config.json`, then type `/content/config.json`
  in **CONFIG_LINK**. Uploaded files are deleted when the Colab session ends.
- **Running on your computer** (Jupyter or VS Code, from the `notebooks/` folder): type the path to your
  `config.json` in **CONFIG_LINK**, or set the environment variable `MICAI_CONFIG` to that path.

Do not share your config through a public link (for example a Google Drive file shared with "Anyone with the link"):
anyone who finds the link could use your token.

## 6. Run notebook 00 once

Open **00 · Set up your own MLflow workspace** and run all cells. In about one minute it:

1. creates the folder `/Shared/MICAI_MLOps_2026` and four experiments: `00_official_datasets`, `01_data_cleaning`,
   `02_training`, `03_drift_monitoring`;
2. publishes the **official dataset versions** (v1 drop, v2 impute, the fixed holdout) to `00_official_datasets`,
   copied from the processed files in `data/wine-quality/official/`;
3. publishes the **reference ("production") model** to `02_training`, the model monitored in notebook 3.

You should see ✅ lines and links to your four experiments. Running it again is safe: finished steps are skipped.

**From a terminal instead of Colab** (in a clone, Python 3.10 or newer):

```bash
pip install -r requirements-dev.txt
python scripts/setup_workspace.py --config config.json
```

**Prefer to create the experiments by hand?** In Databricks: **Workspace** (left sidebar) → **Shared** → **Create**
→ **Folder** → `MICAI_MLOps_2026`. Then **Experiments** → **Create** and give each experiment its full path, for
example `/Shared/MICAI_MLOps_2026/00_official_datasets` (repeat for the other three). You still need notebook 00 for
the official datasets and the reference model; it detects the experiments you created.

## 7. Run notebooks 01, 02 and 03

Open each notebook, leave **CONFIG_LINK** empty, type a nickname, and run the cells from top to bottom. The setup cell
should print `Config read from Colab Secret MICAI_CONFIG and checked` and `Connected to MLflow on Databricks`. Your runs
then appear in your experiments: **Experiments** in the Databricks sidebar.

---

## Troubleshooting

| Message in the setup cell | Cause | Fix |
|---|---|---|
| `Working with a LOCAL MLflow store` | No config was found, or it could not be used | Read the line above it: it names the reason |
| `... has no access to it: ... 'Notebook access'` | The secret exists but this notebook can't read it | Key icon 🔑 → switch on **Notebook access** for `MICAI_CONFIG` |
| `fill in ['databricks_host', ...]` | The config still holds the template text | Edit your copy with your own values (step 4) |
| `the access key was rejected` | Token expired, revoked or mistyped | Generate a new token (step 3) and update the secret |
| `cannot reach https://...` | Wrong address or no internet | Copy the address again, up to `.com` only (step 2) |
| `the config expired on ...` | The `expires` date has passed | Create a new token and update `expires`, or leave it empty |
| `the Colab Secret MICAI_CONFIG is not valid JSON` | Part of the file was not pasted | Paste the whole file, including `{` and `}` |
| `could not create the folder ...` | No permission to create it | Create it by hand (step 6, last paragraph) |
| `No official datasets found in MLflow` | Notebook 00 was not run | Run notebook 00 once; meanwhile the repository copy is used |

## When you are done

Delete the token in **Settings → Developer → Access tokens** and remove the `MICAI_CONFIG` secret in Colab. Your
experiments and runs stay in your Databricks workspace.
