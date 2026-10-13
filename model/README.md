# Models

| Folder | Contents | Included in the ZIP? |
|---|---|---|
| `baseline/` | TF-IDF vectorizer + Logistic Regression (`joblib`), `metadata.json` | Yes (trained, ~2 MB) |
| `word2vec/` | gensim Word2Vec model + mean-pooled Logistic Regression (experiment only) | Yes (~4 MB) |
| `contracts_bert_risk/` | fine-tuned Contracts-BERT: weights, config, tokenizer, `label_mapping.json`, `training_metadata.json` | **No - generate locally** |

The fine-tuned Contracts-BERT model (~440 MB) is **not** included and was not fully trained when this package was built. Create it with:

```powershell
python training/train_model.py
python training/evaluate_model.py
```

Until it exists the app shows "Trained model not found. Please train the model first or use Demo Mode." and offers the TF-IDF baseline (clearly labelled) and Demo Mode.

`_smoke_test/` (created by `train_model.py --smoke-test`) is only a pipeline check and is never used as a real model; if you copy it into `contracts_bert_risk/`, the app shows a smoke-test warning.
