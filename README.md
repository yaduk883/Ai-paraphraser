# AI-Powered Paraphrasing Tool

A Jupyter notebook that rewrites a block of text with a **T5 transformer** (Hugging Face) and then **checks its own output** for meaning, originality, fluency, spelling and grammar before returning it. Everything runs in the notebook with console-style output only (no GUI or web interface).

## Repository contents

| File | Description |
|---|---|
| `ai_paraphrasing_tool.ipynb` | The complete tool, sanity tests, demo and evaluation |
| `README.md` | This file |

## How it works

```
input text -> paragraphs -> sentences (spaCy sentencizer)
                               |
            T5 generates about 7 candidates per sentence
            (beam search + nucleus sampling)
                               |
        clean-up: safe grammar fixes, remove duplicates and exact copies
                               |
   reject if numbers changed or meaning similarity < 0.80
                               |
   rank the rest: 0.40 meaning + 0.20 originality + 0.25 fluency + 0.15 grammar
                               |
   best candidate (or the ORIGINAL sentence if none is safe) -> output + scores
```

| Check | Method |
|---|---|
| Meaning preservation | Cosine similarity of `all-MiniLM-L6-v2` sentence embeddings. Numbers in the text must stay unchanged |
| Originality | Share of new word trigrams compared with the source (BLEU / ROUGE overlap is also reported in the evaluation) |
| Fluency | DistilGPT-2 perplexity relative to the source sentence |
| Spelling | `pyspellchecker`. Words from the source, acronyms and proper nouns are ignored |
| Grammar | Rule checks (repeated words, capitalisation, final punctuation, brackets and quotes, spacing, a/an) and optional LanguageTool |

Design choice: when no candidate is safe, the tool keeps the original sentence rather than return a wrong paraphrase. The share of unchanged sentences is reported.

## Notebook sections

1. Setup (installs and imports)
2. Metrics and neural scorers
3. Grammar, spelling and style checks
4. The paraphraser
5. Sanity tests (use fake models, so no download is needed)
6. Load the models
7. Try it: paste your own text and see the paraphrase with per-sentence scores
8. Evaluation: five sample paragraphs, metric tables, a chart and sample outputs
9. Limitations

## Setup

1. Open the notebook in Jupyter or Google Colab. In Colab, choose a GPU runtime (Runtime > Change runtime type) for faster generation. CPU also works but is slower.
2. Run the cells from top to bottom. The first cell installs the dependencies:
   `transformers, torch, sentencepiece, sentence-transformers, spacy, nltk, pyspellchecker, rouge-score, pandas, matplotlib`
3. The model-loading cell downloads T5-base, MiniLM and DistilGPT-2 from Hugging Face (about 1 GB), so internet access is required.

## Usage

In Section 7, edit the `text` variable and re-run the cell:

```python
result = tool.paraphrase(text)
print(result.paraphrased_text)
print(result.summary)   # similarity, originality, fluency, grammar issues
```

tool = Paraphraser.default(num_candidates=8, min_similarity=0.85)   # stricter meaning check
```

## Evaluation

Section 8 paraphrases five sample paragraphs from different topics and reports, per paragraph and as an average:

- **Accuracy (meaning preservation):** semantic similarity to the source and the share of paragraphs at or above the 0.80 threshold
- **Originality:** trigram novelty (higher means more new wording), and BLEU and ROUGE-L against the source (lower means less copying)
- **Fluency:** perplexity of the output, shown next to the human-written source for reference
- **Correctness:** spelling and grammar issues per 100 words
- **Unchanged sentences:** the share kept as-is because no safe candidate was found

Optionally set `REFERENCES` to a list of human paraphrases (same order as `SAMPLES`) to also get BLEU and ROUGE against references.

Running the notebook also saves `evaluation_results.csv` and `sample_results.md` next to it. These files, and all sample outputs, come from the real models when you run it.



````

Open in `Google Colab` : https://colab.research.google.com/drive/16OBBtvh36-Gw6G0A4iWIwRXek94xX1OF?usp=sharing

