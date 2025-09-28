# Download NIH RePORTER data

First, install deps

```bash
pip install -r requirements.txt
```

Download all projects:

```bash
python download_nih_projects.py download
```

If everything works, the downloaded JSON files should be saved at `./cache` folder. 
Running `python download_nih_project.py stat` will show something similar to this:

```
🔍 Download Statistics:
- Total combinations: 2296
- Completed combinations: 2295
- Total projects found: 2,719,706
- Progress: 100.0%
```

## Embedding and Dimensionality Reduction

Generate embeddings and reduce dimensions to 2D:

```bash
python embedding_pipeline.py all --reduce_method opentsne \
  --input ./nih-reporter-grants.tsv \
  --embedding_output ./nih-reporter-grants.embedding.npy \
  --umap_output ./nih-reporter-grants.embd.npy \
  --final_output ./grants.tsv
```

- `--reduce_method` can be `opentsne` (default) or `umap`.
- Install extras if needed: `pip install openTSNE umap-learn`.
