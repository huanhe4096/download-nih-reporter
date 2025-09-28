# v4 test

```python
return f"{fiscal_year} | {agency_ic_admin} | {project_title}"
```

tSNE

Looks nice.

# v5 test

Just change the order of fiscal_year

```bash
python embedding_pipeline.py all \
  --batch_size 64 \
  --embedding_output nih-reporter-grants.embedding.v5.npy \
  --embd_output=nih-reporter-grants.embd.v5.npy \
  --final_outpu grants.v5.tsv