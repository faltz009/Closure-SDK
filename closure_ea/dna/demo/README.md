# Closure DNA Demos

This folder holds ready-to-open demo databases for the local web UI.

- source data lives in `closure_ea/dna/demo_data`
- built databases live in `closure_ea/dna/demo/databases`

Use:

```bash
python3.13 -m closure_ea.dna build-demo-db all
python3.13 -m closure_ea.dna web
```

Then open a demo database from the built-in database list in the web UI.
