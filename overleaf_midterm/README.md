# Archived Midterm Overleaf Package

This folder is the submitted midterm report package. The final report should be developed separately rather than by continuing to update this archive.

Upload this folder's contents to Overleaf:

- `main.tex`
- `acl.sty`
- `acl_natbib.bst`

Set `main.tex` as the main document.

The report uses an inline `thebibliography` environment, so no `.bib` file is required. `acl_natbib.bst` is included because the ACL style references it and Overleaf may look for it during auxiliary compilation steps.

The ACL style is loaded with:

```tex
\usepackage[preprint]{acl}
```

This works because `acl.sty` is in the same folder as `main.tex`.
