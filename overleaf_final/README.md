# Final Report Overleaf Package

Upload this folder's contents to Overleaf:

- `main.tex`
- `acl.sty`
- `acl_natbib.bst`

Set `main.tex` as the main document.

The report uses an inline `thebibliography` environment, so no `.bib` file is required. `acl_natbib.bst` is included because the ACL style references it during compilation.

The current GitHub-facing exported report is:

- `Beyond_Rewrite_Accuracy_Testing_Logical_Consistency_in_Knowledge_Editing_Final_Report.pdf`

Older root-level `Beyond_Rewrite_Accuracy*Final_Report*.pdf` exports, including `*_OLD_5-22.pdf`, are stale local drafts and should not be used for grading/reference.

The ACL style is loaded with:

```tex
\usepackage[preprint]{acl}
```
