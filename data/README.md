Place locally built manifests here. Do not commit real CSVs or ECG files.

```text
data/mimic_manifests/stage_{0..10}_{train,val,test}.csv
data/echonext_manifests/stage_{0..6}_{train,val,test}.csv
data/echonext_manifests/future_2022_test.csv   # optional 2022 hold-out
```

Required columns: `sample_id`, `patient_id`, `label`, `file_path`.
Optional: `year`, `sample_index` (for stacked `.npy` volumes).

`file_path` must point to waveforms on your machine. Patients must be disjoint across train/val/test within each stage. See `examples/manifest_template.csv`.
