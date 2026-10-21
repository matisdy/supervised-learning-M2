# Data snapshot

`diabetes130us.csv.gz` contains 101,766 hospital stays from OpenML dataset 4541, version 1 (Diabetes 130-US Hospitals). Source: https://openml.org/data/v1/download/1798768/Diabetes130US.arff. The adjacent JSON records provenance, column types, categorical levels and the snapshot checksum.

Use `src.data.load_data()` to preserve row order, categorical types and literal `None`/`NO` values; the loader verifies the checksum and maps the target to recorded readmission at either delay. No network download is required.
