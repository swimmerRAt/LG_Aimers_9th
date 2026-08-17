# PatchTST attribution and modifications

The implementation in `model/patchtst_model.py` is adapted from the official
PatchTST supervised implementation:

- Project: `yuqinie98/PatchTST`
- Source: https://github.com/yuqinie98/PatchTST
- Inspected commit: `204c21efe0b39603ad6e2ca640ef5896646ab1a9`
- Paper: *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers*
- Upstream license: Apache License 2.0

Project-specific changes:

- reduced model width and layer count for a 24–40 point monthly history;
- modernized the module layout for PyTorch 2.x;
- retained only the supervised PatchTST backbone needed by this experiment;
- changed the input/output contract to `[batch, channel, time]` monthly rates;
- added validation and deterministic experiment utilities;
- did not copy the upstream data loaders or experiment runner.

The Apache License 2.0 text is available at
https://www.apache.org/licenses/LICENSE-2.0 and in the upstream repository's
`LICENSE` file.
