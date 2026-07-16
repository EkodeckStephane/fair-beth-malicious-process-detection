# FAIR-X / FAIR-BETH Artifact Manifest

Date: 2026-07-16

Purpose: identify the manuscript sources, environment file, scripts, figures, and key numerical artifacts used by this repository snapshot. SHA-256 hashes are computed over the local files in `new/github_repo`. Raw BETH, MalBehavD-V1, and VirusTotal metadata dumps remain with their original providers or access channels; full computational reproduction requires obtaining the original datasets and locally collecting the metadata described in the README.

| File | SHA-256 |
|---|---|
| `README.md` | `F81E151FAC09DA7C1DAEE9DBD8AF7998F7F3FF56E3A892C1129EE7441BF65812` |
| `requirements.txt` | `6AB89DF4D8AB37C6ACCC3CEF062538E63E16E76E49C2CAB39EBE1A429A9D9F2F` |
| `paper/fair_x_tdsc_v2.tex` | `B8F68D36CF0CFF00FE0AE4D8EA18BC751EC5F1EA0C9F7158DA12D843167679A1` |
| `paper/fair_x_tdsc_v2.pdf` | `BB0A8337DEB96519D7D6E7567EED880D754F5AB36D2FB5010752B0BB5F0E82EF` |
| `paper/references.bib` | `8A325FE27043F3436D6CD6400F55C575ADAFB0FDFAB17670DD42ED987C3FEDDE` |
| `paper/cover_letter_fair_x_tdsc_v2.tex` | `67DC03E7871C304214A63A27D79209BE7EAD68BCE5C5D8496FBC5D09CCD986E6` |
| `paper/cover_letter_fair_x_tdsc_v2.pdf` | `8C1C05E8B0181E28EDB01EE02152098AE5FD96086DD73C6844E24A5E0CBFBF19` |
| `scripts/preprocessing_v2.py` | `BE737B9A8D4F98C60D1FDC3EF2DD67F1A0CF719A2BBE98E90C659345EDE18C29` |
| `scripts/beth_limit_lifting_analyses.py` | `AB5E86B1DCD35FFEBB0003973CDFDC1E17C93DD9B6C7069B0D597F70BF9BEBE0` |
| `scripts/paired_comparison_audit.py` | `A2E3889597A2531A05C35DCBAA8988D9D677F2A4F1670BCB1B22CB6D579FCD0C` |
| `scripts/malbehavd_temporal_audit_full.py` | `7583E59FF61606761B095B3077D4AFAC990AA8483EA7E49F978409B7444AD09F` |
| `scripts/tabular_sota_and_calibration_audit.py` | `08032618A23DF0C78E1B74C6189087AEF7D59383A6E533EEEE901244F8259A00` |
| `scripts/robust_threshold_validation.py` | `52A6AD5DCF3A03E31A3ECEA203150D2B5D660CB4F8A95F1455A0D262E358861E` |
| `scripts/sequence_capacity_ablation.py` | `F43B922534D7278A1D5E233A769EADC7833D0842FB1D461EFA8D5484F5EC2674` |
| `results/beth_additional_audits/tabular_sota_comparison.csv` | `3F458925505C49C8B5FF8CD4F2B2E2410516B0E631E0762AE223C9BAB3014F15` |
| `results/beth_additional_audits/calibration_audit.csv` | `8CE0101189B19FE286DFB5AE812DC2DF3C7C6434B44385647B30A14B261A683F` |
| `results/beth_additional_audits/paired_comparison_audit.csv` | `41641DED363F6B18F9E526B02F6E213221C5DA9619FCA4004AEA625035A6A834` |
| `results/beth_additional_audits/paired_comparison_audit.json` | `FBBD55A87EA91611BB0FC7077D2C32AA5BBC9B1CC9EC9ED0D4814A93A9137D07` |
| `results/beth_additional_audits/reliability_diagrams.png` | `B20370AA904B2CBBDC87E97717052A4409B4D3340CB3BC5BAAB467E14CFA3F72` |
| `results/beth_limit_lifting/beth_prefix_results.csv` | `8311051BEFEF8C325B2E08E6731ECB7F16767725EA37F3D70A6D1E273CF303FC` |
| `results/beth_limit_lifting/beth_cost_optimal_thresholds.csv` | `C414AA68D00A0B0A9193F5804BEFE1B661F044DE0D6C90545EFCDC983AEEE9BB` |
| `results/beth_limit_lifting/beth_evasion_stress_tests.csv` | `CD90FDB43E9D1FFDC9D1AAB365DE3FC32642F27C6B3F0F486A06522F1C393B34` |
| `results/beth_limit_lifting/beth_group_temporal_robustness.csv` | `A76FDB84040E5ECBAF3F7B9615EE9D093A49984B6B267B28D50D18C41A4E5A73` |
| `results/beth_limit_lifting/beth_host_disjoint_robustness_summary.csv` | `7EC12515DBDF8D4D51FFC0326CA616EA7C24EAC696D4F4326ECD8734E5CE29B9` |
| `results/beth_limit_lifting/beth_temporal_70_15_15_result.csv` | `3BE26FDA88BCFFCE34C35752ACD304AF7A2AFCAF7240B1B25AD22260B1D90BF7` |
| `results/beth_limit_lifting/beth_tabrf_permutation_importance.csv` | `383021589DE187B1750BAD6983FFC64F7CF1F77DC46A1AD17C6871C4CA7C8C13` |
| `results/sequence_capacity_ablation/sequence_capacity_ablation.csv` | `7C7691310BB2C8BF29DDAD258DAF7D51BFE4B71EA7E6B9D3BE3EEBEE3D86AADD` |
| `results/sequence_capacity_ablation/summary.json` | `5FBC9773C5196DE33171411DE2D5A6B3C7EEECA4CC754143B0F2029D7CDD7DA7` |
| `results/external_malbehavd/external_results_table.csv` | `41D7814FFFA44188488BBEE82CA81957835484275DD3EB187BFC561F36CB5DF5` |
| `results/external_malbehavd/repeated_split_summary.csv` | `1ABF61CDCBC6346B80FF018896A19D913AC101D1C7ECBDED03C9B0BD43138DBB` |
| `results/external_malbehavd/prefix_results.csv` | `FA73553DA579DBB3DC0588686E51E2A6CAC0FE10BB6D87903EDE890E16B3BA10` |
| `results/external_malbehavd/feature_importance.csv` | `351CB7DB542FBCC3FF040C1CA23FA5297673CA2328CCB6E4A29A2908A5C02085` |
| `results/external_malbehavd_temporal/summary.json` | `9E412DE4530F2E7B9FFAAD9D492070F054447FA23EE800BD2DE518D9E61E6295` |
| `results/external_malbehavd_temporal/cross_fitted_thresholds.csv` | `37AF7156381F0F60408C64213C2A35705F4CC58F909CECEF9760CB08C19CC430` |
| `results/external_malbehavd_temporal/locked_threshold_sensitivity.csv` | `E5D4CA2760DA6D7848D35761C29FD58892B1ECAD13C8A68FEADA75BB368EDFC3` |
| `results/robust_threshold_validation/summary.json` | `71639174CB2024E7E96970CEF2DFA3D670E430917B4CAF34C1DE9926A0CB1B58` |
| `results/robust_threshold_validation/cross_fitted_thresholds.csv` | `EEAB343CF87A46D91A44455505332A247CD47823CF084A7EB437FF124AF0337D` |
| `results/robust_threshold_validation/locked_threshold_test_sensitivity.csv` | `19D395F5C1694ED7B9EE18F944EA18AF71B363E3D373418DC80D40FD372B8D78` |
