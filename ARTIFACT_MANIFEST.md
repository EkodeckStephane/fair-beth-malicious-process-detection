# FAIR-X / FAIR-BETH Artifact Manifest

Generated: 2026-08-26

This manifest records SHA-256 digests for revision-critical executable and machine-readable artifacts. The scientific manuscript is intentionally kept outside this code repository. Raw BETH and MalBehavD datasets are not redistributed and are therefore not hashed here; the BETH raw-file identity audit is recorded separately in `results/canonical/dataset_inventory.csv`.

The manifest excludes itself to avoid a self-referential digest. Binary model checkpoints and transient pickle artifacts are also excluded; canonical claims are derived from committed text/CSV/JSON outputs.

| File | SHA-256 |
|---|---|
| `.github/workflows/revision-canonicalize.yml` | `c232ebceec9110692905abb7dde51790c97b1d452a2c971549e7b3bd865c3877` |
| `.github/workflows/revision-data-audit.yml` | `a66c7d4b72feea3eaa4df86ed057dca3c549fc1b1aa29cd2e943977dfb491e20` |
| `.github/workflows/revision-experiments.yml` | `74c6565e72abee592a1a9871a94ca07d1c8180ac058cb65ec46050f6ded62e4d` |
| `.github/workflows/revision-manifest.yml` | `3cc976cff6a6af93c25b294c2b56c33a27f3f726aee393bb042ab7cf021763c2` |
| `.github/workflows/revision-secondary-robustness.yml` | `0216a720417a5ae2d530037b57a5fd1dea2ec7280f2791296c85d1488f229398` |
| `README.md` | `85a3ee9f2a923292d56f8174b90189dc0d90528e1015400755a9ae6b070146ef` |
| `requirements.txt` | `6ab89df4d8ab37c6accc3cef062538e63e16e76e49c2cab39ebe1a429a9d9f2f` |
| `results/beth_limit_lifting/beth_cost_curve_all_thresholds.csv` | `718032d1a96f79ce8b04eb1d070c245d0bea4d2580695eb975ba34d7824f6fe1` |
| `results/beth_limit_lifting/beth_cost_optimal_thresholds.csv` | `534043376b7821ff64e4797d4ea00f3fc090fdb59784c3396a331596322b95a6` |
| `results/beth_limit_lifting/beth_evasion_stress_tests.csv` | `efcdebe10af8aff94d484307c0f4f25fdd1011e0a4ba688f1d728b8df6e74709` |
| `results/beth_limit_lifting/beth_group_temporal_robustness.csv` | `28ef0d0441f904b50abbe1a04d69dd462c68cfc69672713ca114fd84c6d39027` |
| `results/beth_limit_lifting/beth_host_disjoint_robustness_summary.csv` | `29bdd07e0a0f59c30bfc04f29d6ed2b9a7fac14d4d8c6b4c0dfbe7f4d19092eb` |
| `results/beth_limit_lifting/beth_prefix_results.csv` | `447caf545d6cf612fc76d36736d4d9c1444bde21247961eb603e43678debd599` |
| `results/beth_limit_lifting/beth_tabrf_permutation_importance.csv` | `95870ee51effdcc8f596098ab9b3c2ad0dcf874681f04d66290cda702f5e1d85` |
| `results/beth_limit_lifting/beth_temporal_70_15_15_result.csv` | `2fd155ce09509d3a8274d53fc6d9ca09dafbb8b32abae3b453c94667a729b1ac` |
| `results/beth_limit_lifting/inventory.json` | `947d9e15175ad03587b30a363718727683a29a17a0bba1a363c8ef74090fe0ac` |
| `results/canonical/README.md` | `2ff338ed914eff2148efa41503a9384d47b4eddc72660c89b18fec77aaa3f421` |
| `results/canonical/calibration_uncertainty.csv` | `e7b0dfd48decd8496dab383298ed4cf62aa0003fca69ed2b582d565a14c2ed7b` |
| `results/canonical/dataset_inventory.csv` | `c799c5abf483647f6f97c1c25be1283e65aa507be4821f7a1acbc207bd84dcc6` |
| `results/canonical/diagnostic_gaps.csv` | `f8ce23c994b2bc307eb49c2a1546fa338758702ee47944c72afce7bd4def1dc1` |
| `results/canonical/dns_schema_comparison.csv` | `035c26697241fa2e857d3c685b42bce45138d1765ae815f2852d691c00afb55d` |
| `results/canonical/host_overlap.csv` | `e9fbdd0de478104afb7ccbb5654b93752f3a3500374f52dcc52cda574bc086ed` |
| `results/canonical/main_results.csv` | `f81f44ce97a6f582f25777b3ede1732626aa82b27d8948982e4c635d7a341637` |
| `results/canonical/paired_comparisons.csv` | `5ecdd35d5d04f1c4662feeb2ae0e07031c670072fdeaf40882b48d707c9fca96` |
| `results/canonical/sequence_ablation.csv` | `7343c1da9892a1e68e109c8369e17ade7c132d95757119e414293a2db9e9eb1b` |
| `results/canonical/supplementary_sensitivity.csv` | `1ceed8f3bc8313704d75ef5032f3327cfbe419e6ede11f191badd6666f7b87d1` |
| `results/external_malbehavd/external_results_table.csv` | `4a188ab4b8272bd4c6ceaa7f86cb419e8def612b2744c5d53b3c8592c3b47c4e` |
| `results/external_malbehavd/feature_importance.csv` | `484da88ea001ad54213e224f43f1e733abcda3f94d8d5aed9c65f9982b603e50` |
| `results/external_malbehavd/inventory.json` | `f065b8e02b047f735e8a8471c9996c255006123c39b915b9550afa8f8250c974` |
| `results/external_malbehavd/prefix_results.csv` | `b90432c062fe9394b3d845f28a189dfd736762cfa3a873a82111cdff03dbfb14` |
| `results/external_malbehavd/repeated_split_results.csv` | `f81ed9c600c54f1008fc8032a155232b5880643815f4f3898cf24277f2c58f68` |
| `results/external_malbehavd/repeated_split_summary.csv` | `5143d00e2491200af19b94939ebe6d9e3551c566d9c3f3dd910d6a24ea1af345` |
| `results/external_malbehavd_temporal/cross_fitted_thresholds.csv` | `2d041d13315a55392e232ae4ff6b361622815862e8cb759221f1fddf9da9b833` |
| `results/external_malbehavd_temporal/locked_threshold_sensitivity.csv` | `820604a3a9d222ffb76e41a5c0c2b3af05812eed502c0ac1aad984ff042c39e8` |
| `results/external_malbehavd_temporal/summary.json` | `886331d259b1fba97b838e0d00c2ee0ae2e71df67126bebdb0c9ddac274fbe34` |
| `results/revision_audits/calibration_bin_occupancy.csv` | `f8b8aae6b474c48174b7c6c8b0f878f555302344ada1df148bb5f9c719154f21` |
| `results/revision_audits/calibration_uncertainty.csv` | `e7b0dfd48decd8496dab383298ed4cf62aa0003fca69ed2b582d565a14c2ed7b` |
| `results/revision_audits/dataset_audit_summary.json` | `5bef5ff2f8e785df5383856aa5bd2410c9b1226fed7035a9ea9086a83fb5ed86` |
| `results/revision_audits/dataset_inventory.csv` | `c799c5abf483647f6f97c1c25be1283e65aa507be4821f7a1acbc207bd84dcc6` |
| `results/revision_audits/dns_schema_comparison.csv` | `035c26697241fa2e857d3c685b42bce45138d1765ae815f2852d691c00afb55d` |
| `results/revision_audits/host_overlap.csv` | `e9fbdd0de478104afb7ccbb5654b93752f3a3500374f52dcc52cda574bc086ed` |
| `results/revision_audits/matched_tuning_cv.csv` | `5c126e04eb768539697c391593e3c7fdf89b40844a0ead286f9a71b122768051` |
| `results/revision_audits/matched_tuning_lock.json` | `2ae6bf49c8a73745d1c0d9188a64c3a95a9c73ce50fd2fc3e854aaa12036f23a` |
| `results/revision_audits/matched_tuning_summary.csv` | `2e85a32e3c1cf3423e2cfe83615ad3fb2ddd4917bd8271b33e465b91331d15a4` |
| `results/revision_audits/matched_tuning_test_results.csv` | `3f6de0f28f8a9a9f60196dbbfda4e81288bc2f95c2d94fff7ca562bc28d4e59d` |
| `results/revision_audits/matched_tuning_test_scores.csv` | `70b19edc1ae87ede5b7a467cce9c520c216e5b41bc3cce25433f7d2fad6110aa` |
| `results/revision_audits/sequence_length_matched/sequence_length_matched_ablation.csv` | `7343c1da9892a1e68e109c8369e17ade7c132d95757119e414293a2db9e9eb1b` |
| `results/revision_audits/sequence_length_matched/summary.json` | `0f8350abe7083cb88d8d1a9cfd64e1149feaaeeca6fb77080a206a258755d659` |
| `results/revision_audits/sequence_length_matched/training_history.json` | `e2c9f8e4191a44dda6ac6a4fb9818fb7b9485978344705a05d27d316f9c84f7f` |
| `results/revision_audits/supplementary_source_sensitivity.csv` | `1ceed8f3bc8313704d75ef5032f3327cfbe419e6ede11f191badd6666f7b87d1` |
| `results/revision_audits/supplementary_source_sensitivity_protocol.json` | `0f84b825720483a255cc64d667a7dd7e60a8ebda8da5dc4befd478388f747a38` |
| `results/revision_audits/supplementary_source_thresholds.csv` | `6e30673c2b6e4e8c579c05823c869d080540f4c7ba6a8918cefd69bacb6dac0d` |
| `results/robust_threshold_validation/cross_fitted_thresholds.csv` | `748e404f96e498849e194aceed782c31c30cbaf0bdf19204ac66a298ce1d54e2` |
| `results/robust_threshold_validation/locked_threshold_test_sensitivity.csv` | `fe4555323c69534ed01aa34d5f3f504c3864d75da6f5bb06ef0d40ec30d76d01` |
| `results/robust_threshold_validation/summary.json` | `49798b7968691e8a7cb35c377698aa9ad4bf75bf316b234f14e9edb4c30fada1` |
| `scripts/baseline_rf.py` | `182dfa94b917ecd2c136bfc81a88563a0d294486f9216bad839bfaa43d99adb1` |
| `scripts/beth_limit_lifting_analyses.py` | `ab5e86b1dcd35ffebb0003973cdfdc1e17c93dd9b6c7069b0d597f70bf9bebe0` |
| `scripts/build_canonical_revision_results.py` | `5629e0ed58384082ed579c30c8842214332e2538c673a695a8c4dbda4befd4e0` |
| `scripts/calibration_fusion.py` | `e395a932384a58c691ac0c798324a12cb8db8044b9fa83dccb6e6ae94bb5014c` |
| `scripts/calibration_uncertainty_audit.py` | `bb3a9e0a555c91e8242cb61df3cf84be20aa64e3fc22d05071d7a4fc30cca6b6` |
| `scripts/config.py` | `43e58d940372f07924f5a5463f3d280fac6ad03df05a75ae4d74e950a3d050cc` |
| `scripts/detector_sequence.py` | `11e944f3b77d7ba45d29ea32267263b9a0ff5fddb384739fef1f316408c1e452` |
| `scripts/detector_tabular.py` | `78d5213c78460909f2eb248a6651f651255e22962546302a8270e3e6ad6123e3` |
| `scripts/detector_tactic.py` | `e621e4b5f034304721fdc835d347c0e7d89ab84fe0a54d8c1909a73f25bc1565` |
| `scripts/evaluate.py` | `e2113514264f59a4f207938d8d2a3140052c1f1698390027d2a2014798390942` |
| `scripts/external_malbehavd_validation.py` | `c749eb7f3e65377ceca3a2aab357032a0ea091376bc30067a723ec1a289e779f` |
| `scripts/generate_artifact_manifest.py` | `e34e4e2bf17c7067fc059741ebb43e6cf5d5c70adddfa3be58d6d5b3a8f0fb62` |
| `scripts/generate_mitre_map.py` | `34d8fc894512c88508b41f6f339dcad58344f72255052d8d62571f903c0368bd` |
| `scripts/malbehavd_temporal_audit_full.py` | `7583e59ff61606761b095b3077d4afac990aa8483ea7e49f978409b7444ad09f` |
| `scripts/matched_hyperparameter_tuning.py` | `9823cc9c66f4d4b9ada4ae8150c8d4045e07e429ab8d0c8a978e16496759539d` |
| `scripts/matched_secondary_robustness.py` | `ae7ae297a96ec007695eb9ec0a40f0595ef159b3f1aeb6a0a3f214df915171df` |
| `scripts/paired_comparison_audit.py` | `a2e3889597a2531a05c35dcbaa8988d9d677f2a4f1670bcb1b22cb6d579fcd0c` |
| `scripts/preprocessing_v2.py` | `be737b9a8d4f98c60d1fdc3ef2dd67f1a0cf719a2bbe98e90c659345ede18c29` |
| `scripts/revision_dataset_audit.py` | `0ec401e7545de9cf2f832d27293903b4342d6fe445a04331f5e82a2417c3f905` |
| `scripts/robust_threshold_validation.py` | `52a6ad5dcf3a03e31a3ecea203150d2b5d660cb4f8a95f1455a0d262e358861e` |
| `scripts/run_pipeline.py` | `5c65c5986ae8fffdc81eb04b21b92ac2c5dbefe120c91d2f23e81cde3add8d6b` |
| `scripts/sequence_capacity_ablation.py` | `f43b922534d7278a1d5e233a769eadc7833d0842fb1d461efa8d5484f5ec2674` |
| `scripts/sequence_length_matched_ablation.py` | `bd6e19ab9883552b98cfcafa57eb5418b289231d8017b1ffd246547ad7414ab0` |
| `scripts/supplementary_source_sensitivity.py` | `4ffa8ab91f18ef1a608b2af4265a42567d08768b6f84ad353703fb693eab3b04` |
| `scripts/system_info.py` | `8bc0f04d38603600a2118e21bf9a725218185846ef53a0e900e3659eb4e156ef` |
| `scripts/tabular_sota_and_calibration_audit.py` | `08032618a23df0c78e1b74c6189087aef7d59383a6e533eeee901244f8259a00` |
| `scripts/utils.py` | `123ef475a570a5fba6343da2bdfda71a99542af3f3d7be3607a14790b57620f3` |
