# Changelog

## [1.3.0](https://github.com/Diegodepab/agrivision-khaos/compare/v1.2.2...v1.3.0) (2026-09-14)


### Added

* add automated dataset curation pipeline ([41f0999](https://github.com/Diegodepab/agrivision-khaos/commit/41f09997f478b754850acbadec4c300cd8c9d107))
* add dataset balancing ([2621ebd](https://github.com/Diegodepab/agrivision-khaos/commit/2621ebdaf123a11358cca8da6f4b31bf8203adb4))
* AgriVision Khaos initialization and framework redesign ([2b69967](https://github.com/Diegodepab/agrivision-khaos/commit/2b69967c6bd171acea499233a7ea1a24a90adf8d))
* **augmentation:** introduce image augmentation system and benchmarks ([6b701f3](https://github.com/Diegodepab/agrivision-khaos/commit/6b701f396ec99737203b2c513a84f0a2d720c69e))
* **curation:** expand data pipeline with retrieval, history and geometry ([1c8b7b5](https://github.com/Diegodepab/agrivision-khaos/commit/1c8b7b5303b2a9b78f54ba26460cd90f03e5b455))
* **deduplication:** improve duplicate detection and handling ([5d4a2dc](https://github.com/Diegodepab/agrivision-khaos/commit/5d4a2dcacd3df98568009b91797a27c13854f97c))
* **export:** add dataset asset export support ([e56fbc6](https://github.com/Diegodepab/agrivision-khaos/commit/e56fbc61efea4a01ec9f52f268966bbc5eb7c40f))
* improve dataset curation pipeline ([536d5d2](https://github.com/Diegodepab/agrivision-khaos/commit/536d5d27e9e9f77012ea5c210e0d07dbc6d9b483))
* improve dataset curation pipeline ([9c09c57](https://github.com/Diegodepab/agrivision-khaos/commit/9c09c57521f41571af93879f31c86883ac35d975))
* **pipeline:** improve execution and resume support ([f459962](https://github.com/Diegodepab/agrivision-khaos/commit/f459962dbd428f1e98837847ff9effd6083e9c34))
* **quality:** improve dataset quality metrics ([b4539f6](https://github.com/Diegodepab/agrivision-khaos/commit/b4539f6179142d4712426a1b9a5a5a1933404a72))


### Fixed

* balancing error ([394ba1c](https://github.com/Diegodepab/agrivision-khaos/commit/394ba1c0154bdf27e22cc0643586bdd5e3f81c8e))
* **deps:** add pycocotools for COCO dataset ingestion ([20551de](https://github.com/Diegodepab/agrivision-khaos/commit/20551def9940d02aa3dbf51cfb511ae0d06b28ab))
* improve dataset curation pipeline ([5a293aa](https://github.com/Diegodepab/agrivision-khaos/commit/5a293aa51ae0851ab351140295ba1ab639db6dc6))
* improve ingestion and pipeline processing ([08c0d33](https://github.com/Diegodepab/agrivision-khaos/commit/08c0d335e8fa578287986c676563f27c6ad5269a))
* **ingest:** resolve relative YOLO split paths and aliases ([9776bbe](https://github.com/Diegodepab/agrivision-khaos/commit/9776bbe22860ba33535e3c1ddbb03bc2caa5d413))
* **metrics:** add missing box_blurs return value to resolve unpacking error ([411cfac](https://github.com/Diegodepab/agrivision-khaos/commit/411cfac6319bb643ce54bc71256df982aa46377f))
* **pipeline:** resolve shm memory leaks, enhance Kaggle dataset ingestion, and support YOLO/VOC formats ([d7a0aa7](https://github.com/Diegodepab/agrivision-khaos/commit/d7a0aa7507814b319faf51d743ab46d89d885620))


### Changed

* add dataset documentation ([c8b3c84](https://github.com/Diegodepab/agrivision-khaos/commit/c8b3c84409f0fae3d051d553a707e419c9d5dc1b))
* add integration, execution and dry-run tests ([2590dab](https://github.com/Diegodepab/agrivision-khaos/commit/2590dab7e2848de95cca4808f9910c584dd853dc))
* **curation:** remove fastdup dependency to ensure python 3.14 compatibility ([cdd6cee](https://github.com/Diegodepab/agrivision-khaos/commit/cdd6cee3af39411ed0565ae92d4bfe5ec0204d40))
* document dataset curation workflow ([147015a](https://github.com/Diegodepab/agrivision-khaos/commit/147015ac3a4ee165c629dbfe437a305c5e06a42d))
* document manual review workflow ([fa71bed](https://github.com/Diegodepab/agrivision-khaos/commit/fa71bed3e5c7cf96b5f96ec077058977cc129f06))
* inicializa estructura y dvc ([2473344](https://github.com/Diegodepab/agrivision-khaos/commit/2473344ad9b1df0550095cb75995459789b78f75))
* migrate source code to agrivision_khaos package ([401e2a5](https://github.com/Diegodepab/agrivision-khaos/commit/401e2a5a8c35cb793564c784b703eb98f9387387))
* setup github workflows and enhance docker environment ([1f3eae5](https://github.com/Diegodepab/agrivision-khaos/commit/1f3eae5288e681bbda79b684c9f3ac9f7655bd54))
* update architecture, workflows and add execution guides ([9f22364](https://github.com/Diegodepab/agrivision-khaos/commit/9f223644ee00b9fadd3184b371aa0bb4a68517f6))
* update manual review and unattended pipeline guides ([e1dd291](https://github.com/Diegodepab/agrivision-khaos/commit/e1dd291552aaad37387376dc9db998da3bc7bff5))
* update pipeline architecture to reflect fiftyone native deduplication ([13d3f2d](https://github.com/Diegodepab/agrivision-khaos/commit/13d3f2d208ef5135d5fe45c97b6a02e9882aaa7c))
* update pipeline documentation and configuration ([8f4b6c0](https://github.com/Diegodepab/agrivision-khaos/commit/8f4b6c0764c38bfde4d437609009d059194dfe00))
* update project configuration and dependencies ([3464bd4](https://github.com/Diegodepab/agrivision-khaos/commit/3464bd42e4aef9a3936a01e929d16a920cdd50d5))
* update project development and CI configuration ([0f3b9ad](https://github.com/Diegodepab/agrivision-khaos/commit/0f3b9ad9fd9672d98f96d071fe2196e06c7a7f08))
* update workflow and manual review documentation ([72bc8d9](https://github.com/Diegodepab/agrivision-khaos/commit/72bc8d9fabd51546c84886071b449d8f3c0ec325))
* update workflow configurations, makefile and execution profiles ([b2b4cce](https://github.com/Diegodepab/agrivision-khaos/commit/b2b4ccedd9181b32cbf10aef118abfe82bc3680d))

## Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
