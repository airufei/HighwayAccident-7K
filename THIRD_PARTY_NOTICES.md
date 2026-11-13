# Third-party notices

## Software and models

- Ultralytics YOLO is an external dependency. Its open-source distribution is licensed
  under AGPL-3.0. Preserve its copyright and license notices when obtaining or
  redistributing that dependency: https://github.com/ultralytics/ultralytics
- PyTorch, NumPy, scikit-learn, Matplotlib, OpenCV and other dependencies retain their
  own licenses. Installing this repository does not change those licenses.
- The model checkpoints contain trained tensors for the published architecture.
  Their identity manifests distinguish the original training checkpoints from the
  inference-only exports. Removing optimizer state does not change model tensors.

## Source datasets

- CCD: https://github.com/Cogito2012/CarCrashDataset
- TUM Traffic Accid3nD: https://accident-dataset.github.io/
- TUMTraf / A9: https://github.com/tum-traffic-dataset/tum-traffic-dataset-dev-kit
- CARLA simulator: https://carla.org/

These links identify source providers; they are not a blanket redistribution license.
The source inventory distinguishes public data, authorized real footage, video samples,
and generated data. The authors' reannotations are separate from source media and
pre-existing labels. The code repository's license does not override source terms.

Additional external evaluations use the recorded DADA-2000 and CCD score streams.
These are model outputs and labels used for the reported analyses, not copies of the
underlying videos. CCD also contributes to the assembled training/validation collection;
its name alone does not establish an independent external test.
