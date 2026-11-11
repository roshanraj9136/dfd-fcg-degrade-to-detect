
**train split → test split, degradation types seen at other severities**

| method | clean AUROC | degraded AUROC | false alarms @0.5 | worst | fakes caught @0.5 | balanced acc | detect @5% FA | @10% | @20% | mixed-pool AUROC |
|---|---|---|---|---|---|---|---|---|---|---|
| DFD-FCG (released) | 0.999 | 0.946 | 22.5% | 63% | 87.9% | 82.7% | 67.2% | 77.6% | 85.9% | 0.920 |
| Platt recalibration | 0.997 | 0.942 | 12.2% | 49% | 81.5% | 84.7% | 67.1% | 78.3% | 86.2% | 0.917 |
| quality-regression shift (face crop) | 0.998 | 0.948 | 8.4% | 34% | 81.3% | 86.5% | 75.5% | 82.7% | 90.9% | 0.954 |
| background-quality nulling | 0.992 | 0.928 | 17.3% | 43% | 84.5% | 83.6% | 65.3% | 76.5% | 86.7% | 0.930 |
| quality-augmented head re-fit | 0.967 | 0.880 | 8.0% | 21% | 71.7% | 81.8% | 65.9% | 73.1% | 79.6% | 0.892 |
| blur-probe contrast | 1.000 | 0.943 | 15.1% | 53% | 83.6% | 84.3% | 68.3% | 78.6% | 86.9% | 0.925 |
| probe responses, learned | 0.995 | 0.935 | 15.8% | 50% | 84.6% | 84.4% | 68.2% | 79.5% | 87.7% | 0.932 |
| **self-calibrated (ours)** | 0.999 | 0.945 | 6.9% | 24% | 79.1% | 86.1% | 73.5% | 83.3% | 88.3% | 0.951 |
| **self-calibrated + blur probe, strict (ours)** | 1.000 | 0.938 | 2.4% | 11% | 61.0% | 79.3% | 72.8% | 82.6% | 90.3% | 0.951 |

**train split → test split, tested degradation type unseen**

| method | clean AUROC | degraded AUROC | false alarms @0.5 | worst | fakes caught @0.5 | balanced acc | detect @5% FA | @10% | @20% | mixed-pool AUROC |
|---|---|---|---|---|---|---|---|---|---|---|
| DFD-FCG (released) | 0.999 | 0.946 | 22.5% | 63% | 87.9% | 82.7% | 67.2% | 77.6% | 85.9% | 0.920 |
| Platt recalibration | 0.997 | 0.942 | 12.2% | 49% | 81.5% | 84.7% | 67.1% | 78.3% | 86.2% | 0.917 |
| quality-regression shift (face crop) | 0.998 | 0.938 | 15.1% | 69% | 86.2% | 85.5% | 59.7% | 79.1% | 89.7% | 0.954 |
| background-quality nulling | 0.992 | 0.912 | 27.1% | 100% | 85.8% | 79.3% | 6.0% | 61.2% | 80.9% | 0.930 |
| quality-augmented head re-fit | 0.967 | 0.880 | 8.0% | 31% | 68.8% | 80.4% | 62.7% | 70.1% | 76.9% | 0.892 |
| blur-probe contrast | 1.000 | 0.942 | 18.1% | 63% | 84.1% | 83.0% | 64.7% | 75.9% | 84.9% | 0.925 |
| probe responses, learned | 0.995 | 0.938 | 17.3% | 54% | 82.2% | 82.5% | 64.5% | 76.7% | 84.2% | 0.932 |
| **self-calibrated (ours)** | 0.999 | 0.945 | 8.7% | 44% | 78.1% | 84.7% | 71.0% | 80.5% | 85.8% | 0.951 |
| **self-calibrated + blur probe, strict (ours)** | 1.000 | 0.938 | 2.3% | 14% | 61.4% | 79.6% | 71.0% | 80.0% | 87.2% | 0.951 |

**5-fold CV over test identity pairs, tested type unseen**

| method | clean AUROC | degraded AUROC | false alarms @0.5 | worst | fakes caught @0.5 | balanced acc | detect @5% FA | @10% | @20% | mixed-pool AUROC |
|---|---|---|---|---|---|---|---|---|---|---|
| DFD-FCG (released) | 0.999 | 0.946 | 22.5% | 63% | 87.9% | 82.7% | 67.2% | 77.6% | 85.9% | 0.920 |
| Platt recalibration | 0.993 | 0.932 | 15.3% | 51% | 83.2% | 84.0% | 64.7% | 75.6% | 86.4% | 0.913 |
| quality-regression shift (face crop) | 0.997 | 0.928 | 16.4% | 41% | 84.0% | 83.8% | 67.9% | 77.1% | 86.2% | 0.946 |
| background-quality nulling | 0.982 | 0.882 | 26.7% | 100% | 80.9% | 77.1% | 6.9% | 44.9% | 73.1% | 0.913 |
| quality-augmented head re-fit | 0.983 | 0.900 | 17.3% | 41% | 72.9% | 77.8% | 57.8% | 67.2% | 75.8% | 0.918 |
| blur-probe contrast | 0.999 | 0.938 | 17.1% | 57% | 80.5% | 81.7% | 62.4% | 73.5% | 83.2% | 0.925 |
| probe responses, learned | 0.999 | 0.933 | 16.0% | 50% | 83.6% | 83.8% | 62.1% | 75.4% | 85.5% | 0.936 |
| **self-calibrated (ours)** | 0.999 | 0.942 | 7.8% | 24% | 75.9% | 84.0% | 71.9% | 77.3% | 86.2% | 0.950 |
| **self-calibrated + blur probe, strict (ours)** | 1.000 | 0.937 | 1.4% | 6% | 59.5% | 79.0% | 71.4% | 80.0% | 87.7% | 0.953 |

**Per condition, released vs self-calibrated (type unseen), threshold 0.5: false alarms / fakes caught / AUROC**

| condition | released | self-calibrated |
|---|---|---|
| clean | 3% / 98% / 0.999 | 1% / 98% / 0.999 |
| RE_0 | 3% / 95% / 0.992 | 0% / 90% / 0.992 |
| CS_3 | 13% / 93% / 0.978 | 11% / 90% / 0.975 |
| CC_3 | 3% / 90% / 0.988 | 1% / 83% / 0.980 |
| BW_3 | 7% / 93% / 0.984 | 3% / 92% / 0.980 |
| GNC_3 | 0% / 17% / 0.873 | 0% / 13% / 0.879 |
| GB_1 | 16% / 97% / 0.983 | 0% / 88% / 0.986 |
| GB_3 | 40% / 97% / 0.955 | 4% / 80% / 0.960 |
| GB_5 | 63% / 98% / 0.905 | 11% / 73% / 0.903 |
| JPEG_1 | 10% / 97% / 0.988 | 1% / 88% / 0.988 |
| JPEG_3 | 33% / 97% / 0.952 | 17% / 90% / 0.948 |
| JPEG_5 | 59% / 95% / 0.885 | 44% / 90% / 0.882 |
| VC_1 | 7% / 93% / 0.979 | 1% / 85% / 0.981 |
| VC_3 | 10% / 88% / 0.946 | 4% / 70% / 0.947 |
| VC_5 | 33% / 88% / 0.879 | 13% / 72% / 0.871 |

**Unseen manipulation and unseen dataset (clean videos)**

| method | FaceShifter AUROC | FaceShifter caught @0.5 | DFDC AUROC | DFDC balanced acc @0.5 | DFDC false alarms | DFDC caught |
|---|---|---|---|---|---|---|
| DFD-FCG (released) | 0.925 | 73.3% | 0.786 | 71.2% | 10.0% | 52.5% |
| **self-calibrated (ours)** | 0.925 | 53.3% | 0.789 | 68.8% | 7.5% | 45.0% |
| **self-calibrated + blur probe, strict (ours)** | 0.954 | 26.7% | 0.819 | 63.8% | 2.5% | 30.0% |
| blur-probe contrast | 0.943 | 46.7% | 0.806 | 68.8% | 7.5% | 45.0% |

**Response of each head to a 2-px blur of the aligned face (clean inputs, mean logit change)**

| group | spatial | temporal | fusion |
|---|---|---|---|
| REAL | +1.13 | +1.11 | +1.44 |
| DF | -0.72 | +0.01 | -0.81 |
| F2F | +0.04 | +0.17 | -0.14 |
| FS | -0.00 | +0.57 | +0.08 |
| NT | +0.48 | +0.18 | +0.16 |
| FSh | +0.71 | +0.58 | +0.63 |
| DFDC_R | +0.80 | +0.89 | +0.94 |
| DFDC_F | +0.30 | +0.62 | +0.42 |

| input condition | real (fusion) | fake (fusion) | AUROC of the response alone |
|---|---|---|---|
| clean | +1.44 | -0.17 | 0.992 |
| RE_0 | +1.45 | +0.32 | 0.936 |
| CS_3 | +1.41 | +0.31 | 0.945 |
| CC_3 | +1.33 | +0.47 | 0.876 |
| BW_3 | +1.29 | +0.43 | 0.872 |
| GNC_3 | +0.43 | +0.55 | 0.443 |
| GB_1 | +1.28 | +0.49 | 0.900 |
| GB_3 | +1.11 | +0.50 | 0.854 |
| GB_5 | +1.03 | +0.53 | 0.810 |
| JPEG_1 | +1.30 | +0.44 | 0.911 |
| JPEG_3 | +1.10 | +0.48 | 0.836 |
| JPEG_5 | +0.96 | +0.55 | 0.731 |
| VC_1 | +1.35 | +0.42 | 0.881 |
| VC_3 | +1.11 | +0.54 | 0.762 |
| VC_5 | +0.70 | +0.37 | 0.661 |
| DFDC clean | +0.94 | +0.42 | 0.738 |

**Earlier two-video findings re-tested (clean FF++ test videos)**

- REAL (n=70): accuracy 97%, spatial-only 93%, temporal-only 87%; spatial share (Shapley) 55%; top query nose 43%; top spatial region nose 49%; top temporal region skin 48%; query signed {'lips': -0.2549999952316284, 'skin': -0.1088000014424324, 'eyes': 0.03849999979138374, 'nose': -0.459199994802475}; top spatial layers [22, 21, 24], temporal [22, 19, 10]
- FAKE (n=60): accuracy 98%, spatial-only 95%, temporal-only 98%; spatial share (Shapley) 42%; top query eyes 35%; top spatial region eyes 30%; top temporal region skin 36%; query signed {'lips': 0.2775999903678894, 'skin': 0.3095000088214874, 'eyes': 0.5004000067710876, 'nose': 0.11699999868869781}; top spatial layers [20, 18, 16], temporal [22, 24, 9]
- DF (n=15): accuracy 100%, spatial-only 93%, temporal-only 100%; spatial share (Shapley) 52%; top query eyes 40%; top spatial region eyes 37%; top temporal region skin 33%; query signed {'lips': 0.6104000210762024, 'skin': 0.3605000078678131, 'eyes': 0.840399980545044, 'nose': 0.4771000146865845}; top spatial layers [20, 18, 21], temporal [22, 24, 23]
- F2F (n=15): accuracy 100%, spatial-only 100%, temporal-only 100%; spatial share (Shapley) 31%; top query skin 33%; top spatial region skin 32%; top temporal region skin 33%; query signed {'lips': 0.1850000023841858, 'skin': 0.3698999881744385, 'eyes': 0.31700000166893005, 'nose': -0.02879999950528145}; top spatial layers [16, 18, 20], temporal [22, 9, 10]
- FS (n=15): accuracy 100%, spatial-only 100%, temporal-only 100%; spatial share (Shapley) 58%; top query eyes 51%; top spatial region eyes 43%; top temporal region skin 41%; query signed {'lips': 0.10920000076293945, 'skin': 0.27469998598098755, 'eyes': 0.7064999938011169, 'nose': 0.1648000031709671}; top spatial layers [20, 18, 17], temporal [22, 24, 23]
- NT (n=15): accuracy 93%, spatial-only 87%, temporal-only 93%; spatial share (Shapley) 26%; top query skin 31%; top spatial region skin 33%; top temporal region skin 36%; query signed {'lips': 0.20589999854564667, 'skin': 0.23260000348091125, 'eyes': 0.13770000636577606, 'nose': -0.14509999752044678}; top spatial layers [22, 21, 8], temporal [22, 9, 8]
- FSh (n=15): accuracy 73%, spatial-only 40%, temporal-only 80%; spatial share (Shapley) 42%; top query nose 33%; top spatial region nose 36%; top temporal region skin 46%; query signed {'lips': -0.24570000171661377, 'skin': 0.14550000429153442, 'eyes': 0.1363999992609024, 'nose': -0.2443999946117401}; top spatial layers [22, 24, 21], temporal [22, 23, 24]

Fingerprints: 79 fake videos, method identified from the explanation 77% vs 57% from the 3 logits (chance 25%). Confusion [[16, 1, 2, 1], [2, 13, 0, 4], [0, 0, 20, 0], [0, 7, 1, 12]]

EN counterfactual: {'n_false_alarms': 212, 'false_alarms_fixed': 82, 'mean_nuisance_share_in_false_alarms': 0.37398134273408856}

Geometry: {'s': {'drift_energy_top': [0.2678, 0.3809, 0.4557, 0.5134, 0.5576, 0.5983], 'cos_mean_drift_vs_forgery': 0.32369068481178, 'cos_top_drift_dir_vs_forgery': 0.18005413868332742}, 't': {'drift_energy_top': [0.6237, 0.7121, 0.7676, 0.8107, 0.8481, 0.8758], 'cos_mean_drift_vs_forgery': -0.037819319106252136, 'cos_top_drift_dir_vs_forgery': 0.17863844515107707}, 'a': {'drift_energy_top': [0.5901, 0.672, 0.7341, 0.7827, 0.82, 0.8495], 'cos_mean_drift_vs_forgery': -0.1856993137026544, 'cos_top_drift_dir_vs_forgery': 0.3200673111819775}}
