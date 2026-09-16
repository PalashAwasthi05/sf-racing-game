# Facade priority ranking v15 (driver view)

Stations every 8 m on drivable OSM highways (primary, secondary, tertiary, unclassified, residential, living_street) within 200.0 m of the local origin (228 stations; 380 at 265.0 m). Driver eye height 1.25 m. Wall weight = length x min(height, 12 m) x incidence / max(25, d^2), stations 3..160 m, incidence >= 0.15, 2D footprint occlusion, best 12 stations per wall (uncapped variant also reported).

Total score 609.11. Buildings covering 85%: **19**; 95%: **29** of 58 (at 265 m: 26 and 37). Zero-score at 200 m: 11, of which never driver-visible at either radius: 5.

'Unverified prior' buildings inside the 85% set (7): `sf_201006_0001081`, `sf_201006_0000645`, `sf_201006_0001103`, `osm_129176908`, `sf_201006_0002333`, `sf_201006_0006267`, `sf_201006_0006942`

Never driver-visible (zero at both radii; leave as priors) (5): `sf_201006_0001461`, `sf_201006_0001907`, `sf_201006_0002014`, `sf_201006_0003254`, `sf_201006_0029858`

Zero only at 200 m because their roads lie outside the station disc (6): `osm_129176917`, `osm_346093777`, `sf_201006_0005658`, `sf_201006_0006801`, `sf_201006_0028162`, `sf_201006_0136107`

Sanity: walls processed 469 of 469; shuffle-invariant ranking: True.

| rank | id | score | share | cum | rank uncapped | rank @265 m | visible frontage m | walls vis/total | nearest road m (class) | nearest station m | evidence | flags |
|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---|
| 1 | `sf_201006_0000306` | 94.511 | 15.5% | 15.5% | 1 | 1 | 469.8 / 469.8 | 31/31 | 6.93 (residential) | 7.07 | Image hierarchy + LiDAR levels | top85 |
| 2 | `sf_201006_0000213` | 81.506 | 13.4% | 28.9% | 2 | 2 | 564.4 / 564.4 | 37/37 | 9.01 (primary) | 9.61 | Image hierarchy + LiDAR levels | top85 |
| 3 | `sf_201006_0000003` | 56.730 | 9.3% | 38.2% | 3 | 4 | 545.1 / 813.5 | 26/42 | 7.94 (unclassified) | 13.51 | Image hierarchy + LiDAR levels | top85 |
| 4 | `sf_201006_0000115` | 56.090 | 9.2% | 47.4% | 4 | 3 | 204.2 / 337.8 | 4/5 | 5.17 (secondary) | 10.0 | Image hierarchy + LiDAR levels | top85 |
| 5 | `sf_201006_0001081` | 31.481 | 5.2% | 52.6% | 5 | 5 | 103.7 / 123.7 | 7/8 | 4.94 (residential) | 6.05 | Unverified prior | prior-in-85 top85 |
| 6 | `sf_201006_0002210` | 25.421 | 4.2% | 56.8% | 6 | 10 | 118.0 / 140.5 | 9/10 | 4.89 (residential) | 5.19 | Image hierarchy + LiDAR levels | top85 |
| 7 | `osm_149167937` | 23.768 | 3.9% | 60.7% | 7 | 13 | 204.1 / 204.1 | 21/21 | 7.76 (primary) | 7.98 | Image layout / visual grammar | top85 |
| 8 | `sf_201006_0000645` | 21.769 | 3.6% | 64.2% | 8 | 14 | 164.4 / 164.4 | 13/13 | 8.77 (primary) | 8.9 | Unverified prior | prior-in-85 top85 |
| 9 | `osm_46264108` | 15.783 | 2.6% | 66.8% | 10 | 9 | 169.6 / 259.9 | 2/8 | 3.75 (primary) | 9.85 | Image hierarchy + LiDAR levels | top85 |
| 10 | `osm_573287814` | 15.499 | 2.5% | 69.4% | 9 | 18 | 70.9 / 100.5 | 9/10 | 12.86 (tertiary) | 12.99 | Image layout / visual grammar | top85 |
| 11 | `sf_201006_0002834` | 15.040 | 2.5% | 71.8% | 12 | 19 | 102.9 / 102.9 | 3/3 | 5.28 (residential) | 5.42 | Image hierarchy + LiDAR levels | top85 |
| 12 | `sf_201006_0001103` | 14.973 | 2.5% | 74.3% | 11 | 17 | 53.2 / 68.5 | 6/9 | 11.02 (primary) | 11.25 | Unverified prior | prior-in-85 top85 |
| 13 | `osm_129176908` | 13.509 | 2.2% | 76.5% | 13 | 23 | 94.4 / 144.7 | 7/11 | 12.28 (secondary) | 12.29 | Unverified prior | prior-in-85 top85 |
| 14 | `sf_201006_0002333` | 11.354 | 1.9% | 78.4% | 15 | 8 | 80.1 / 147.4 | 3/7 | 5.72 (unclassified) | 6.03 | Unverified prior | prior-in-85 top85 |
| 15 | `sf_201006_0001817` | 10.183 | 1.7% | 80.1% | 14 | 25 | 35.4 / 72.3 | 1/7 | 10.99 (primary) | 11.03 | Image hierarchy + LiDAR levels | top85 |
| 16 | `sf_201006_0006267` | 9.317 | 1.5% | 81.6% | 17 | 26 | 75.3 / 98.0 | 6/8 | 5.42 (residential) | 5.42 | Unverified prior | prior-in-85 top85 |
| 17 | `osm_149167949` | 8.241 | 1.4% | 82.9% | 16 | 29 | 27.1 / 54.3 | 1/2 | 9.94 (primary) | 10.45 | Image layout / visual grammar | top85 |
| 18 | `sf_201006_0006942` | 8.164 | 1.3% | 84.3% | 19 | 27 | 20.7 / 65.4 | 1/3 | 6.18 (residential) | 6.22 | Unverified prior | prior-in-85 top85 |
| 19 | `sf_201006_0002604` | 7.351 | 1.2% | 85.5% | 21 | 32 | 51.9 / 93.5 | 2/5 | 11.8 (tertiary) | 11.82 | Image hierarchy + LiDAR levels | top85 |
| 20 | `sf_201006_0000227` | 7.108 | 1.2% | 86.7% | 20 | 6 | 299.4 / 318.0 | 27/29 | 7.02 (primary) | 10.09 | Unverified prior | prior top95 |
| 21 | `sf_201006_0007874` | 6.906 | 1.1% | 87.8% | 18 | 33 | 54.1 / 60.6 | 2/3 | 10.54 (primary) | 10.63 | Image hierarchy + LiDAR levels | top95 |
| 22 | `sf_201006_0003024` | 6.690 | 1.1% | 88.9% | 24 | 22 | 63.4 / 88.3 | 3/4 | 5.2 (unclassified) | 5.59 | Unverified prior | prior top95 |
| 23 | `sf_201006_0002920` | 6.656 | 1.1% | 90.0% | 23 | 34 | 25.9 / 49.1 | 1/2 | 11.89 (tertiary) | 12.23 | Image hierarchy + LiDAR levels | top95 |
| 24 | `sf_201006_0005523` | 6.532 | 1.1% | 91.0% | 22 | 35 | 45.5 / 45.5 | 2/2 | 12.15 (secondary) | 12.25 | Unverified prior | prior top95 |
| 25 | `sf_201006_0003782` | 6.283 | 1.0% | 92.1% | 25 | 36 | 59.2 / 59.2 | 3/3 | 12.44 (tertiary) | 12.91 | Image hierarchy + LiDAR levels | top95 |
| 26 | `sf_201006_0003596` | 5.610 | 0.9% | 93.0% | 27 | 16 | 78.7 / 78.7 | 4/4 | 4.51 (residential) | 9.86 | Unverified prior | prior top95 |
| 27 | `sf_201006_0004743` | 5.562 | 0.9% | 93.9% | 26 | 37 | 57.1 / 74.4 | 5/6 | 10.04 (primary) | 10.11 | Image layout / visual grammar | top95 |
| 28 | `sf_201006_0001102` | 4.190 | 0.7% | 94.6% | 29 | 41 | 27.0 / 53.4 | 1/2 | 13.25 (tertiary) | 13.41 | Unverified prior | prior top95 |
| 29 | `sf_201006_0001245` | 4.186 | 0.7% | 95.3% | 30 | 40 | 63.6 / 113.5 | 5/11 | 12.83 (tertiary) | 12.84 | Unverified prior | prior top95 |
| 30 | `sf_201006_0002680` | 3.946 | 0.6% | 95.9% | 31 | 30 | 36.6 / 73.2 | 1/2 | 12.86 (secondary) | 13.28 | Unverified prior | prior |
| 31 | `sf_201006_0002044` | 3.686 | 0.6% | 96.5% | 32 | 39 | 18.7 / 74.9 | 1/3 | 13.16 (tertiary) | 13.16 | Unverified prior | prior |
| 32 | `sf_201006_0056871` | 3.546 | 0.6% | 97.1% | 28 | 43 | 87.1 / 87.1 | 3/3 | 23.26 (primary) | 23.44 | Unverified prior | prior |
| 33 | `sf_201006_0002005` | 3.284 | 0.5% | 97.7% | 33 | 45 | 17.8 / 35.6 | 1/2 | 13.08 (tertiary) | 13.08 | Unverified prior | prior |
| 34 | `sf_201006_0001270` | 2.955 | 0.5% | 98.1% | 34 | 12 | 80.5 / 186.2 | 4/10 | 6.36 (residential) | 13.11 | Unverified prior | prior |
| 35 | `sf_201006_0003021` | 2.686 | 0.4% | 98.6% | 35 | 47 | 19.8 / 39.9 | 1/2 | 13.08 (tertiary) | 13.16 | Unverified prior | prior |
| 36 | `sf_201006_0003726` | 1.897 | 0.3% | 98.9% | 37 | 38 | 49.7 / 65.6 | 6/9 | 11.08 (primary) | 11.25 | Unverified prior | prior |
| 37 | `osm_945538326` | 1.754 | 0.3% | 99.2% | 36 | 7 | 120.1 / 176.7 | 7/10 | 9.63 (primary) | 16.0 | Unverified prior | prior |
| 38 | `osm_149253145` | 1.123 | 0.2% | 99.4% | 38 | 11 | 101.6 / 173.4 | 9/13 | 7.1 (residential) | 13.08 | Unverified prior | prior |
| 39 | `osm_129176919` | 1.095 | 0.2% | 99.6% | 39 | 48 | 31.1 / 35.8 | 4/5 | 13.04 (secondary) | 13.23 | Unverified prior | prior |
| 40 | `sf_201006_0015276` | 0.951 | 0.2% | 99.7% | 40 | 49 | 14.2 / 28.4 | 1/2 | 11.87 (residential) | 12.03 | Unverified prior | prior |
| 41 | `sf_201006_0006227` | 0.886 | 0.1% | 99.9% | 41 | 44 | 12.8 / 25.5 | 1/2 | 11.54 (tertiary) | 13.86 | Unverified prior | prior |
| 42 | `sf_201006_0002572` | 0.328 | 0.1% | 99.9% | 42 | 21 | 27.6 / 89.5 | 1/3 | 11.55 (tertiary) | 23.49 | Image hierarchy + LiDAR levels |  |
| 43 | `sf_201006_0010890` | 0.251 | 0.0% | 99.9% | 43 | 31 | 15.2 / 75.4 | 1/5 | 6.33 (residential) | 14.32 | Unverified prior | prior |
| 44 | `sf_201006_0004051` | 0.217 | 0.0% | 100.0% | 44 | 28 | 41.9 / 100.3 | 4/8 | 12.29 (secondary) | 26.39 | Unverified prior | prior |
| 45 | `sf_201006_0001497` | 0.064 | 0.0% | 100.0% | 45 | 15 | 37.8 / 131.9 | 1/9 | 5.98 (unclassified) | 17.21 | Unverified prior | prior |
| 46 | `sf_201006_0019339` | 0.025 | 0.0% | 100.0% | 46 | 51 | 11.1 / 30.6 | 2/5 | 45.6 (secondary) | 45.69 | Unverified prior | prior |
| 47 | `sf_201006_0005190` | 0.006 | 0.0% | 100.0% | 47 | 42 | 12.1 / 55.9 | 1/3 | 4.78 (residential) | 33.73 | Image hierarchy + LiDAR levels |  |
| 48 | `osm_129176917` | 0.000 | 0.0% | 100.0% | 48 | 53 | 0.0 / 54.9 | 0/5 | 45.04 (secondary) | 45.16 | Unverified prior | zero-at-200 prior |
| 49 | `osm_346093777` | 0.000 | 0.0% | 100.0% | 49 | 24 | 0.0 / 134.8 | 0/8 | 5.68 (unclassified) | 76.4 | Unverified prior | zero-at-200 prior |
| 50 | `sf_201006_0001461` | 0.000 | 0.0% | 100.0% | 50 | 54 | 0.0 / 131.8 | 0/5 | 12.39 (secondary) | 54.77 | Unverified prior | never-visible prior |
| 51 | `sf_201006_0001907` | 0.000 | 0.0% | 100.0% | 51 | 55 | 0.0 / 144.1 | 0/7 | 14.48 (secondary) | 97.78 | Unverified prior | never-visible prior |
| 52 | `sf_201006_0002014` | 0.000 | 0.0% | 100.0% | 52 | 56 | 0.0 / 120.0 | 0/6 | 12.82 (secondary) | 104.11 | Unverified prior | never-visible prior |
| 53 | `sf_201006_0003254` | 0.000 | 0.0% | 100.0% | 53 | 57 | 0.0 / 83.8 | 0/4 | 52.87 (secondary) | 53.06 | Unverified prior | never-visible prior |
| 54 | `sf_201006_0005658` | 0.000 | 0.0% | 100.0% | 54 | 46 | 0.0 / 42.1 | 0/7 | 11.37 (tertiary) | 34.48 | Unverified prior | zero-at-200 prior |
| 55 | `sf_201006_0006801` | 0.000 | 0.0% | 100.0% | 55 | 20 | 0.0 / 85.2 | 0/9 | 4.97 (residential) | 51.72 | Image hierarchy + LiDAR levels | zero-at-200 |
| 56 | `sf_201006_0028162` | 0.000 | 0.0% | 100.0% | 56 | 50 | 0.0 / 56.4 | 0/6 | 27.57 (residential) | 29.33 | Unverified prior | zero-at-200 prior |
| 57 | `sf_201006_0029858` | 0.000 | 0.0% | 100.0% | 57 | 58 | 0.0 / 56.5 | 0/4 | 23.55 (residential) | 34.11 | Unverified prior | never-visible prior |
| 58 | `sf_201006_0136107` | 0.000 | 0.0% | 100.0% | 58 | 52 | 0.0 / 49.2 | 0/6 | 72.26 (unclassified) | 97.18 | Unverified prior | zero-at-200 prior |

Scores are geometric predictions of driver-visible facade area, not verified sightings; occlusion is a 2D footprint approximation.
