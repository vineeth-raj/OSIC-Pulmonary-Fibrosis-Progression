# [OSIC-Pulmonary-Fibrosis-Progression](https://www.kaggle.com/c/osic-pulmonary-fibrosis-progression)
## Silver Medal Solution (67th place)

## Initial Thoughts:
This was quite interesting competition for me as there was a huge shakeup though it is only a tabular data competition but using the CT scans will give a boost at the same time it leads to a drastic drop in leaderboard when it is given higher priority than the tabular data.

## Overview:
Our final solution was an ensemble of efficient-net-b5(trained on CT-Scans), elastic-net(trained on tabular data) and quantile-regression-model(trained on tabular data).

![alt text](https://camo.githubusercontent.com/9ef04dbed1f513462e82394ab07f5c204791cb2b9913d67c77b2f84372504e95/68747470733a2f2f7777772e676f6f676c65617069732e636f6d2f646f776e6c6f61642f73746f726167652f76312f622f6b6167676c652d666f72756d2d6d6573736167652d6174746163686d656e74732f6f2f696e626f782532463335343331333925324633623166323436316461366433313635383830366531616365383431646435662532466c756e672e706e673f67656e65726174696f6e3d3136303230333330333439353830393926616c743d6d65646961)
