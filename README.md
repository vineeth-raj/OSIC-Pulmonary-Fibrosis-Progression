# [OSIC-Pulmonary-Fibrosis-Progression](https://www.kaggle.com/c/osic-pulmonary-fibrosis-progression)
## Silver Medal Solution (67th place)

## Initial Thoughts:
This was quite interesting competition for me as there was a huge shakeup though it is only a tabular data competition but using the CT scans will give a boost at the same time it leads to a drastic drop in leaderboard when it is given higher priority than the tabular data.In this competition we realized that it was a both regression and classification competition.

## Overview:
Our final solution was an ensemble of efficient-net-b5(trained on CT-Scans), elastic-net(trained on tabular data) and quantile-regression-model(trained on tabular data).

## Models:
Our final effnet model was efficient-b5-model trained with CT scan and the training was model -> GlobalAveragePooling -> Adding Gaussian Noise -> Dropout -> Dense with Adam Optimizer. So this outputs the slope and initial FVC(Forced Vital Capacity) and with that slope we calculated the FVC for later weeks based on the initial FVC. We trained this model for 50 epochs and got a CV score(mean absolute error) of 3.4775497118631997.

![alt text](https://camo.githubusercontent.com/9ef04dbed1f513462e82394ab07f5c204791cb2b9913d67c77b2f84372504e95/68747470733a2f2f7777772e676f6f676c65617069732e636f6d2f646f776e6c6f61642f73746f726167652f76312f622f6b6167676c652d666f72756d2d6d6573736167652d6174746163686d656e74732f6f2f696e626f782532463335343331333925324633623166323436316461366433313635383830366531616365383431646435662532466c756e672e706e673f67656e65726174696f6e3d3136303230333330333439353830393926616c743d6d65646961)

## Competition Metric:
We used Competition Metric as a CV for our models. The competition metric is:
                                                                          σclipped=max(σ,70),

                                                                    Δ=min(|FVCtrue−FVCpredicted|,1000),

                                                                    metric=−2–√Δσclipped−ln(2–√σclipped).

## Training:
This was trained using the following params:
  - Adam Optimizer with Reduce on Plateau Scheduler
  - Trained with an LR of 0.003 for 50 epochs and 5 folds
  - Batch size of 4 for the effnet-model
  - Next is we used quantile regression for the tabular data with a new quantile loss:
                                                      def new_asy_qloss(y_true,y_pred):
                                                          qs = [0.2, 0.50, 0.8]
                                                          q = tf.constant(np.array([qs]), dtype=tf.float32)
                                                          e = y_true - y_pred
                                                          epsilon = 0.8
                                                          v = tf.maximum( -(1-q)*(e+q*epsilon), q*(e-(1-q)*epsilon))
                                                          v1 = tf.maximum(v,0.0)
                                                          return K.mean(v1)
   - Trained the quantile regression model for 5 folds and 855 epochs.
   - Trained with Adam Optimizer and LR of 0.1
   - Next we used ElasticNet for the tabular data with alpha=0.3 and l1_ratio=0.8
   - We trained this for 10 folds
   - No batch accumulation or mixed precision
   - We used only kaggle kernels- one for training effnet and another for inference of effnet and online training and inference of remaining models.
  
  ![alt text](https://camo.githubusercontent.com/c87f991d9ebe682771b66cf6ddd07aaa7f5d3ab900dc4a66a137c2e3d60de63d/68747470733a2f2f7777772e676f6f676c65617069732e636f6d2f646f776e6c6f61642f73746f726167652f76312f622f6b6167676c652d666f72756d2d6d6573736167652d6174746163686d656e74732f6f2f696e626f78253246333534333133392532463639613139363766626533663430363764376664396636653565393133353063253246666f6c64732e706e673f67656e65726174696f6e3d3136303230333635363832383433313526616c743d6d65646961)
