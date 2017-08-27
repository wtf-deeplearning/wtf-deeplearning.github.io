# WTF Deep Learning!!!

## Table Of Content
- [Github](#github)
- [Paper](#paper)
	- [Survey Review](#survey-review)
	- [Theory Future](#theory-future)
	- [Optimization Regularization](#optimization-regularization)
	- [NetworkModels](#networkmodels)
	- [Image](#image)
	- [Caption](#caption)
	- [Video Human Activity](#video-human-activity)
	- [Word Embedding](#word-embedding)
	- [Machine Translation QnA](#machine-translation-qna)
	- [Speech Etc](#speech-etc)
	- [RL Robotics](#rl-robotics)
	- [Unsupervised](#unsupervised)
	- [Hardware Software](#hardware-software)
	- [Bayesian](#bayesian)
- [License](#license)

## Github
~~~bash
git clone https://github.com/wtf-deeplearning/wtf-deeplearning.github.io.git
~~~

## Paper
### Survey Review
- Deep learning (2015), Y. LeCun, Y. Bengio and G. Hinton [[pdf]](survey-review/NatureDeepReview.pdf)
source: https://www.cs.toronto.edu/~hinton/absps/NatureDeepReview.pdf
- Deep learning in neural networks: An overview (2015), J. Schmidhuber [[pdf]](survey-review/DeepLearningInNeuralNetworksOverview.JSchmidhuber2015.pdf)
source: http://www2.econ.iastate.edu/tesfatsi/DeepLearningInNeuralNetworksOverview.JSchmidhuber2015.pdf
- Representation learning: A review and new perspectives (2013), Y. Bengio et al. [[pdf]](survey-review/BengioETAL12.pdf)
source : http://www.cl.uni-heidelberg.de/courses/ws14/deepl/BengioETAL12.pdf

### Theory Future
- Distilling the knowledge in a neural network (2015), G. Hinton et al. [[pdf]](theory-future/1503.02531.pdf)
source : http://arxiv.org/pdf/1503.02531
- Deep neural networks are easily fooled: High confidence predictions for unrecognizable images (2015), A. Nguyen et al. [[pdf]](theory-future/1412.1897.pdf)
source : http://arxiv.org/pdf/1412.1897
- How transferable are features in deep neural networks? (2014), J. Yosinski et al. *(Bengio)* [[pdf]](theory-future/5347-how-transferable-are-features-in-deep-neural-networks.pdf)
source : http://papers.nips.cc/paper/5347-how-transferable-are-features-in-deep-neural-networks.pdf
- Why does unsupervised pre-training help deep learning (2010), E. Erhan et al. *(Bengio)* [[pdf]](theory-future/AISTATS2010_ErhanCBV10.pdf)
source : http://machinelearning.wustl.edu/mlpapers/paper_files/AISTATS2010_ErhanCBV10.pdf
- Understanding the difficulty of training deep feedforward neural networks (2010), X. Glorot and Y. Bengio [[pdf]](theory-future/AISTATS2010_GlorotB10.pdf)
source : http://machinelearning.wustl.edu/mlpapers/paper_files/AISTATS2010_GlorotB10.pdf

### Optimization Regularization
- Taking the human out of the loop: A review of bayesian optimization (2016), B. Shahriari et al. [[pdf]](optimization-regularization/BayesOptLoop.pdf)
source : https://www.cs.ox.ac.uk/people/nando.defreitas/publications/BayesOptLoop.pdf
- Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift (2015), S. Loffe and C. Szegedy [[pdf]](optimization-regularization/1502.03167.pdf)
source : http://arxiv.org/pdf/1502.03167
- Delving deep into rectifiers: Surpassing human-level performance on imagenet classification (2015), K. He et al. [[pdf]](optimization-regularization/He_Delving_Deep_into_ICCV_2015_paper.pdf)
source : http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf
- Dropout: A simple way to prevent neural networks from overfitting (2014), N. Srivastava et al. *(Hinton)* [[pdf]](optimization-regularization/srivastava14a.pdf)
source : http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf
- Adam: A method for stochastic optimization (2014), D. Kingma and J. Ba [[pdf]](optimization-regularization/1412.6980.pdf)
source : http://arxiv.org/pdf/1412.6980
- Regularization of neural networks using dropconnect (2013), L. Wan et al. *(LeCun)* [[pdf]](optimization-regularization/icml2013_wan13.pdf)
source : http://machinelearning.wustl.edu/mlpapers/paper_files/icml2013_wan13.pdf
- Improving neural networks by preventing co-adaptation of feature detectors (2012), G. Hinton et al. [[pdf]](optimization-regularization/1207.0580.pdf)
source : http://arxiv.org/pdf/1207.0580.pdf
- Spatial pyramid pooling in deep convolutional networks for visual recognition (2014), K. He et al. [[pdf]](optimization-regularization/1406.4729)
source : http://arxiv.org/pdf/1406.4729
- Random search for hyper-parameter optimization (2012) J. Bergstra and Y. Bengio [[pdf]](optimization-regularization/bergstra12a.pdf)
source : http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a

### NetworkModels
- Deep residual learning for image recognition (2016), K. He et al. *(Microsoft)* [[pdf]](networkmodels/1512.03385)
source : http://arxiv.org/pdf/1512.03385
- Going deeper with convolutions (2015), C. Szegedy et al. *(Google)* [[pdf]](networkmodels/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf)
source : http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Szegedy_Going_Deeper_With_2015_CVPR_paper.pdf
- Fast R-CNN (2015), R. Girshick [[pdf]](networkmodels/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf)
source : http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf
- Very deep convolutional networks for large-scale image recognition (2014), K. Simonyan and A. Zisserman [[pdf]](networkmodels/1409.1556.pdf)
source : http://arxiv.org/pdf/1409.1556
- Fully convolutional networks for semantic segmentation (2015), J. Long et al. [[pdf]](networkmodels/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf)
source : http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf
- OverFeat: Integrated recognition, localization and detection using convolutional networks (2014), P. Sermanet et al. *(LeCun)* [[pdf]](networkmodels/1312.6229.pdf)
source : http://arxiv.org/pdf/1312.6229
- Visualizing and understanding convolutional networks (2014), M. Zeiler and R. Fergus [[pdf]](networkmodels/1311.2901.pdf)
source : http://arxiv.org/pdf/1311.2901
- Maxout networks (2013), I. Goodfellow et al. *(Bengio)* [[pdf]](networkmodels/1302.4389v4.pdf)
source : http://arxiv.org/pdf/1302.4389v4
- ImageNet classification with deep convolutional neural networks (2012), A. Krizhevsky et al. *(Hinton)* [[pdf]](networkmodels/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
source : http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
- Large scale distributed deep networks (2012), J. Dean et al. [[pdf]](networkmodels/4687-large-scale-distributed-deep-networks.pdf)
source : http://papers.nips.cc/paper/4687-large-scale-distributed-deep-networks.pdf
- Deep sparse rectifier neural networks (2011), X. Glorot et al. *(Bengio)* [[pdf]](networkmodels/AISTATS2011_GlorotBB11.pdf)
source : http://machinelearning.wustl.edu/mlpapers/paper_files/AISTATS2011_GlorotBB11.pdf

### Image
- Imagenet large scale visual recognition challenge (2015), O. Russakovsky et al. [[pdf]](image/1409.0575)
source : http://arxiv.org/pdf/1409.0575
- Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks (2015), S. Ren et al. [[pdf]](image/5638-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.pdf)
source : http://papers.nips.cc/paper/5638-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.pdf
- DRAW: A recurrent neural network for image generation (2015), K. Gregor et al. [[pdf]](image/1502.04623.pdf)
source : http://arxiv.org/pdf/1502.04623
- Rich feature hierarchies for accurate object detection and semantic segmentation (2014), R. Girshick et al. [[pdf]](image/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf)
source : http://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf
- Learning and transferring mid-Level image representations using convolutional neural networks (2014), M. Oquab et al. [[pdf]](image/Oquab_Learning_and_Transferring_2014_CVPR_paper.pdf)
source : http://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Oquab_Learning_and_Transferring_2014_CVPR_paper.pdf
- DeepFace: Closing the Gap to Human-Level Performance in Face Verification (2014), Y. Taigman et al. *(Facebook)* [[pdf]](image/Taigman_DeepFace_Closing_the_2014_CVPR_paper.pdf)
source : http://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Taigman_DeepFace_Closing_the_2014_CVPR_paper.pdf
- Decaf: A deep convolutional activation feature for generic visual recognition (2013), J. Donahue et al. [[pdf]](image/1310.1531.pdf)
source : http://arxiv.org/pdf/1310.1531
- Learning Hierarchical Features for Scene Labeling (2013), C. Farabet et al. *(LeCun)* [[pdf]](image/farabet-pami-13.pdf)
source : https://hal-enpc.archives-ouvertes.fr/docs/00/74/20/77/PDF/farabet-pami-13.pdf
- Learning hierarchical invariant spatio-temporal features for action recognition with independent subspace analysis (2011), Q. Le et al. [[pdf]](image/cvpr_LeZouYeungNg11.pdf)
source : http://robotics.stanford.edu/~wzou/cvpr_LeZouYeungNg11.pdf
- Learning mid-level features for recognition (2010), Y. Boureau *(LeCun)* [[pdf]](image/boureau-cvpr-10.pdf)
source : http://ece.duke.edu/~lcarin/boureau-cvpr-10.pdf

### Caption
- Show, attend and tell: Neural image caption generation with visual attention (2015), K. Xu et al. *(Bengio)* [[pdf]](caption/1502.03044.pdf)
source : http://arxiv.org/pdf/1502.03044
- Show and tell: A neural image caption generator (2015), O. Vinyals et al. [[pdf]](caption/Vinyals_Show_and_Tell_2015_CVPR_paper.pdf)
source : http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Vinyals_Show_and_Tell_2015_CVPR_paper.pdf
- Long-term recurrent convolutional networks for visual recognition and description (2015), J. Donahue et al. [[pdf]](caption/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf) 
source : http://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf
- Deep visual-semantic alignments for generating image descriptions (2015), A. Karpathy and L. Fei-Fei [[pdf]](caption/Karpathy_Deep_Visual-Semantic_Alignments_2015_CVPR_paper.pdf)
source : http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Karpathy_Deep_Visual-Semantic_Alignments_2015_CVPR_paper.pdf

### Video Human Activity
- Large-scale video classification with convolutional neural networks (2014), A. Karpathy et al. *(FeiFei)* [[pdf]](video-human-activity/karpathy14.pdf)
source : vision.stanford.edu/pdf/karpathy14.pdf
- A survey on human activity recognition using wearable sensors (2013), O. Lara and M. Labrador [[pdf]](video-human-activity/Lara%20-%20Human%20Activity%20Recognition%20-%202013.pdf)
source : http://romisatriawahono.net/lecture/rm/survey/computer%20vision/Lara%20-%20Human%20Activity%20Recognition%20-%202013.pdf
- 3D convolutional neural networks for human action recognition (2013), S. Ji et al. [[pdf]](video-human-activity/icml2010_JiXYY10.pdf)
source : http://machinelearning.wustl.edu/mlpapers/paper_files/icml2010_JiXYY10.pdf
- Deeppose: Human pose estimation via deep neural networks (2014), A. Toshev and C. Szegedy [[pdf]](video-human-activity/Toshev_DeepPose_Human_Pose_2014_CVPR_paper.pdf)
source : http://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Toshev_DeepPose_Human_Pose_2014_CVPR_paper.pdf
- Action recognition with improved trajectories (2013), H. Wang and C. Schmid [[pdf]](video-human-activity/Wang_Action_Recognition_with_2013_ICCV_paper.pdf)
source : http://www.cv-foundation.org/openaccess/content_iccv_2013/papers/Wang_Action_Recognition_with_2013_ICCV_paper.pdf

### Word Embedding
- Glove: Global vectors for word representation (2014), J. Pennington et al. [[pdf]](word-embedding/nn-pres.pdf)
source : http://llcao.net/cu-deeplearning15/presentation/nn-pres.pdf
- Sequence to sequence learning with neural networks (2014), I. Sutskever et al. [[pdf]](word-embedding/5346-sequence-to-sequence-learning-with-neural-networks.pdf)
source : http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf
- Distributed representations of sentences and documents (2014), Q. Le and T. Mikolov [[pdf]](word-embedding/1405.4053.pdf) *(Google)*
source : http://arxiv.org/pdf/1405.4053
- Distributed representations of words and phrases and their compositionality (2013), T. Mikolov et al. *(Google)* [[pdf]](word-embedding/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
source : http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf
- Efficient estimation of word representations in vector space (2013), T. Mikolov et al. *(Google)* [[pdf]](word-embedding/1301.3781.pdf)
source : http://arxiv.org/pdf/1301.3781
- Word representations: a simple and general method for semi-supervised learning (2010), J. Turian *(Bengio)* [[pdf]](word-embedding/P10-1040.pdf)
source : http://www.anthology.aclweb.org/P/P10/P10-1040.pdf

### Machine Translation QnA
- Towards ai-complete question answering: A set of prerequisite toy tasks (2015), J. Weston et al. [[pdf]](machine-translation/1502.05698.pdf)
source : http://arxiv.org/pdf/1502.05698
- Neural machine translation by jointly learning to align and translate (2014), D. Bahdanau et al. *(Bengio)* [[pdf]](machine-translation/1409.0473.pdf)
source : http://arxiv.org/pdf/1409.0473
- Learning phrase representations using RNN encoder-decoder for statistical machine translation (2014), K. Cho et al. *(Bengio)* [[pdf]](machine-translation/1406.1078.pdf)
source : http://arxiv.org/pdf/1406.1078
- A convolutional neural network for modelling sentences (2014), N. kalchbrenner et al. [[pdf]](machine-translation/1404.2188v1.pdf)
source : http://arxiv.org/pdf/1404.2188v1
- Convolutional neural networks for sentence classification (2014), Y. Kim [[pdf]](machine-translation/1408.5882.pdf)
source : http://arxiv.org/pdf/1408.5882
- The stanford coreNLP natural language processing toolkit (2014), C. Manning et al. [[pdf]](machine-translation/acl2014-corenlp.pdf)
source : http://www.surdeanu.info/mihai/papers/acl2014-corenlp.pdf
- Recursive deep models for semantic compositionality over a sentiment treebank (2013), R. Socher et al. [[pdf]](machine-translation/10.1.1.383.1327.pdf)
source : http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.383.1327&rep=rep1&type=pdf
- Natural language processing (almost) from scratch (2011), R. Collobert et al. [[pdf]](machine-translation/1103.0398.pdf)
source : http://arxiv.org/pdf/1103.0398
- Recurrent neural network based language model (2010), T. Mikolov et al. [[pdf]](machine-translation/rnnlm_mikolov.pdf)
source : http://www.fit.vutbr.cz/research/groups/speech/servite/2010/rnnlm_mikolov.pdf

### Speech Etc
- Speech recognition with deep recurrent neural networks (2013), A. Graves *(Hinton)* [[pdf]](speech/1303.5778.pdf)
source : http://arxiv.org/pdf/1303.5778.pdf
- Deep neural networks for acoustic modeling in speech recognition: The shared views of four research groups (2012), G. Hinton et al. [[pdf]](speech/SPM_DNN_12.pdf)
source : http://www.cs.toronto.edu/~asamir/papers/SPM_DNN_12.pdf
- Context-dependent pre-trained deep neural networks for large-vocabulary speech recognition (2012) G. Dahl et al. [[pdf]](speech/10.1.1.337.7548.pdf)
source : http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.337.7548&rep=rep1&type=pdf

## RL Robotics
- Mastering the game of Go with deep neural networks and tree search, D. Silver et al. *(DeepMind)* [[pdf]](robotics/AlphaGoNaturePaper.pdf)
source : https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf
- Human-level control through deep reinforcement learning (2015), V. Mnih et al. *(DeepMind)* [[pdf]](robotics/nature14236.pdf)
source : http://www.davidqiu.com:8888/research/nature14236.pdf
- Deep learning for detecting robotic grasps (2015), I. Lenz et al. [[pdf]](robotics/lenz_lee_saxena_deep_learning_grasping_ijrr2014.pdf)
source : http://www.cs.cornell.edu/~asaxena/papers/lenz_lee_saxena_deep_learning_grasping_ijrr2014.pdf
- Playing atari with deep reinforcement learning (2013), V. Mnih et al. *(DeepMind)* [[pdf]](robotics/1312.5602.pdf)
source : http://arxiv.org/pdf/1312.5602.pdf

### Unsupervised
- Building high-level features using large scale unsupervised learning (2013), Q. Le et al. [[pdf]](unsupervised/1112.6209.pdf)
source : http://arxiv.org/pdf/1112.6209
- Contractive auto-encoders: Explicit invariance during feature extraction (2011), S. Rifai et al. *(Bengio)* [[pdf]](unsupervised/ICML2011Rifai_455.pdf)
source : http://machinelearning.wustl.edu/mlpapers/paper_files/ICML2011Rifai_455.pdf
- An analysis of single-layer networks in unsupervised feature learning (2011), A. Coates et al. [[pdf]](unsupervised/AISTATS2011_CoatesNL11.pdf)
source : http://machinelearning.wustl.edu/mlpapers/paper_files/AISTATS2011_CoatesNL11.pdf
- Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion (2010), P. Vincent et al. *(Bengio)* [[pdf]](unsupervised/vincent10a.pdf)
source : http://www.jmlr.org/papers/volume11/vincent10a/vincent10a.pdf
- A practical guide to training restricted boltzmann machines (2010), G. Hinton [[pdf]](unsupervised/guideTR.pdf)
source : http://www.csri.utoronto.ca/~hinton/absps/guideTR.pdf

### Hardware Software
- TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems (2016), M. Abadi et al. *(Google)* [[pdf]](hardware-software/1603.04467.pdf)
source : http://arxiv.org/pdf/1603.04467
- MatConvNet: Convolutional neural networks for matlab (2015), A. Vedaldi and K. Lenc [[pdf]](hardware-software/1412.4564.pdf)
source : http://arxiv.org/pdf/1412.4564
- Caffe: Convolutional architecture for fast feature embedding (2014), Y. Jia et al. [[pdf]](hardware-software/1408.5093.pdf)
source : http://arxiv.org/pdf/1408.5093
- Theano: new features and speed improvements (2012), F. Bastien et al. *(Bengio)* [[pdf]](hardware-software/1211.5590.pdf)
source : http://arxiv.org/pdf/1211.5590

### Bayesian
#### 2013: 
 1. Deep gaussian processes|Andreas C. Damianou,Neil D. Lawrence|2013 <br>
    Source: http://www.jmlr.org/proceedings/papers/v31/damianou13a.pdf

#### 2014: 
 1. Avoiding pathologies in very deep networks|D Duvenaud, O Rippel, R Adams|2014 <br>
    Source: http://www.jmlr.org/proceedings/papers/v33/duvenaud14.pdf
 2. Nested variational compression in deep Gaussian processes|J Hensman, ND Lawrence|2014
    Source: https://arxiv.org/abs/1412.1370

#### 2015: 
 1. On Modern Deep Learning and Variational Inference  |Yarin Gal, Zoubin Ghahramani|2015 <br>
    Source: http://www.approximateinference.org/accepted/GalGhahramani2015.pdf
 2. Rapid Prototyping of Probabilistic Models: Emerging Challenges in Variational Inference   |Yarin Gal, |2015<br>
    Source: http://www.approximateinference.org/accepted/Gal2015.pdf 
 3. Bayesian Convolutional Neural Networks with Bernoulli Approximate Variational Inference |Yarin Gal, Zoubin Ghahramani|2015<br>
    Source: http://arxiv.org/abs/1506.02158
 4. Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning  |Yarin Gal, Zoubin Ghahramani|2015<br>
    Source: http://arxiv.org/abs/1506.02142
 5. Dropout as a Bayesian Approximation: Insights and Applications     |Yarin Gal, |2015
    Source: https://sites.google.com/site/deeplearning2015/33.pdf?attredirects=0
 6. Bayesian Convolutional Neural Networks with Bernoulli Approximate Variational Inference |Yarin Gal, Zoubin Ghahramani|2015<br>
    Source: http://arxiv.org/abs/1506.02158
 7. Scalable Variational Gaussian Process Classification|J Hensman, AGG Matthews, Z Ghahramani|2015
    Source: http://www.jmlr.org/proceedings/papers/v38/hensman15.pdf
 
#### 2016:
 1. Relativistic Monte Carlo | Xiaoyu Lu| 2016 <br>
    Source: https://arxiv.org/abs/1609.04388
 2. Risk versus Uncertainty in Deep Learning: Bayes, Bootstrap and the Dangers of Dropout | Ian Osband| 2016 <br>
    Source: http://bayesiandeeplearning.org/papers/BDL_4.pdf
 3. Semi-supervised deep kernel learning|Neal Jean, Michael Xie, Stefano Ermon|2016 <br>
    Source: http://bayesiandeeplearning.org/papers/BDL_5.pdf
 4. Categorical Reparameterization with Gumbel-Softmax| Eric Jang, Shixiang Gu,Ben Poole| 2016 <br>
    Source: http://bayesiandeeplearning.org/papers/BDL_8.pdf
    Video: https://www.youtube.com/watch?v=JFgXEbgcT7g
 5. Learning to Optimise: Using Bayesian Deep Learning for Transfer Learning in Optimisation| Jonas Langhabel,   Jannik Wolff| 2016<br> 
    Source: http://bayesiandeeplearning.org/papers/BDL_9.pdf
 6. One-Shot Learning in Discriminative Neural Networks| Jordan Burgess,James Robert Lloyd,Zoubin Ghahramani| 2016<br>
    Source: http://bayesiandeeplearning.org/papers/BDL_10.pdf
 7. Distributed Bayesian Learning with Stochastic Natural-gradient Expectation Propagation| Leonard Hasenclever,
Stefan Webb| 2016 <br>
    Source: http://bayesiandeeplearning.org/papers/BDL_11.pdf
 8. Knots in random neural networks| Kevin K. Chen| 2016 <br>
    Source: http://bayesiandeeplearning.org/papers/BDL_2.pdf
 9. Discriminative Bayesian neural networks know what they do not know | Christian Leibig, Siegfried Wahl| 2016 <br>
    Source: http://bayesiandeeplearning.org/papers/BDL_12.pdf
 10. Variational Inference in Neural Networks using an Approximate Closed-Form Objective|Wolfgang Roth and Franz Pernkopf|2016 <br>
    Source: http://bayesiandeeplearning.org/papers/BDL_13.pdf
 11. Combining sequential deep learning and variational Bayes for semi-supervised inference| Jos van der Westhuizen, Dr. Joan Lasenby| 2016 <br>
    Source: http://bayesiandeeplearning.org/papers/BDL_14.pdf
 12. Importance Weighted Autoencoders with Random Neural Network Parameters| Daniel Hernández-Lobato,Thang D. Bui,Yinzhen Li| 2016 
Stefan Webb| 2016 <br>
    Source: http://bayesiandeeplearning.org/papers/BDL_15.pdf
 13. Variational Graph Auto-Encoders| Thomas N. Kipf,Max Welling| 2016 <br>
    Source: http://bayesiandeeplearning.org/papers/BDL_16.pdf
 14. Dropout-based Automatic Relevance Determination| Dmitry Molchanov, Arseniy Ashuha, Dmitry Vetrov| 2016 <br>
    Source: http://bayesiandeeplearning.org/papers/BDL_18.pdf
 15. Scalable GP-LSTMs with Semi-Stochastic Gradients| Maruan Al-Shedivat, Andrew Gordon Wilson, Yunus Saatchi, Zhiting Hu and Eric P. Xing| 2016 <br>
    Source: http://bayesiandeeplearning.org/papers/BDL_19.pdf
 16. Approximate Inference for Deep Latent Gaussian Mixture Models|Eric Nalisnick, Lars Hertel and Padhraic Smyth|2016 <br>
    Source: http://bayesiandeeplearning.org/papers/BDL_20.pdf
 17. Learning to Draw Samples: With Application to Amortized MLE for Generative Adversarial Training | Dilin Wang, Yihao Feng and Qiang Liu| 2016 <br>
    Source: http://bayesiandeeplearning.org/papers/BDL_21.pdf 
    Video: https://www.youtube.com/watch?v=fi-UUQe2Pss
 18. Learning and Policy Search in Stochastic Dynamical Systems with Bayesian Neural Networks| Stefan Depeweg, José Miguel Hernández-Lobato, Finale Doshi-Velez and Steffen Udluft| 2016<br> 
    Source: https://arxiv.org/abs/1605.07127
 19. Accelerating Deep Gaussian Processes Inference with Arc-Cosine Kernels  | Kurt Cutajar, Edwin V. Bonilla, Pietro Michiardi and Maurizio Filippone| 2016 <br>
    Source: http://bayesiandeeplearning.org/papers/BDL_24.pdf
 20. Embedding Words as Distributions with a Bayesian Skip-gram Model | Arthur Bražinskas, Serhii Havrylov and Ivan Titov| 2016 <br>
    Source: http://bayesiandeeplearning.org/papers/BDL_25.pdf
 21. Variational Inference on Deep Exponential Family by using Variational Inferences on Conjugate Models|Mohammad Emtiyaz Khan and Wu Lin|2016 <br>
    Source: http://bayesiandeeplearning.org/papers/BDL_26.pdf
 22. Neural Variational Inference for Latent Dirichlet Allocation| Akash Srivastava and Charles Sutton| 2016 <br>
    Source: http://bayesiandeeplearning.org/papers/BDL_27.pdf
 23. Hierarchical Bayesian Neural Networks for Personalized Classification | Ajjen Joshi, Soumya Ghosh, Margrit Betke and Hanspeter Pfister| 2016<br> 
    Source: http://bayesiandeeplearning.org/papers/BDL_28.pdf
 24. Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles| Balaji Lakshminarayanan, Alexander Pritzel and Charles Blundell| 2016<br>
    Source: http://bayesiandeeplearning.org/papers/BDL_29.pdf
 25. Asynchronous Stochastic Gradient MCMC with Elastic Coupling| Jost Tobias Springenberg, Aaron Klein, Stefan Falkner and Frank Hutter| 2016 <br>
    Source: http://bayesiandeeplearning.org/papers/BDL_30.pdf
 26. The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables|Chris J. Maddison, Andriy Mnih and Yee Whye Teh| 2016 <br>
    Source: http://bayesiandeeplearning.org/papers/BDL_31.pdf
 27. Known Unknowns: Uncertainty Quality in Bayesian Neural Networks | Ramon Oliveira, Pedro Tabacof and Eduardo Valle| 2016 <br>
    Source: http://bayesiandeeplearning.org/papers/BDL_32.pdf
 28. Normalizing Flows on Riemannian Manifolds |Mevlana Gemici, Danilo Rezende and Shakir Mohamed|2016 <br>
    Source: http://bayesiandeeplearning.org/papers/BDL_33.pdf
 29. Posterior Distribution Analysis for Bayesian Inference in Neural Networks| Pavel Myshkov and Simon Julier| 2016 <br>
    Source: http://bayesiandeeplearning.org/papers/BDL_34.pdf
 30. Deep Bayesian Active Learning with Image Data| Yarin Gal, Riashat Islam and Zoubin Ghahramani| 2016<br> 
Stefan Webb| 2016 <br>
    Source: http://bayesiandeeplearning.org/papers/BDL_35.pdf
 31. Bottleneck Conditional Density Estimators|Rui Shu, Hung Bui and Mohammad Ghavamzadeh| 2016 <br>
    Source: http://bayesiandeeplearning.org/papers/BDL_36.pdf
 32. A Tighter Monte Carlo Objective with Renyi alpha-Divergence Measures| Stefan Webb and Yee Whye Teh| 2016 <br>
    Source: http://bayesiandeeplearning.org/papers/BDL_37.pdf
 33. Bayesian Neural Networks for Predicting Learning Curves| Aaron Klein, Stefan Falkner, Jost Tobias Springenberg and Frank Hutter| 2016 <br>
    Source: http://bayesiandeeplearning.org/papers/BDL_38.pdf
 34. Nested Compiled Inference for Hierarchical Reinforcement Learning|Tuan Anh Le, Atılım Güneş Baydin and Frank Wood|2016 <br>
    Source: http://bayesiandeeplearning.org/papers/BDL_41.pdf
 35. Open Problems for Online Bayesian Inference in Neural Networks | Robert Loftin and David Roberts| 2016 <br>
    Source: http://bayesiandeeplearning.org/papers/BDL_42.pdf
 36. Deep Probabilistic Programming| Dustin Tran, Matt Hoffman, Kevin Murphy, Rif Saurous, Eugene Brevdo, and David Blei| 2016<br> 
    Source: http://bayesiandeeplearning.org/papers/BDL_43.pdf
 37. Markov Chain Monte Carlo for Deep Latent Gaussian Models  |Matthew Hoffman| 2016 <br>
    Source: http://bayesiandeeplearning.org/papers/BDL_44.pdf
 38. Semi-supervised Active Learning with Deep Probabilistic Generative Models | Amar Shah and Zoubin Ghahramani| 2016<br> 
    Source: http://bayesiandeeplearning.org/papers/BDL_43.pdf
 39. Thesis: Uncertainty in Deep Learning  | Yarin Gal| PhD Thesis, 2016 <br>
    Source: http://mlg.eng.cam.ac.uk/yarin/thesis/thesis.pdf, Blog: http://mlg.eng.cam.ac.uk/yarin/blog_2248.html <br>
 40. Deep survival analysis|R. Ranganath, A. Perotte, N. Elhadad, and D. Blei|2016 <br>
    Source: http://www.cs.columbia.edu/~blei/papers/RanganathPerotteElhadadBlei2016.pdf
 41. Towards Bayesian Deep Learning: A Survey| Hao Wang, Dit-Yan Yeung|2016 <br>
    Source: https://arxiv.org/pdf/1604.01662
 #### 2017
 1.  Dropout Inference in Bayesian Neural Networks with Alpha-divergences |Yingzhen Li, Yarin Gal|2017 <br>
    Source: https://arxiv.org/abs/1703.02914
 2.  What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?  |Alex Kendall, Yarin Gal|2017 <br>
    Source: https://arxiv.org/abs/1703.04977

## License
[![CC0](http://mirrors.creativecommons.org/presskit/buttons/88x31/svg/cc-zero.svg)](https://creativecommons.org/publicdomain/zero/1.0/)
