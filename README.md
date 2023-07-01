# Awesome-Brain-Decode

🧠 **A curated list of awesome brain decode datasets, papers and challenges.**

🚧 If there are any papers, datasets(recently published) that I missed, please let me know.

👩 Author: Hao Cong

📪 Contact: conghao0905@163.com

🍄 Status: Updating


## Contents

- [Datasets](#Datasets)

- [Papers](#Papers)

  - [Image](#Image-reconstruction)

  - [Video](#Video-reconstruction)

  - [Language](#Language-reconstruction)

- [Challenges](#Challenges)

- [Related Works](#Related-Works)

- [Tools](#Tools)

- [Courses](#Courses)


## Datasets 

**[`NSD`]**[A massive 7T fMRI dataset to bridge cognitive neuroscience and artificial in- telligence](https://www.nature.com/articles/s41593-021-00962-x)

**[`BOLD5000`]**[BOLD5000, a public fMRI dataset while viewing 5000 visual images](https://www.nature.com/articles/s41597-019-0052-3)

**[`fMRIonImageNet`]**[Generic decoding of seen and imagined objects using hierarchical visual features](https://www.nature.com/articles/ncomms15037)

**[`Vim-1`]**[This data set contains BOLD fMRI responses in human subjects viewing natural images](https://crcns.org/data-sets/vc/vim-1/about-vim-1)

**[`HCP`]**[Human Connectome Project](https://www.humanconnectome.org/)

**[`DIR`]**[Deep image reconstruction from human brain activity](https://openneuro.org/datasets/ds001506/versions/1.3.1)


**[`OpenNEURO-GOD`]**[Generic Object Decoding Dataset(fMRI on ImageNet)](https://openneuro.org/datasets/ds001246/versions/1.2.1)

**[`OpenNEURO`]**[An fMRI dataset for testing semantic language decoders](https://openneuro.org/datasets/ds004510/versions/1.1.0)


**[`MindBigData2022`]**[A Large Dataset of Brain Signals(EEG)](https://arxiv.org/pdf/2212.14746.pdf)

**[`fMRI-Video`]**[A large single-participant fMRI dataset for probing brain responses to naturalistic stimuli in space and time](https://www.biorxiv.org/content/10.1101/687681v1)

**[`EEGVisual2022`]**[A large and rich EEG dataset for modeling human visual object recognition](https://www.sciencedirect.com/science/article/pii/S1053811922008758#sec0033)

**[`Naturalistic Neuroimaging Database`]**[A ‘Naturalistic Neuroimaging Database’ for understanding the brain using ecological stimuli](https://www.biorxiv.org/content/10.1101/2020.05.22.110817v1)

**[`O3D`]**[The open diffusion data derivatives, brain data upcycling via integrated publishing of derivatives and reproducible open cloud services](https://www.nature.com/articles/s41597-019-0073-y)

**[`ThingsEEG-Text`]**[A large and rich EEG dataset for modeling human visual object recognition](https://osf.io/3jk45/)


## Papers

### **Reviews**

**[`Frontiersin 2021`]** [Natural Image Reconstruction from fMRI using Deep Learning: A Survey](https://arxiv.org/abs/2110.09006)

**[`Automation and Computing 2021`]** [fMRI-based Decoding of Visual Information from Human Brain Activity: A Brief Review](https://d-nb.info/1228844151/34)

**[`arXiv 2021`]** [Using Deep Learning for Visual Decoding and Reconstruction from Brain Activity: A Review](https://arxiv.org/abs/2108.04169)


### **Image reconstruction**

 **[`arXiv 2023`]**[Mind-eye: Reconstructing the Mind’s Eye: fMRI-to-Image with Contrastive Learning and Diffusion Priors](https://arxiv.org/abs/2305.18274)

- [code] *https://github.com/MedARC-AI/fMRI-reconstruction-NSD*


 **[`arXiv 2023`]**[Brain-Diffuser Natural scene reconstruction from fMRI signals using generative latent diffusion](http://arxiv.org/abs/2303.05334)

- [code] *https://github.com/ozcelikfu/brain-diffuser*


 **[`CVPR 2023`]**[Mind-vis: Seeing Beyond the Brain Conditional Diffusion Model with Sparse Masked Modeling for Vision Decoding](http://arxiv.org/abs/2211.06956)

- [code] *https://github.com/zjc062/mind-vis*


 **[`TPAMI 2023`]**[BraVL: Decoding Visual Neural Representations by Multimodal Learning of Brain-Visual-Linguistic Features](http://arxiv.org/abs/2210.06756)  

- [code] *https://github.com/ChangdeDu/BraVL*


 **[`CVPR 2023`]**[SD-brain: High-resolution image reconstruction with latent diffusion models from human brain activity](https://www.biorxiv.org/content/10.1101/2022.11.18.517004v3)  

- [code] *https://github.com/yu-takagi/StableDiffusionReconstruction*


 **[`NeurIPS 2022`]**[Mind Reader Reconstructing complex images from brain activities](http://arxiv.org/abs/2210.01769)  

- [code] *https://github.com/sklin93/mind-reader*

 **[`Neuroscience 2022`]**[Brain2Pix: Fully convolutional naturalistic video reconstruction from brain activity](https://www.biorxiv.org/content/10.1101/2021.02.02.429430v1)  

- [code] *https://github.com/neuralcodinglab/brain2pix*

 **[`NeuroImage 2022`]**[Self-Supervised Natural Image Reconstruction and Large-Scale Semantic Classification from Brain Activity](https://doi.org/10.1101/2020.09.06.284794)  

 **[`arXiv 2022`]**[Decoding natural image stimuli from fMRI data with a surface-based convolutional network](http://arxiv.org/abs/2212.02409)  

 **[`arXiv 2022`]**[The Brain-Inspired Decoder for Natural Visual Image Reconstruction](https://arxiv.org/abs/2207.08591)  

**[`IJCNN 2022`]**[Reconstruction of Perceived Images from fMRI Patterns and Semantic Brain Exploration using Instance-Conditioned GANs](http://arxiv.org/abs/2202.12692)  

- [code] *https://github.com/ozcelikfu/IC-GAN_fMRI_Reconstruction*

**[`IEEE Access 2021`]**[Perceived Image Decoding from Brain Activity Using Shared Information of Multi-subject fMRI Data](https://ieeexplore.ieee.org/document/9349437)

**[`NeuroImage 2021`]**[Reconstructing Seen Image from Brain Activity by Visually-guided Cognitive Representation and Adversarial Learning](https://arxiv.org/abs/1906.12181)

**[`arXiv 2020`]**[Reconstructing Natural Scenes from fMRI Patterns using BigBiGAN](https://arxiv.org/abs/2001.11761)

**[`Sci. Reports 2020`]**[Hyperrealistic neural decoding for reconstructing faces from fMRI activations via the GAN latent space](https://www.biorxiv.org/content/10.1101/2020.07.01.168849v3.full)

**[`NeurIPS 2019`]**[From voxels to pixels and back Self-supervision in natural-image reconstruction from fMRI](http://arxiv.org/abs/1907.02431) 

- [code] *https://github.com/WeizmannVision/ssfmri2im*


### **Video reconstruction**

**[`arXiv 2023`]**[Mind-video: Cinematic Mindscapes: High-quality Video Reconstruction from Brain Activity](https://arxiv.org/abs/2305.11675)

- [code] *https://github.com/jqin4749/MindVideo*


**[`arXiv 2023`]**[A Penny for Your (visual) Thoughts: Self-Supervised Reconstruction of Natural Movies from Brain Activity](https://arxiv.org/abs/2206.03544)

- [page] *https://www.wisdom.weizmann.ac.il/~vision/VideoReconstFromFMRI/*


**[`Cerebral Cortex 2022`]**[Reconstructing rapid natural vision with fMRI-conditional video generative adversarial network](https://academic.oup.com/cercor/article-abstract/32/20/4502/6515038)


**[`Cerebral Cortex 2018`]**[Neural encoding and decoding with deep learning for dynamic natural vision](https://arxiv.org/pdf/1608.03425.pdf)


### **Language reconstruction**

**[`Nature Neuroscience 2022`]**[Semantic reconstruction of continuous language from non-invasive brain recordings](https://www.biorxiv.org/content/10.1101/2022.09.29.509744v1)  

**[`Communications Biology 2022`]**[Brains and algorithms partially converge in natural language processing](https://www.nature.com/articles/s42003-022-03036-1)


## Challenges

🏆 **[`AlgonautsProject 2023`]**[How the Human Brain Makes Sense of Natural Scenes](http://algonauts.csail.mit.edu/)

🏆 [The SENSORIUM 2023 Competition](https://www.sensorium-competition.net/)

🏆 **[`DecMeg2014`]**[Predict visual stimuli from MEG recordings of human brain activity](https://www.kaggle.com/competitions/decoding-the-human-brain)


## Related works

### **Self-supervised learning**

[A Cookbook of Self-Supervised Learning](https://arxiv.org/abs/2304.12210)

### **Generative models**

[VAE:Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) [ [code](https://github.com/AntixK/PyTorch-VAE)]

[Very Deep VAEs](https://arxiv.org/abs/2011.10650) [ [code](https://github.com/openai/vdvae)]

[Versatile Diffusion](https://arxiv.org/abs/2211.08332) [ [code](https://github.com/SHI-Labs/Versatile-Diffusion)]


### **Multi-modal learning**

[CLIP: Contrastive language-image pre-training](https://arxiv.org/abs/2103.00020) [ [code](https://github.com/openai/CLIP)]


## Tools

[Huggingface:The AI community building the future](https://github.com/huggingface)

[AllenSDK: Code for processing and analyzing data in the Allen Brain Atlas](https://github.com/AllenInstitute/AllenSDK)

[Labml.ai: Deep learning paper implementations](https://github.com/labmlai/annotated_deep_learning_paper_implementations)

[OpenMMLab: Computer vision foundation](https://github.com/open-mmlab)

## Courses

[Neuromatch: Computational neuroscience course](https://compneuro.neuromatch.io/tutorials/intro.html)

[UCSD: Modern brain-computer interface design](https://sccn.ucsd.edu/wiki/Introduction_To_Modern_Brain-Computer_Interface_Design)

