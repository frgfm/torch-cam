# TorchCAM: class activation explorer

TorchCAM provides a minimal yet flexible way to explore the spatial importance of features on your PyTorch model outputs. Check out the live demo on [HuggingFace Spaces](https://huggingface.co/spaces/frgfm/torch-cam) 🤗

<p align="center">
    <img src="https://github.com/frgfm/torch-cam/releases/download/v0.3.1/example.png" alt="CAM visualization" width="70%">
</p>

This project is meant for:

* ⚡ **exploration**: easily assess the influence of spatial features on your model's outputs
* 👩‍🔬 **research**: quickly implement your own ideas for new CAM methods

## CAM zoo

### Activation-based methods
   * CAM from ["Learning Deep Features for Discriminative Localization"](https://arxiv.org/pdf/1512.04150.pdf)
   * Score-CAM from ["Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks"](https://arxiv.org/pdf/1910.01279.pdf)
   * SS-CAM from ["SS-CAM: Smoothed Score-CAM for Sharper Visual Feature Localization"](https://arxiv.org/pdf/2006.14255.pdf)
   * IS-CAM from ["IS-CAM: Integrated Score-CAM for axiomatic-based explanations"](https://arxiv.org/pdf/2010.03023.pdf)

### Gradient-based methods
   * Grad-CAM from ["Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"](https://arxiv.org/pdf/1610.02391.pdf)
   * Grad-CAM++ from ["Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks"](https://arxiv.org/pdf/1710.11063.pdf)
   * Smooth Grad-CAM++ from ["Smooth Grad-CAM++: An Enhanced Inference Level Visualization Technique for Deep Convolutional Neural Network Models"](https://arxiv.org/pdf/1908.01224.pdf)
   * X-Grad-CAM from ["Axiom-based Grad-CAM: Towards Accurate Visualization and Explanation of CNNs"](https://arxiv.org/pdf/2008.02312.pdf)
   * Layer-CAM from ["LayerCAM: Exploring Hierarchical Class Activation Maps for Localization"](http://mmcheng.net/mftp/Papers/21TIP_LayerCAM.pdf)
