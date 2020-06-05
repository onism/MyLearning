ref: COBRA: Contrastive Bi-Modal Representation Algorithm
https://arxiv.org/pdf/2005.03687v2.pdf

These joint embedding spaces fail to sufficiently reduce
the modality gap, which affects the performance in downstream tasks. 

1. We propose a novel joint cross-modal embedding framework called COBRA which represents the data across different modalities in a common manifold.
2. We utilize a set of loss functions in a novel way, which jointly preserve not only the relationship between different intra cross-modal data samples but also preserve the relationship between inter cross-modal data asamples
3. We empirically validate our model by achieving state-of-the-art on four diverse downstream tasks.

![framework](https://github.com/ovshake/cobra/blob/master/images/Architecture.JPG)

### Multi-modal Fusion
Early fusion techniques that are based on simple concatenation do not capture the intra modal relations well.

Late fusion techniaues on the other hand prioritize intra modal learning abilities compromising on cross-modal differentiability.

Literature suggests that cross modal tasks benefit more from learning a joint embedding space than employ multi-modal fusion techniques.

### Contrastive Learning Paradigms

## Model Architecture

Our goal is to represent the data in a common manifold, such that the classwise representations are modality invariant and discriminatory. 
**We use an autoencoder for each modality to generate representations that are high fidelity in nature. We utilize an orthogonal transform layer, which takes as input the hidden space representations from the encoders of each modality, and projects these representations into a joint space that is modality invariant and discriminates between classes well.**




