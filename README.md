# Enforcing Isotropic Embedding Spaces for Contrastive-Generative Models on MNIST

## 1. Project Goal

The goal of this project is to investigate whether contrastive learning embeddings, when explicitly forced to follow an isotropic distribution, can support generative modeling through simple latent perturbations and decoding.

Specifically, we will study:

- Whether enforcing an isotropic (Gaussian or hyperspherical) embedding distribution prevents representational collapse.
- Whether smooth perturbations in the embedding space correspond to meaningful changes in generated images (MNIST digits).
- Whether a decoder trained on such embeddings can function as a generative model.

**Approach (Hyperspherical):** Keep embeddings normalized on a sphere, enforce uniform hyperspherical distribution via Gaussianized projections (SIGReg-style).

## 2. Model Architecture

We use three components:

### (a) Encoder $f(x)$

A CNN or small ResNet mapping MNIST images $x \in \mathbb{R}^{28\times28}$ to embeddings:
$$z=f_{\theta}(x)\in\mathbb{R}^d$$
Typical choices: $d=32, 64, 128.$

### (b) Contrastive loss LInfoNCE

Two augmentations per MNIST example (small affine transforms):
$$L_{\text{InfoNCE}}=-\log\frac{\exp(\text{sim}(z_i, z_i^+)/\tau)}{\sum_j \exp(\text{sim}(z_i, z_j)/\tau)}$$
($\text{sim}$ = cosine similarity)

### (c) Decoder $g_{\phi}(z)$

A small deconvolutional network mapping embeddings back to MNIST images:
$$\hat{x}=g_{\phi}(z)$$

## 3. Enforcing Isotropy in Embedding Space: Hyperspherical Uniform Distribution (SIGReg-Sphere)

### Goal

Contrastive learning already enforces unit-norm embeddings:
$$\tilde{z}=\frac{z}{\Vert z \Vert}$$
Uniform distribution on the sphere $S^{d-1}$ has the property that for large $d, a\cdot\tilde{z}\sim\mathcal{N}(0,1/d).$

### Method

1. Use L2-normalized embeddings:
   $$z_j=\frac{z_j}{\Vert z_j \Vert}$$

2. Random projection directions as before:
   $$u_{i, j}=a_i \cdot \tilde{z_j}$$

3. Scale by $\sqrt{d}$:
   $$\hat{u_{i, j}}=\sqrt{d} u_{i, j}$$
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Under uniform-sphere distribution: $\hat{u_{i, j}}\sim\mathcal{N}(0,1).$

4. Apply Gaussian goodness-of-fit test (Epps–Pulley):
   $$T_i=EP(\{\hat{u_{i, j}}\}, \mathcal{N}(0,1))$$

5. Regularization objective:
   $$L_{\text{SphereReg}}=\frac{1}{m}\sum_{i=1}^m T_i$$

6. Total training loss:
   $$L=L_{\text{InfoNCE}}+ L_{\text{SphereReg}}$$

This regularizer enforces approximate uniform hyperspherical embeddings, which are maximally spread and ideal for smooth semantic interpolation.

## 4. Generative Modeling Procedure

After training encoder $f_{\theta}$ and decoder $g_{\phi}$:

### Sampling Procedure

$$z=\frac{\epsilon}{\Vert\epsilon\Vert}, \epsilon\sim\mathcal{N}(0,I_d);     x=g(z)$$

### Latent Interpolation

Explore:
$$z(t)=(1-t)z_1+tz_2\ \ \ \text{(then normalize if needed)}$$
Check whether interpolations generate smooth digit transitions.

## 5. Evaluation Metrics

Representational quality

- Covariance of embedding distribution
- Sphericity / isotropy tests
- Eigenvalue spread of covariance matrix

Generative quality

- Reconstruction error
- Sample diversity
- Frechet distance (FID) on MNIST
- Smoothness of interpolation trajectories

## 6. Expected Scientific Outcomes

Do contrastive embeddings become invertible when isotropic?

Does isotropy improve generative smoothness?

Is the spherical model or Gaussian model better at generating MNIST digits?

How does SIGReg compare to standard uniformity losses in contrastive learning (e.g., InfoNCE-only)?
