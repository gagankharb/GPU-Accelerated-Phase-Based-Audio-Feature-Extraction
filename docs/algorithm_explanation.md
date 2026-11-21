# Algorithm Explanation: Phase Residuals and Cepstral Projections

## 1. Frame Segmentation and Analysis Windowing
For an input sampled signal $x[n]$ at sample rate $f_s = 16000\text{ Hz}$, overlapping frames are extracted using:
- Frame length $L = 400$ samples ($25\text{ ms}$)
- Hop size $H = 160$ samples ($10\text{ ms}$)
- FFT size $N = 512$ (zero-padded)

For frame $t \in [0, T-1]$, raw samples begin at $s_t = t \cdot H$. A symmetric Hann window $w[m]$ is applied:
$$w[m] = 0.5 \cdot \left(1 - \cos\left(\frac{2\pi m}{L - 1}\right)\right), \quad 0 \le m < L$$
Zero-padding is appended for $L \le m < N$.

## 2. Low-Energy RMS Gating
Because unvoiced segments or silence exhibit chaotic, unstructured phase values, low-energy frames are gated:
$$\text{RMS}(t) = \sqrt{\frac{1}{L} \sum_{m=0}^{L-1} (x[s_t + m])^2}$$
If $\text{RMS}(t) < \epsilon_{\text{gate}}$ (default $10^{-4}$), phase residuals are set to zero, suppressing background noise amplification.

## 3. Batched cuFFT and Phase Extraction
A 1D Real-to-Complex (R2C) discrete Fourier transform is computed across all frames concurrently:
$$X(k, t) = \sum_{m=0}^{N-1} x_{\text{padded}}[m, t] \, e^{-j \frac{2\pi k m}{N}}, \quad 0 \le k < \frac{N}{2}$$
Phase angle is derived via 4-quadrant arctangent:
$$\phi(k, t) = \text{atan2}\left(\text{Im}(X(k,t)), \, \text{Re}(X(k,t))\right) \in [-\pi, \pi]$$

## 4. Phase Residual Formulation
For stationary sinusoids matching bin center frequency $k$, the expected phase increment between frames spaced by $H$ samples is:
$$\Delta\theta_{\text{expected}}(k) = \frac{2\pi k H}{N}$$
The raw deviation between successive frames is:
$$\Delta\phi_{\text{raw}}(k, t) = \phi(k, t) - \phi(k, t-1) - \Delta\theta_{\text{expected}}(k)$$
This raw quantity is wrapped strictly to $[-\pi, \pi]$:
$$\text{wrap}(\theta) = \theta - 2\pi \cdot \text{round}\left(\frac{\theta}{2\pi}\right)$$

## 5. Phase-Based Cepstral Decomposition
The wrapped frequency-domain phase profile $\Delta\phi(k, t)$ ($K = 256$ bins) is decomposed onto an orthogonal basis:
- **Cosine Cepstral Coefficients** ($n = 1 \dots 10$):
  $$c_n^{\cos}(t) = \frac{1}{K} \sum_{k=0}^{K-1} \Delta\phi(k, t) \cos\left(\frac{\pi n (k + 0.5)}{K}\right)$$
- **Sine Cepstral Coefficients** ($n = 1 \dots 10$):
  $$c_n^{\sin}(t) = \frac{1}{K} \sum_{k=0}^{K-1} \Delta\phi(k, t) \sin\left(\frac{\pi n (k + 0.5)}{K}\right)$$
This yields 20 static descriptors per frame.

## 6. Temporal Differential Features
To capture dynamic spectral transitions, first ($\Delta$) and second ($\Delta^2$) derivatives are computed using 5-point linear regression:
$$d(t) = \frac{\sum_{n=1}^{D} n \cdot (c(t+n) - c(t-n))}{2 \sum_{n=1}^{D} n^2}, \quad D = 2$$
$$d(t) = \frac{1 \cdot (c(t+1) - c(t-1)) + 2 \cdot (c(t+2) - c(t-2))}{10}$$
Boundary indices are clamped to $[0, T-1]$.
The final feature vector per frame comprises 60 features (20 static + 20 delta + 20 delta-delta).
