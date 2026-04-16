# Primera-Hackaton-de-modelamiento-Matematico-FCFM
Fingerprint Obfuscation algorithm that changes the fingerprint so its not recognizable as the original persons fingerprint while leaving negligible evidence detectable by fingerprint algorithms. 
<div align="center">
  <img src="https://github.com/Rajevic-0/Primera-Hackaton-de-modelamiento-Matematico-FCFM/blob/main/Example.png">
</div>


## 🥈 2nd Place 
Members: 
- Tomas Rajevic (Computer science and Engineering)
- Jou-Jin Ho Ku (Electrical Engineering)
- Vicente Villarroel (Mathematical Engineering)

## Features

The obfuscation process is composed of three main stages:

### 1. Elastic Deformation
Applies a smooth, non-linear spatial transformation across the entire fingerprint image.

- Breaks global minutiae consistency  
- Distorts ridge flow patterns  
- Mimics natural skin deformation while preventing reliable matching  

### 2. Local Swirls
Introduces localized rotational distortions in selected regions of the fingerprint.

- Perturbs critical structures such as deltas and cores  
- Disrupts orientation-based feature extraction  
- Maintains visual plausibility with controlled intensity  

### 3. Ridge Noise Injection
Adds fine-grained noise along ridge patterns.

- Alters ridge texture and continuity  
- Reduces clarity of ridge endings and bifurcations  
- Simulates acquisition noise while degrading feature precision  

## Pipeline

The transformations are applied sequentially:

1. Input fingerprint image  
2. Apply elastic deformation  
3. Apply local swirl perturbations  
4. Inject ridge noise  
5. Output obfuscated fingerprint  
