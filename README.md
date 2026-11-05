# ActiveLearning_Alloy

> **This repository contains the code for the paper:**
> M. Park, M. Kim, Y. Kim, S. J. Kwak, W.-B. Jung, H.-T. Jung, W. B. Lee, Y. Kim,
> "[From prediction to synthesis: DFT-active learning-guided design of multimetallic catalysts for hydrogen evolution](https://www.sciencedirect.com/science/article/abs/pii/S1385894725055299)",
> *Chemical Engineering Journal* **518**, 164693 (2025). DOI: [10.1016/j.cej.2025.164693](https://doi.org/10.1016/j.cej.2025.164693)
>
> If you use this code, please cite the paper (see [Citation](#citation)).

Active learning based model to sufficiently navigate HER alloy catalyst, consisting of five metal components. 

This repository contains the implementation of an active learning model to navigate HER catalyst. The codes are organized to generate features, preprocess data, and perform Gaussian process regression to predict energies and select the next points for calculation.

## Usage
Clone the repository: git clone https://github.com/minhee2043/ActiveLearning_Alloy.git
cd your-repo

Install the required dependencies: 
The following dependencies are required to run the scripts in this repository. 
```
numpy>=1.21.2
scipy>=1.7.1
pandas>=1.3.3
matplotlib>=3.4.3
scikit-learn>=0.24.2
```

Run the scripts in the following order

Generate features for the surface motif:
```
python get_GPRdataspace.py
```
Generate DFT-calculated dataspace:
```
python possiblefp.py
```
Convert surface motif to feature vector (code inserted when DFT calculation is performed):
```
from motif_analyzer import Slab
trajectory = ase.io.read('<name of trajectory')
motif = Slab(trajectory)
feature = np.array(mmotif.features(['Pt','Ru','Cu','Ni','Fe'],zones=['ens','sf','ssf','sn','ssn']))
```
Run active learning model
```
python mygaussian.py
```
Calculate activity based on Boltzmann distribution for GPR predicted csv files (need adjustment according to filename and size):
```
python calc_act.py
```


## Data Format
-Input Data:
  -Surface motifs and their corresponding features.
  -DFT-calculated dataspace.

-Output Data:
  -GPR predicted energies by the feature vectors for surface motifs.
  -Activity calculations of the multimetallic alloy according to the metal ratio.

## Explanation of the Codes
**get_GPRdataspace.py**
Generates all the features for the surface motif. This represents the total dataspace that requires prediction of energies.

**possiblefp.py**
Due to the periodic boundary condition of the plane-wave basis set, a DFT-calculated dataspace is generated. The candidates for DFT-calculation are selected within this dataspace.

**helperMethods.py**
Contains simple mathematical functions that help the featurization process.

**motif_analyzer.py**
Converts the surface motif of the adsorption site to a feature vector. Designation of site type needed.

**mygaussian.py**
Our active learning model. Contains Gaussian process regression model and acquisition function for selecting the next calculated points.

**calc_act.py**
Based on the Boltzmann distribution of reaction kinetics, the activity of multimetallic alloy catalyst is calculated according to each metal ratio. 

## Modifications for Other Users
**Dataspace Generation**: Modify get_GPRdataspace.py to include additional features or change the existing feature generation logic.

**DFT-calculated Dataspace**: Update possiblefp.py to change the alloy conditions

**Active Learning Model**: Customize mygaussian.py to change the Gaussian process regression model or the acquisition function. Filename should be edtited.

Feel free to modify the code to suit your specific requirements and improve the model's performance.

## Citation
If you find this code useful in your research, please cite:
```bibtex
@article{park2025prediction,
  title   = {From prediction to synthesis: DFT-active learning-guided design of multimetallic catalysts for hydrogen evolution},
  author  = {Park, Minhee and Kim, Minki and Kim, Yesol and Kwak, Seung Jae and Jung, Woo-Bin and Jung, Hee-Tae and Lee, Won Bo and Kim, YongJoo},
  journal = {Chemical Engineering Journal},
  volume  = {518},
  pages   = {164693},
  year    = {2025},
  doi     = {10.1016/j.cej.2025.164693}
}
```

