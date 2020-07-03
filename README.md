# WCCI - Clinical BCI Challenge

Submission for Clinical BCI Challenge

# Team RIGOLETTO

Team leader is Marie-Constance Corsi (Aramis project-team, Inria Paris, Paris Brain Institute, Paris, France), assisted by Florian Yger (LAMSADE, Univ. Paris-Dauphine, Paris, France) and Sylvain Chevallier (LISV, Univ. Paris-Saclay, Velizy, France).

# Code description

Matlab files are first run to extract features from raw dataset using the [Brainstorm toolbox](https://neuroimage.usc.edu/brainstorm/), saving results in .mat files.
The .mat files are opened in Python to make the prediction.

- Download the dataset from [GitHub](https://github.com/5anirban9/Clinical-Brain-Computer-Interfaces-Challenge-WCCI-2020-Glasgow)
- Run "ComputeFCEstimators.m" matlab file to extract the features (in /Matlab/Matlab_code)
- Run "python rigoletto-predict.py"

The extracted features are available in /Matlab/Matlab_db in .mat files (cf the associated subfolders) or via the Brainstorm project ("RIGOLETTO_bst_project").
The necessary packages for running the python script are indicated in the `requirements.txt` file and could be installed with `pip install -r requirements.txt`.



# Licence

The code is released under [GNU GPL v3](https://www.gnu.org/licenses/gpl-3.0.en.html)


