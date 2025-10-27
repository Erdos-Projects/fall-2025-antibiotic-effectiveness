# fall-2025-antibiotic-effectiveness
Team project: fall-2025-antibiotic-effectiveness

**Team Members:** <br>
[Mustafain Ali](https://github.com/alimustafain) <br>
[Tinghao Huang](https://github.com/tinghao-huang-4939)<br>
[Dominique Hughes](https://github.com/dhughe13)<br>
[Chiara Mattamira](https://github.com/cmattamira)<br>
[Haejun (Stella) Oh](https://github.com/Haejun-Oh)<br>

## Project Overview

*We developed an end-to-end pipeline to predict the resistance of nine commonly used antibiotics prescribed for E. coli infections.*

**Stakeholders:** Medical providers, public health agencies (federal, state, and local).

Since laboratory susceptibility tests can take 24–48 hours to return results, a predictive model can assist clinicians in making more informed empirical treatment decisions at the time of patient visit.

In addition, many smaller hospitals and clinics lack the resources or infrastructure to routinely perform resistance testing. For these settings, our model could provide valuable guidance by estimating likely resistance patterns based on patient-level factors and historical data.

Finally, the model helps highlight which patient and contextual features are most associated with resistance—providing insights that can inform both clinical practice and public health strategies.

## Data Access
This project uses the Antimicrobial Resistance and Microbiome Dynamics (ARMD) dataset ([Dryad Repository](https://datadryad.org/dataset/doi:10.5061/dryad.jq2bvq8kp)) developed by Stanford University researchers.

The ARMD dataset contains 751,075 microbiological culture records from 283,715 unique patients collected between 1999 and 2024. It integrates detailed microbiological, demographic, and clinical information — including culture type (urine, respiratory, blood), identified organisms, antibiotic susceptibility (55 antibiotics across five categories), and patient-level variables such as age, sex, ADI score, ward location, prior antibiotic exposure, and recent nursing home visits.

For this project, we subset the dataset to:
- Focus exclusively on E. coli isolates (the most common organism in the dataset)
- Include records from 2016 onward to ensure consistency in antibiotic testing standards and to ensure that our predictions reflect contemporary resistance patterns relevant to the current clinical practice
- Restrict analyses to nine antibiotics tested in all of E. coli samples

## Modeling approach and main results: 
We split the dataset into 80% training and 20% testing subsets. Of the training data, 20% was reserved as a validation set. 
We compared baseline and advanced models: 
- Logistic regression
- Support Vector Machine (SVM)
- Random forests,
- XGBoost

Model performance was evaluated using accuracy and false negative rate (FNR).
The model with the lowest FNR on the test set was selected as the final model.
Feature importance analysis was then conducted to identify the most influential predictors.


## Dependencies
 XGBoost, Scikit-Learn, Pandas, Numpy, Matplotlib, Seaborn





## Folder organization and summary


## Contact
To further discuss this project or use associate code feel free to contact us: <br>
Mustafain Ali - ma9614@g.rit.edu <br>
Tinghao Huang - tinghaohuang76@gmail.com  <br>
Dominique Hughes - dhughe13@asu.edu <br>
Chiara Mattamira - chiara.mattamira@gmail.com <br>
Haejun (Stella) Oh - ohhu@mail.uc.edu <br>
