# Diabetes Risk Factor Analysis using NHANES Dataset

## Project Overview

This project analyzes the National Health and Nutrition Examination Survey (NHANES) dataset to uncover patterns related to diabetes risk factors and biomarkers. Using advanced data mining techniques and unsupervised machine learning, this analysis identifies significant patterns and relationships that contribute to diabetes risk.

## Author
- **Hoang Son Pham** 

## Introduction

Diabetes represents a significant public health challenge, affecting millions of Americans and contributing to substantial healthcare costs and reduced quality of life. This project employs a combination of data cleaning techniques, exploratory data analysis, visualization methods, and unsupervised machine learning algorithms to extract meaningful insights from the multidimensional NHANES health dataset.

## Data Sources

The analysis uses the following NHANES datasets:

1. **Demographics (DEMO_L)**: Participant demographics including age, gender, race, education level, household information
2. **Dietary Intake (DR1TOT_L)**: Comprehensive nutritional information from 24-hour dietary recall interviews
3. **Body Measurements (BMX_L)**: Anthropometric data including height, weight, BMI, waist circumference
4. **Blood Pressure (BPXO_L)**: Systolic and diastolic blood pressure readings
5. **Glycohemoglobin (GHB_L)**: Laboratory results for HbA1c, the primary indicator for diabetes diagnosis

## Methodology

### Data Preprocessing
- Columns with more than 30% missing values were dropped
- Remaining missing values were imputed using appropriate methods:
  - Mean imputation for numerical features
  - Mode imputation for categorical features
- Feature standardization using RobustScaler to handle outliers
- Categorical feature encoding

### Exploratory Data Analysis
- Statistical summary of key variables
- Distribution analysis of health indicators
- Correlation analysis between variables
- Feature importance analysis

### Dimensionality Reduction and Clustering
- Principal Component Analysis (PCA) for dimensionality reduction
- t-SNE for non-linear dimensionality reduction
- K-means clustering (optimal k=7)
- DBSCAN density-based clustering
- Cluster validation using silhouette scores and domain knowledge

## Key Findings

1. **Risk Factor Identification**: Age, waist circumference, BMI, and blood pressure emerged as the strongest predictors of diabetes status, with odds ratios confirming their significant impact.

2. **Clustering Insights**: K-means clustering identified 7 distinct subgroups with unique health profiles that don't perfectly align with traditional clinical diabetes classifications.

3. **Demographic Patterns**: Significant differences in diabetes prevalence were observed across racial groups, highlighting potential health disparities.

4. **Dietary Correlations**: Negative correlations were found between certain dietary components (such as dietary fiber, total folate) and both BMI and glycohemoglobin levels.

5. **Feature Importance**: The radar chart analysis revealed that certain clusters were characterized by distinct patterns of fatty acid profiles, alcohol consumption, and body weight measurements.

## Visualizations

The project includes several advanced visualizations:

1. **Correlation Heatmaps**: Identifying relationships between health indicators
2. **Interactive Scatter Plots**: Exploring paired relationships between key variables
3. **Dimensionality Reduction Plots**: Visualizing high-dimensional data in 2D/3D
4. **Radar Charts**: Comparing feature importance across clusters
5. **Distribution Plots**: Examining variable distributions by cluster and diabetes status

## Ethical Considerations

The analysis acknowledges important ethical considerations:

1. **Privacy Protection**: All data is anonymized with no personally identifiable information
2. **Responsible Reporting**: Results are presented at group level with appropriate context
3. **Scientific Accuracy**: Correlations are presented without claiming causation
4. **Contextual Interpretation**: Demographic differences are presented with social determinants context

## Conclusion

This analysis demonstrates the value of data mining approaches in extracting insights from complex health datasets and contributes to our understanding of the multifactorial nature of diabetes risk. The identified patterns and relationships can inform more targeted screening approaches, personalized interventions, and improved public health strategies.

## Technologies Used

- **Python**: Primary programming language
- **Pandas & NumPy**: Data manipulation and numerical analysis
- **Scikit-learn**: Machine learning algorithms
- **Matplotlib & Seaborn**: Static visualizations
- **Bokeh**: Interactive visualizations
- **SciPy**: Statistical analysis

## Repository Structure

- **DiabetesAnalysis.ipynb**: Main Jupyter notebook containing the complete analysis
- **jupytext_script.py**: Python script version of the notebook
- **Data files**: Original NHANES dataset files (*.xpt.txt)

## Future Research Directions
Future research could build upon these findings by:
1. Incorporating longitudinal data to track progression of risk factors
2. Applying supervised learning for predictive modeling
3. Investigating identified clusters in more detail for tailored interventions

## Acknowledgements

This project uses data from the National Health and Nutrition Examination Survey (NHANES), conducted by the Centers for Disease Control and Prevention (CDC). 
