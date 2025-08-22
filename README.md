# ML-to-Predict-Properties-of-Galaxies
Bachelor Thesis: Exploring Machine Learning Architecture for Predicting Stellar Mass and Star Formation Rate of Galaxies from SDSS, AllWISE, and GAMA Data

Abstract:
The exponential growth of astronomical data has driven the need for efficient methods to predict galactic properties such as star formation rate (SFR) and stellar mass (SM), as traditional approaches such as spectral energy distribution (SED) fitting sed fitting requires greater computing power for massive datasets. This study aims to evaluate and compare four machine learning architectures, including Artificial Neural Network (ANN), Wide and Deep Neural Network (WDNN), XGBoost, and CatBoost, in predicting these properties solely from photometric data. The models were trained using two datasets: optical-infrared data from SDSS and AllWISE with targets from the MPA-JHU DR8 catalog, and 21-band panchromatic data from the Galaxy And Mass Assembly (GAMA) with targets from the MAGPHYS catalog.

The results show that the gradient boosting models (CatBoost and XGBoost) offer the best balance between accuracy and efficiency. These models achieve high precision, especially on GAMA data (standard deviation error, σ_SM ≈ 0.07 dex and σ_SFR ≈ 0.16 dex), with significantly shorter training times compared to ANN and WDNN. Consistently, stellar mass was found to be easier to predict than SFR, and the use of GAMA 21-band panchromatic data significantly improved prediction accuracy for all models. This study confirms that ML methods, particularly gradient boosting, are highly accurate, efficient, and crucial tools for analyzing galaxy properties in future large-scale astronomical surveys.

Keywords: Machine Learning, Galactic Properties, Photometry.
