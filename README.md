# Balancing the Grid: Residual Load Prediction

In this project, the required electricity generation from non-renewable sources was predicted to complement renewable energy, as renewable sources alone cannot fully meet demand and are volatile based on different variables like weather. To ensure grid stability, the necessary residual load was forecasted. Three distinct modeling approaches were employed and compared to address the challenge: time series analysis, machine learning, and deep learning. The primary objective was to estimate how much energy needs to be backed up by non-renewables while transitioning gradually toward a more renewable-based energy system. Relevant energy and weather data were collected, processed, and cleaned to train the models.

## Sources
Weather Dataset: [DWD](https://www.dwd.de/EN/ourservices/cdc/cdc_ueberblick-klimadaten_en.html) [Universität Freiburg](https://weather.uni-freiburg.de/start_en.html)
Electricity Generation and Consumption Dataset: [SMARD](https://www.smard.de/en/marktdaten?marketDataAttributes=%7B%22resolution%22:%22hour%22,%22from%22:1726775047106,%22to%22:1727984647105,%22moduleIds%22:%5B1004066,1001226,1001225,1004067,1004068,1001228,1001224,1001223,1004069,1004071,1004070,1001227,5000410%5D,%22selectedCategory%22:8,%22activeChart%22:true,%22style%22:%22color%22,%22categoriesModuleOrder%22:%7B%7D,%22region%22:%22DE%22%7D)
