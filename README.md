# Large Scale Battery Dispatch Schedule
## Folder Structure

- `src/`: Contains all the code for the project.
- `data/`: Contains datasets, data models, and other data-related files.
- `docs/`: Contains organizational and project management files, including meeting notes and schedules.

Each folder is organized to separate the core components of the project, making it easy to navigate.

## Project Details
### Problem

Volatile generation capacities such as PV and wind pose major challenges for electricity grids. Batteries can contribute both from a grid-supporting perspective and to improving the market situation (negative electricity prices). 

### Solution

The aim is to use market data as well as weather and demand data to build a predictive model that contains a dispatch strategy for storing and withdrawing electricity.

### Method/How?

- Get Data from EPEX and other market sources as well as weather data and demand curves from Energycharts
- Clean and filter data to remove outliers, inconsistencies, and noise that may affect analysis.
- Machine Learning regressions to build an predictive model
- Model the dispatch with financial data
- Real time implementation

**Data Sources**: Epex, Energycharts

Important links:

[Microsoft Teams](https://teams.live.com/l/invite/FEA0DChUuH2yZFE2AQ)  
[Project roadmap in Notion](https://techlabs.notion.site/Project-Roadmap-WT24-14e1127e595e80b8a059efd09318c29f)
