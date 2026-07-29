# Data directory

Place the cross-source scenario dataset here before running the analysis.

## Expected file

```
data/Energy_Storage_Data_Collection_Vyara_Tsoneva.xlsx
```

The script auto-detects the following filename:

`Energy_Storage_Data_Collection_Vyara_Tsoneva.xlsx`


## Dataset description

The dataset contains **141 scenario-observations**, which are extracted from **45 published sources** between 2018 and 2026. Each observation corresponds to a specific scenario-horizon combination reported by a study, so a single publication that reports 2030, 2040, and 2050 horizons under two scenario variants contributes six rows.

Sources include:
- National and pan-European system-planning studies
- Industry outlook publications (BloombergNEF, SolarPower Europe, WindEurope, EASE, Wood Mackenzie, Ember)
- Peer-reviewed academic model runs (Golombek et al. 2022, Pickering et al. 2022, Victoria et al. 2019, Upadhyay et al. 2025, and others)
- Institutional projections (IEA, JRC, IRENA)
- Network-based scenarios (ENTSO-E, EEA & ACER)

Inclusion required at least one storage-relevant quantity or one physical driver to be reported from a study, which main focus is Europe/ EU, for a defined European/ EU, global, or EU-country specific geographic scope, if data is available.

## Columns expected

The script uses the following columns (source column names on the left, internal names on the right):

| Source column                     | Internal name          |
|-----------------------------------|------------------------|
| Publisher/Author                  | publisher_author       |
| Publishing Year                   | publishing_year        |
| Region                            | region                 |
| Horizon (Year)                    | year                   |
| Total Storage (GW)                | storage_gw             |
| Total Storage (GWh)               | storage_gwh            |
| Battery (GW)                      | battery_gw             |
| Battery (GWh)                     | battery_gwh            |
| Pumped Hydro (GW)                 | pumped_hydro_gw        |
| VRE (%)                           | vre_pct                |
| RES (%)                           | res_pct                |
| Solar (GW)                        | solar_gw               |
| Wind (GW)                         | wind_gw                |
| Nuclear (GW)                      | nuclear_gw             |
| Coal (GW)                         | coal_gw                |
| Natural Gas (GW)                  | natural_gas_gw         |
| Hydro (GW)                        | hydro_gw               |
| H2 Electrolyzers (GW)             | h2_electrolyzers_gw    |
| Electricity Demand (GWh)          | elec_demand_gwh        |
| Other Flexibility, H/M/L          | other_flexibility      |
| Standardized Scenario Category    | scenario_std           |

Columns not listed here are ignored by the analysis. Missing columns are handled and the corresponding features are dropped from the ML step.

## Row-level attribution

Every row in the dataset is annotated with its publisher and publication year. The full source list with per-row attribution is documented in **Appendix A** of the thesis.

Each row is also classified into one of four source categories used in Figure 1:

| Category       | Includes                                                             |
|----------------|----------------------------------------------------------------------|
| Institutional  | IEA, IRENA, European Commission, JRC, National Environment Agencies  |
| Industry       | EASE, SolarPower Europe, WindEurope, Ember, Statista, Wood Mackenzie, BloombergNEF, Global Renewables Alliance |
| Network-based  | ENTSO-E, ENTSOG                                                      |
| Academic       | Peer-reviewed model runs (identified by the "et al." convention)     |

The classification rules live in `src/config.py` (`SOURCE_CATEGORY_RULES`) and can be edited if a new source needs a different placement.

## Data availability

The dataset itself is not distributed via this repository. Consult the thesis appendix for the row-level source list and reconstruct the file from the primary sources (all publicly available), or contact the author for access under the same terms.

## What is *not* committed

- Raw `.xlsx` / `.csv` / `.tsv` files are excluded via `.gitignore`
- Only this `README.md` is tracked in the `data/` directory