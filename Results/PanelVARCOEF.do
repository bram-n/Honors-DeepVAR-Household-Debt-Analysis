* Clear workspace and import data
clear all
set more off

* Import CSV file
import delimited "/Users/bram/Desktop/Honors Draft for Reviewers/Honors-DeepVAR-Household-Debt-Analysis/Results/Data/WithoutPolicyRate.csv", clear

* Generate log_GDP and drop unnecessary variables
gen log_GDP = ln(gdp)
drop cpi exchange_rate gdp


gen date_only = substr(time_period, 1, 10)

* Convert the date string to a Stata date
gen date = date(date_only, "YMD")

* Format as a date
format date %td
gen quarter= qofd(date)
format quarter %tq

drop date_only time_period

encode country, generate(country_id)

* Label variables
label variable log_GDP "Log GDP"
label variable household_debt "Household Debt"
label variable private_debt "Private Debt"


* Sort and set up panel structure
sort country_id quarter
xtset country_id quarter

by country_id: gen lag_GDP = L.log_GDP
by country_id: gen lag_hhd = L.household_debt
by country_id: gen lag_pd = L.private_debt


gen l_GDP_dif =log_GDP[_n]-lag_GDP
gen hd_dif =household_debt[_n]-lag_hhd
gen pd_dif =private_debt[_n]-lag_pd



drop if missing(l_GDP_dif) | missing(hd_dif) | missing(pd_dif)
spbalance, balance 
export delimited "balanced_data.csv", replace


xtunitroot llc household_debt, lags(3) // p value .0059 so not stationary
xtunitroot llc log_GDP, lags(3) //not stationary
xtunitroot llc private_debt, lags(3) //not stationary

xtdescribe
// // tsfill

xtunitroot fisher l_GDP_dif, dfuller trend lags(3) 
xtunitroot fisher hd_dif, dfuller trend lags(3) // checking stationarity 
xtunitroot fisher pd_dif, dfuller trend lags(3) // checking stationarity 



xtvar l_GDP_dif hd_dif pd_dif, lags(3)

pvar l_GDP_dif hd_dif pd_dif, lags(4) 

pvarstable
matrix b = e(b)

matrix list b

* Subset coefficients into B1, B2, and B3
matrix B1 = b[1,1..9]'   // First 9 coefficients
matrix B2 = b[1,10..18]' // Next 9 coefficients
matrix B3 = b[1,19..27]' // Last 9 coefficients

* Combine them side by side into matrix B
matrix B = B1, B2, B3

* List the combined matrix
matrix list B

* Define row names for better organization
matrix rownames B = "l_GDP_dif_L1" "hd_dif_L1" "pd_dif_L1" ///
                     "l_GDP_dif_L2" "hd_dif_L2" "pd_dif_L2" ///
                     "l_GDP_dif_L3" "hd_dif_L3" "pd_dif_L3"

* Define column names for each equation in the system
matrix colnames B = "GDP_eq" "HD_eq" "PD_eq"

* Save coefficients in an Excel file
putexcel set "coef3.xlsx", replace

* Save the matrix B in Excel, starting from cell A2, including row and column names
putexcel A2 = matrix(B), names




pvarirf, mc(30) impulse(hd_dif) level(95) table

matrix V = e(V)

* List the matrix to inspect its structure
matrix list V

* Save the variance-covariance matrix in an Excel file
putexcel set "variance_covariance3.xlsx", replace

* Save the matrix V in Excel, starting from cell A2, including row and column names
putexcel A2 = matrix(V), names


