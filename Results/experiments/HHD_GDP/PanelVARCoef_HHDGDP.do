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


* Sort and set up panel structure
sort country_id quarter
xtset country_id quarter

by country_id: gen lag_GDP = L.log_GDP
by country_id: gen lag_hhd = L.household_debt


gen l_GDP_dif =log_GDP[_n]-lag_GDP
gen hd_dif =household_debt[_n]-lag_hhd



drop if missing(l_GDP_dif) | missing(hd_dif) 
spbalance, balance 
// export delimited "balanced_data.csv", replace

save balanced_data_HHDGDP.dta, replace

xtunitroot llc household_debt, lags(3) // p value .0059 so not stationary
xtunitroot llc log_GDP, lags(3) //not stationary

xtdescribe
// // tsfill

xtunitroot fisher l_GDP_dif, dfuller trend lags(3)  
xtunitroot fisher hd_dif, dfuller trend lags(3) // checking stationarity 


drop if quarter >= tq(2016q1)

// use balanced_data_HHDGDP.dta, clear
pvar l_GDP_dif hd_dif, lags(1) td

// pvarstable
matrix b = e(b)

matrix list b

* Subset coefficients into B1, B2
matrix B1 = b[1,1..2]'   
matrix B2 = b[1,3..4]' 

* Combine them side by side into matrix B
matrix B = B1, B2

* List the combined matrix
matrix list B

* Define row names for better organization
matrix rownames B = "l_GDP_dif_L1" ///
					"hd_dif_L1" ///

* Define column names for each equation in the system
matrix colnames B = "GDP_eq" "HD_eq" 

* Save coefficients in an Excel file
putexcel set "coef1_GDPHHD.xlsx", replace

* Save the matrix B in Excel, starting from cell A2, including row and column names
putexcel A2 = matrix(B), names





// pvarirf, mc(30) impulse(hd_dif) level(95) table
//
// matrix V = e(V)
//
// * List the matrix to inspect its structure
// matrix list V
//
// * Save the variance-covariance matrix in an Excel file
// putexcel set "variance_covariance3.xlsx", replace
//
// * Save the matrix V in Excel, starting from cell A2, including row and column names
// putexcel A2 = matrix(V), names


// cd "/Users/bram/Desktop/Honors Draft for Reviewers/Honors-DeepVAR-Household-Debt-Analysis/Results/experiments/HHD_GDP/pvarcoefs_GDPHHD/"

pvar l_GDP_dif hd_dif, lags(2) td

// pvarstable
matrix b = e(b)

matrix list b

* Subset coefficients into B1, B2
matrix B1 = b[1,1..4]'   // First 9 coefficients
matrix B2 = b[1,5..8]' // Next 9 coefficients


* Combine them side by side into matrix B
matrix B = B1, B2

* List the combined matrix
matrix list B

* Define row names for better organization
matrix rownames B = "l_GDP_dif_L1" "l_GDP_dif_L2" ///
                     "hd_dif_L1" "hd_dif_L2" ///
           

* Define column names for each equation in the system
matrix colnames B = "GDP_eq" "HD_eq"

* Save coefficients in an Excel file
// local folder_path "/Users/bram/Desktop/Honors Draft for Reviewers/Honors-DeepVAR-Household-Debt-Analysis/Results/experiments/HHD_GDP/pvarcoefs_GDPHHD/"

putexcel set "coef2_GDPHHD.xlsx", replace

* Save the matrix B in Excel, starting from cell A2, including row and column names
putexcel A2 = matrix(B), names


// pvarirf, mc(30) impulse(hd_dif) level(95) save(irf_lag_2.dta)
//
// use irf_lag_2.dta, clear
//
// export excel using irf_lag_2.xlsx, firstrow(variables) replace


use balanced_data_HHDGDP.dta, clear


pvar l_GDP_dif hd_dif, lags(3) td

// pvarstable
matrix b = e(b)

matrix list b

* Subset coefficients into B1, B2
matrix B1 = b[1,1..6]'   
matrix B2 = b[1,7..12]' 

* Combine them side by side into matrix B
matrix B = B1, B2


* List the combined matrix
matrix list B

* Define row names for better organization
matrix rownames B = "l_GDP_dif_L1" "l_GDP_dif_L2" "l_GDP_dif_L3" ///
                     "hd_dif_L1" "hd_dif_L2" "hd_dif_L3" ///

* Define column names for each equation in the system
matrix colnames B = "GDP_eq" "HD_eq"

* Save coefficients in an Excel file
putexcel set "coef3_GDPHHD.xlsx", replace

* Save the matrix B in Excel, starting from cell A2, including row and column names
putexcel A2 = matrix(B), names


// pvarirf, mc(30) impulse(hd_dif) level(95) save(irf_lag_3.dta), 

// use irf_lag_3.dta, clear

// export excel using irf_lag_3.xlsx, firstrow(variables) replace







pvar l_GDP_dif hd_dif, lags(4) td

// pvarstable
matrix b = e(b)

matrix list b

* Subset coefficients into B1, B2
matrix B1 = b[1,1..8]'   
matrix B2 = b[1,9..16]' 

* Combine them side by side into matrix B
matrix B = B1, B2

* List the combined matrix
matrix list B

* Define row names for better organization
matrix rownames B = "l_GDP_dif_L1" "l_GDP_dif_L2" "l_GDP_dif_L3" "l_GDP_dif_L4" ///
                     "hd_dif_L1" "hd_dif_L2" "hd_dif_L3" "hd_dif_L4" ///

* Define column names for each equation in the system
matrix colnames B = "GDP_eq" "HD_eq"


* Save coefficients in an Excel file
putexcel set "coef4_GDPHHD.xlsx", replace

putexcel A2 = matrix(B), names







pvar l_GDP_dif hd_dif, lags(5) td

// pvarstable
matrix b = e(b)

matrix list b

* Subset coefficients into B1, B2
matrix B1 = b[1,1..10]'   
matrix B2 = b[1,11..20]' 

* Combine them side by side into matrix B
matrix B = B1, B2

* List the combined matrix
matrix list B

* Define row names for better organization
matrix rownames B = "l_GDP_dif_L1" "l_GDP_dif_L2" "l_GDP_dif_L3" "l_GDP_dif_L4" "l_GDP_dif_L5" ///
                    "hd_dif_L1" "hd_dif_L2" "hd_dif_L3" "hd_dif_L4" "hd_dif_L5"

* Define column names for each equation in the system
matrix colnames B = "GDP_eq" "HD_eq"

* Save coefficients in an Excel file
putexcel set "coef5_GDPHHD.xlsx", replace

putexcel A2 = matrix(B), names



pvar l_GDP_dif hd_dif, lags(6) td

// pvarstable
matrix b = e(b)

matrix list b

* Subset coefficients into B1, B2
matrix B1 = b[1,1..12]'   
matrix B2 = b[1,13..24]' 
* Combine them side by side into matrix B
matrix B = B1, B2

* List the combined matrix
matrix list B

* Define row names for better organization
matrix rownames B = "l_GDP_dif_L1" "l_GDP_dif_L2" "l_GDP_dif_L3" "l_GDP_dif_L4" "l_GDP_dif_L5" "l_GDP_dif_L6" ///
                     "hd_dif_L1" "hd_dif_L2" "hd_dif_L3" "hd_dif_L4" "hd_dif_L5" "hd_dif_L6"

* Define column names for each equation in the system
matrix colnames B = "GDP_eq" "HD_eq"

* Save coefficients in an Excel file
putexcel set "coef6_GDPHHD.xlsx", replace

putexcel A2 = matrix(B), names




