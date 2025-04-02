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
// export delimited "balanced_data.csv", replace
drop if quarter >= tq(2016q1)

xtunitroot llc household_debt, lags(3) // p value .0059 so not stationary
xtunitroot llc log_GDP, lags(3) //not stationary
xtunitroot llc private_debt, lags(3) //not stationary

xtdescribe
// // tsfill

xtunitroot fisher l_GDP_dif, dfuller trend lags(3) 
xtunitroot fisher hd_dif, dfuller trend lags(3) // checking stationarity 
xtunitroot fisher pd_dif, dfuller trend lags(3) // checking stationarity 



 
pvar l_GDP_dif hd_dif pd_dif, lags(3) td

// pvarstable
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
matrix rownames B = "l_GDP_dif_L1" "l_GDP_dif_L2" "l_GDP_dif_L3" ///
                     "hd_dif_L1" "hd_dif_L2" "hd_dif_L3" ///
                     "pd_dif_L1" "pd_dif_L2" "pd_dif_L3"

* Define column names for each equation in the system
matrix colnames B = "GDP_eq" "HD_eq" "PD_eq"

* Save coefficients in an Excel file
putexcel set "coef3_PD.xlsx", replace

* Save the matrix B in Excel, starting from cell A2, including row and column names
putexcel A2 = matrix(B), names






pvar l_GDP_dif hd_dif pd_dif, lags(2) 

// pvarstable
matrix b = e(b)

matrix list b

* Subset coefficients into B1, B2, and B3
matrix B1 = b[1,1..6]'   
matrix B2 = b[1,7..12]' 
matrix B3 = b[1,13..18]' 

* Combine them side by side into matrix B
matrix B = B1, B2, B3

* List the combined matrix
matrix list B

* Define row names for better organization
matrix rownames B = "l_GDP_dif_L1" "l_GDP_dif_L2" ///
					     "hd_dif_L1" "hd_dif_L2" ///
                       "pd_dif_L1" "pd_dif_L2" 

* Define column names for each equation in the system
matrix colnames B = "GDP_eq" "HD_eq" "PD_eq"

* Save coefficients in an Excel file
putexcel set "coef2_PD.xlsx", replace

* Save the matrix B in Excel, starting from cell A2, including row and column names
putexcel A2 = matrix(B), names





pvar l_GDP_dif hd_dif pd_dif, lags(1) td

// pvarstable
matrix b = e(b)

matrix list b

* Subset coefficients into B1, B2, and B3
matrix B1 = b[1,1..3]'   
matrix B2 = b[1,4..6]' 
matrix B3 = b[1,7..9]' 

* Combine them side by side into matrix B
matrix B = B1, B2, B3

* List the combined matrix
matrix list B

* Define row names for better organization
matrix rownames B = "l_GDP_dif_L1" ///
					"hd_dif_L1" ///
					"pd_dif_L1" 

* Define column names for each equation in the system
matrix colnames B = "GDP_eq" "HD_eq" "PD_eq"

* Save coefficients in an Excel file
putexcel set "coef1_PD.xlsx", replace

* Save the matrix B in Excel, starting from cell A2, including row and column names
putexcel A2 = matrix(B), names





// pvarirf, mc(30) impulse(hd_dif) level(95) table







pvar l_GDP_dif hd_dif pd_dif, lags(4) td

// pvarstable
matrix b = e(b)

matrix list b

* Subset coefficients into B1, B2, and B3
matrix B1 = b[1,1..12]'   
matrix B2 = b[1,13..24]' 
matrix B3 = b[1,25..36]' 

* Combine them side by side into matrix B
matrix B = B1, B2, B3

* List the combined matrix
matrix list B

* Define row names for better organization
matrix rownames B = "l_GDP_dif_L1" "l_GDP_dif_L2" "l_GDP_dif_L3" "l_GDP_dif_L4" ///
                     "hd_dif_L1" "hd_dif_L2" "hd_dif_L3" "hd_dif_L4" ///
                     "pd_dif_L1" "pd_dif_L2" "pd_dif_L3" "pd_dif_L4"

* Define column names for each equation in the system
matrix colnames B = "GDP_eq" "HD_eq" "PD_eq"

* Save coefficients in an Excel file
putexcel set "coef4_PD.xlsx", replace

* Save the matrix B in Excel, starting from cell A2, including row and column names
putexcel A2 = matrix(B), names








pvar l_GDP_dif hd_dif pd_dif, lags(5) td

// pvarstable
matrix b = e(b)

matrix list b

* Subset coefficients into B1, B2, and B3
matrix B1 = b[1,1..15]'   
matrix B2 = b[1,16..30]' 
matrix B3 = b[1,31..45]' 

* Combine them side by side into matrix B
matrix B = B1, B2, B3

* List the combined matrix
matrix list B

* Define row names for better organization
matrix rownames B = "l_GDP_dif_L1" "l_GDP_dif_L2" "l_GDP_dif_L3" "l_GDP_dif_L4" "l_GDP_dif_L5" ///
                     "hd_dif_L1" "hd_dif_L2" "hd_dif_L3" "hd_dif_L4" "hd_dif_L5" ///
                     "pd_dif_L1" "pd_dif_L2" "pd_dif_L3" "pd_dif_L4" "pd_dif_L5"

* Define column names for each equation in the system
matrix colnames B = "GDP_eq" "HD_eq" "PD_eq"

* Save coefficients in an Excel file
putexcel set "coef5_PD.xlsx", replace

* Save the matrix B in Excel, starting from cell A2, including row and column names
putexcel A2 = matrix(B), names








pvar l_GDP_dif hd_dif pd_dif, lags(6) td

pvarstable
matrix b = e(b)

matrix list b

* Subset coefficients into B1, B2, and B3
matrix B1 = b[1,1..18]'   
matrix B2 = b[1,19..36]' 
matrix B3 = b[1,37..54]' 

* Combine them side by side into matrix B
matrix B = B1, B2, B3

* List the combined matrix
matrix list B

* Define row names for better organization
matrix rownames B = "l_GDP_dif_L1" "l_GDP_dif_L2" "l_GDP_dif_L3" "l_GDP_dif_L4" "l_GDP_dif_L5" "l_GDP_dif_L6" ///
                     "hd_dif_L1" "hd_dif_L2" "hd_dif_L3" "hd_dif_L4" "hd_dif_L5" "hd_dif_L6" ///
                     "pd_dif_L1" "pd_dif_L2" "pd_dif_L3" "pd_dif_L4" "pd_dif_L5" "pd_dif_L6"

* Define column names for each equation in the system
matrix colnames B = "GDP_eq" "HD_eq" "PD_eq"

* Save coefficients in an Excel file
putexcel set "coef6_PD.xlsx", replace

* Save the matrix B in Excel, starting from cell A2, including row and column names
putexcel A2 = matrix(B), names










