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


xtunitroot llc household_debt, lags(3) // p value .0059 so not stationary
xtunitroot llc log_GDP, lags(3) //not stationary
xtunitroot llc private_debt, lags(3) //not stationary

xtdescribe
// // tsfill

xtunitroot fisher l_GDP_dif, dfuller trend lags(3) 
xtunitroot fisher hd_dif, dfuller trend lags(3) // checking stationarity 
xtunitroot fisher pd_dif, dfuller trend lags(3) // checking stationarity 

xtvar2 l_GDP_dif hd_dif pd_dif, mc lags(3) 

estat ic

matrix list e(b) 

matrix V = e(V) 

matrix se = vecdiag(V)
matrix se_sqrt = se
forvalues i = 1/27 {
    matrix se_sqrt[1,`i'] = sqrt(se[1,`i'])
}
matrix list se_sqrt







//
// by country_id: gen l1_l_GDP_dif = L1.l_GDP_dif
// by country_id: gen l2_l_GDP_dif = L2.l_GDP_dif
// by country_id: gen l3_l_GDP_dif = L3.l_GDP_dif
//
// by country_id: gen l1_hd_dif = L1.hd_dif
// by country_id: gen l2_hd_dif = L2.hd_dif
// by country_id: gen l3_hd_dif = L3.hd_dif
//
// by country_id: gen l1_pd_dif = L1.pd_dif
// by country_id: gen l2_pd_dif = L2.pd_dif
// by country_id: gen l3_pd_dif = L3.pd_dif
//
// xtreg l_GDP_dif l1_l_GDP_dif l2_l_GDP_dif l3_l_GDP_dif ///
//             l1_hd_dif l2_hd_dif l3_hd_dif ///
//             l1_pd_dif l2_pd_dif l3_pd_dif, fe
//
// xtreg hd_dif l1_l_GDP_dif l2_l_GDP_dif l3_l_GDP_dif ///
//             l1_hd_dif l2_hd_dif l3_hd_dif ///
//             l1_pd_dif l2_pd_dif l3_pd_dif, fe
//			
// xtreg pd_dif l1_l_GDP_dif l2_l_GDP_dif l3_l_GDP_dif ///
// l1_hd_dif l2_hd_dif l3_hd_dif ///
// l1_pd_dif l2_pd_dif l3_pd_dif, fe


