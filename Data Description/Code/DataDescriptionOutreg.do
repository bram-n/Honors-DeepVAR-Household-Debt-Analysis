import excel using "/Users/bram/Desktop/Honors Economics/Data Description/Code/above70Data.xlsx", firstrow clear

outreg2 using above70DataStatSummary.doc, dec(2) word title("Table 2: Statistical Summary when Household debt >= 70") replace sum(log) keep(privatedebt Householddebt policyrate ExchangeRate GDP Per_GDP Per_HHD) label

clear

import excel using "/Users/bram/Desktop/Honors Economics/Data Description/Code/below70Data.xlsx", firstrow clear

outreg2 using below70DataStatSummary.doc, dec(2) word title("Table 1: Statistical Summary when Household debt < 70") replace sum(log) keep(privatedebt Householddebt policyrate ExchangeRate GDP Per_GDP Per_HHD) label


