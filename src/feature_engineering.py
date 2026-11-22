"""Explicit questionnaire-led feature/target definitions. No political inputs."""
FEATURES={
'age':('Q1','Q1',18,120,'numeric','Demographic life-course position'),
'sex':('Q101','Q100',1,2,'categorical','Interviewer-coded respondent gender; binary instrument limitation'),
'urban':('URBRUR','URBRUR',1,3,'categorical','Residential setting'),
'education':('Q97','Q94',0,9,'ordinal','Attainment, not years of schooling or parental education'),
'employment':('Q95A','Q93A',0,3,'categorical','Paid work and unemployment status'),
'occupation':('Q95C','Q93B',0,95,'categorical','Current or last occupation; special national codes harmonised'),
'food_deprivation':('Q7A','Q6A',0,4,'ordinal','Experienced material insecurity'),
'water_deprivation':('Q7B','Q6B',0,4,'ordinal','Experienced material insecurity'),
'medical_deprivation':('Q7C','Q6C',0,4,'ordinal','Experienced material insecurity'),
'fuel_deprivation':('Q7D','Q6D',0,4,'ordinal','Experienced material insecurity'),
'cash_deprivation':('Q7E','Q6E',0,4,'ordinal','Frequency without cash income, not income amount'),
'asset_radio':('Q92A','Q90A',0,2,'categorical','Household asset ownership'),
'asset_tv':('Q92B','Q90B',0,2,'categorical','Household asset ownership'),
'asset_vehicle':('Q92C','Q90C',0,2,'categorical','Household asset ownership'),
'asset_computer':('Q92D','Q90D',0,2,'categorical','Household asset ownership'),
'asset_bank':('Q92E','Q90E',0,2,'categorical','Financial access'),
'asset_phone':('Q92F','Q90F',0,2,'categorical','Communication asset'),
'electricity_home':('Q94A','Q92A',0,1,'categorical','Household infrastructure'),
'unsafe_neighbourhood':('Q8A','Q7A',0,4,'ordinal','Experienced local insecurity'),
'fear_crime_home':('Q8B','Q7B',0,4,'ordinal','Experienced local insecurity'),
'area_electricity':('EA_SVC_A','EA_SVC_A',0,1,'categorical','Enumerator-observed local infrastructure'),
'area_water':('EA_SVC_B','EA_SVC_B',0,1,'categorical','Enumerator-observed local infrastructure'),
'area_school':('EA_FAC_B','EA_FAC_B',0,1,'categorical','Enumerator-observed education availability'),
'area_clinic':('EA_FAC_D','EA_FAC_D',0,1,'categorical','Enumerator-observed health-service availability')}
TARGETS={
'economic':{'r8':'Q47B','r9':None,'valid':[1,2,3,4,5],'positive':[4,5],'description':'Agreement that higher tax rates on rich people to benefit poor people are fair; neutral retained as non-agreement'},
'cultural':{'r8':'Q86C','r9':'Q87C','valid':[1,2,3,4,5],'positive':[3,4,5],'description':'Does not dislike having homosexual neighbours (would not care or would like); narrow tolerance item, not sexual orientation'},
'political':{'r8':'Q21','r9':'Q23','valid':[1,2,3],'positive':[3],'description':'Democracy preferable to any other government versus indifference/conditional nondemocratic preference'},
'welfare':{'r8':'Q45','r9':None,'valid':[1,2,3,4],'positive':[1,2],'description':'Higher taxes with more government services preferred over lower taxes with fewer services; neither excluded'}}
MACROS={'gdp_pc':'NY.GDP.PCAP.KD','unemployment':'SL.UEM.TOTL.ZS','gini':'SI.POV.GINI','education_spending':'SE.XPD.TOTL.GD.ZS','public_health_spending':'SH.XPD.GHED.GD.ZS'}
A=['age','sex','urban']; B=list(FEATURES)[:18]; C=list(FEATURES)+['log_gdp_pc','unemployment','gini','education_spending','public_health_spending']
