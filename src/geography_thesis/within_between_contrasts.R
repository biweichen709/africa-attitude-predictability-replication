if (dir.exists(".Rlib")) .libPaths(c(normalizePath(".Rlib"), .libPaths()))
library(glmmTMB)
rows<-list()
for(target in c('economic','cultural','political','welfare')){
 f<-readRDS(paste0('models/geography_thesis/',target,'_within_between.rds'));beta<-fixef(f)$cond;V<-vcov(f)$cond;s<-read.csv(paste0('results/geography_thesis/analysis/between_scales_',target,'.csv'))
 for(i in seq_len(nrow(s))){x<-s$term[i];scale<-s$between_sd[i];w<-paste0('w_',x);b<-paste0('b_',x);delta<-beta[b]/scale-beta[w];se<-sqrt(V[b,b]/scale^2+V[w,w]-2*V[b,w]/scale);rows[[length(rows)+1]]<-data.frame(target=target,variable=x,within_original_unit=beta[w],between_original_unit=beta[b]/scale,difference=delta,se=se,lower=delta-1.96*se,upper=delta+1.96*se,inference='Model-based unadjusted exploratory Wald contrast')}
}
r<-do.call(rbind,rows);write.csv(r,'results/geography_thesis/analysis/within_between_contrasts.csv',row.names=FALSE);print(r[r$variable %in% c('education','deprivation'),])
