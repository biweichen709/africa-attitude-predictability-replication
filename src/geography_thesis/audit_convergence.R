if (dir.exists(".Rlib")) .libPaths(c(normalizePath(".Rlib"), .libPaths()))
library(glmmTMB);library(jsonlite)
rows<-list()
for(target in c('economic','cultural','political','welfare'))for(model in c('within_between','random_slope','moderation','wb_unweighted','wb_raw_weight','null_laplace')){
 path<-paste0('models/geography_thesis/',target,'_',model,'.rds');if(!file.exists(path))next;f<-readRDS(path);gr<-f$sdr$gradient.fixed;se<-sqrt(diag(f$sdr$cov.fixed));scaled<-max(abs(gr)*se);rows[[length(rows)+1]]<-data.frame(target=target,model=model,convergence=f$fit$convergence,pdHess=f$sdr$pdHess,max_raw_gradient=max(abs(gr)),max_se_scaled_gradient=scaled)
 if(model=='moderation'){print(target);print(data.frame(term=names(fixef(f)$cond),beta=fixef(f)$cond,gradient=gr[seq_along(fixef(f)$cond)]))}
}
r<-do.call(rbind,rows);write.csv(r,'results/geography_thesis/analysis/convergence_scaled_audit.csv',row.names=FALSE);print(r)
