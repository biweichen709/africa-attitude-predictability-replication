if (dir.exists(".Rlib")) .libPaths(c(normalizePath(".Rlib"), .libPaths()))

library(glmmTMB);library(lme4);library(jsonlite)
root<-'models/geography_thesis';out<-'results/geography_thesis/analysis';rows<-list()
for(target in c('economic','cultural','political','welfare')){
 files<-list.files(root,pattern=paste0('^',target,'_loco_.*[.]rds$'),full.names=TRUE);stopifnot(length(files)==34)
 for(file in files){
  bundle<-readRDS(file);country<-sub('[.]rds$','',sub(paste0(target,'_loco_'),'',basename(file)));f<-bundle$models$mixed;mf<-model.frame(f);train<-unique(as.character(mf$iso3));stopifnot(!(country %in% train),length(train)==33,!(country %in% bundle$transform$means$iso3));ww<-model.weights(mf);sums<-tapply(ww,mf$iso3,sum);stopifnot(max(sums)-min(sums)<1e-6)
  rows[[length(rows)+1]]<-data.frame(target=target,held_country=country,n_train=nrow(mf),countries_train=length(train),target_absent=TRUE,training_means_only=TRUE,equal_country_mass=TRUE,optimizer=f$fit$convergence,positive_definite_hessian=f$sdr$pdHess)
  write.csv(do.call(rbind,rows),file.path(out,'saved_model_leakage_audit.csv'),row.names=FALSE)
 }
}
write.csv(do.call(rbind,rows),file.path(out,'saved_model_leakage_audit.csv'),row.names=FALSE)
pkgs<-installed.packages();write.csv(pkgs[,c('Package','Version','Built')],file.path(out,'R_package_versions.csv'),row.names=FALSE)
capture.output(sessionInfo(),file=file.path(out,'R_final_session.txt'));cat('SAVED_MODEL_AUDIT_PASS',length(rows),'LOCO models\n')
