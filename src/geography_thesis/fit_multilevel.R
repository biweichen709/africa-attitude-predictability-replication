if (dir.exists(".Rlib")) .libPaths(c(normalizePath(".Rlib"), .libPaths()))
args <- commandArgs(trailingOnly=TRUE)

suppressPackageStartupMessages(library(lme4)); library(jsonlite);suppressPackageStartupMessages(library(glmmTMB))
out <- 'results/geography_thesis/analysis'; dir.create('models/geography_thesis',showWarnings=FALSE,recursive=TRUE)
core <- c('age10','female','urban_i','education','deprivation','assets','electricity','employed')
predict_only <- '--predict-only' %in% args; args <- setdiff(args,'--predict-only')
targets <- if(length(args)) args else c('economic','cultural','political','welfare')
a <- read.csv(file.path(out,'r8_prepared.csv'),stringsAsFactors=FALSE,na.strings=c('','NA'))
b <- read.csv(file.path(out,'r10_prepared.csv'),stringsAsFactors=FALSE,na.strings=c('','NA'))
wts <- function(d,mode='equal') {if(mode=='unweighted') return(rep(1,nrow(d))); w<-d$weight_raw; if(mode=='raw') return(w/mean(w)); w/ave(w,d$iso3,FUN=sum)*nrow(d)/length(unique(d$iso3))}
augment <- function(train,test=NULL) {
  wt <- wts(train); mu <- sapply(core,function(x) weighted.mean(train[[x]],wt)); means <- aggregate(train[core]*train$weight_raw,list(iso3=train$iso3),sum); totals<-aggregate(train$weight_raw,list(iso3=train$iso3),sum)
  means[core] <- means[core]/totals$x
  between_sd <- sapply(means[core],sd); between_sd[between_sd<1e-8] <- 1
  cm <- c(weighted.mean(train$log_gdp_pc,wt),weighted.mean(train$v2x_polyarchy,wt)); cs <- c(sqrt(weighted.mean((train$log_gdp_pc-cm[1])^2,wt)),sqrt(weighted.mean((train$v2x_polyarchy-cm[2])^2,wt)))
  trans <- function(d) {
    mm<-means[match(d$iso3,means$iso3),core,drop=FALSE];for(j in seq_along(core)) {mm[[j]][is.na(mm[[j]])]<-mu[j];d[[paste0('w_',core[j])]]<-d[[core[j]]]-mm[[j]];d[[paste0('b_',core[j])]]<-(mm[[j]]-mu[j])/between_sd[j]}
    d$gdp_z<-(d$log_gdp_pc-cm[1])/cs[1];d$institution_z<-(d$v2x_polyarchy-cm[2])/cs[2];d
  }
  list(train=trans(train),test=if(!is.null(test)) trans(test) else NULL,means=means,grand=mu,between_sd=between_sd,context_mean=cm,context_sd=cs)
}
fitlog <- function(d,terms,weights=wts(d)) glm(as.formula(paste('y~',paste(terms,collapse='+'))),data=d,weights=weights,family=binomial())
diagnostics <- list()
varframe <- function(f) {
 if(inherits(f,'merMod'))return(as.data.frame(lme4::VarCorr(f)))
 v<-VarCorr(f)$cond; rows<-list();for(g in names(v)){vv<-v[[g]];for(i in seq_len(nrow(vv)))rows[[length(rows)+1]]<-data.frame(grp=g,var1=rownames(vv)[i],var2=NA,vcov=vv[i,i],sdcor=sqrt(vv[i,i]))};do.call(rbind,rows)
}
fitmixed <- function(d,terms,random='(1|iso3)',nodes=1,mode='equal',key='') {
 warnings<-character();d$fit_weight<-wts(d,mode)
 form<-as.formula(paste('y~',paste(terms,collapse='+'),'+',random));isnull<-length(terms)==1 && terms=='1' && nodes==9
 if(isnull){
  f<-withCallingHandlers(glmer(form,data=d,weights=fit_weight,family=binomial(),nAGQ=9,control=glmerControl(optimizer='bobyqa',optCtrl=list(maxfun=10000))),warning=function(w){warnings<<-c(warnings,conditionMessage(w));invokeRestart('muffleWarning')})
  gg<-f@optinfo$derivs$gradient; mg<-if(is.numeric(gg) && length(gg)) max(abs(gg)) else NA_real_
  diagnostics[[key]]<<-list(n=nrow(d),countries=length(unique(d$iso3)),engine='lme4 AGQ9',nodes=9,weight_mode=mode,warnings=unique(warnings),singular=isSingular(f),optimizer=f@optinfo$conv,messages=f@optinfo$conv$lme4$messages,max_gradient=mg,logLik=as.numeric(logLik(f)))
 }else{
  prof<-length(terms)==length(core) && all(terms==core)
  f<-withCallingHandlers(glmmTMB(form,data=d,weights=fit_weight,family=binomial(),control=glmmTMBControl(profile=prof,optCtrl=list(iter.max=1000,eval.max=1500))),warning=function(w){warnings<<-c(warnings,conditionMessage(w));invokeRestart('muffleWarning')})
  diagnostics[[key]]<<-list(n=nrow(d),countries=length(unique(d$iso3)),engine='glmmTMB Laplace',nodes=1,weight_mode=mode,warnings=unique(warnings),singular=any(varframe(f)$sdcor<1e-4),optimizer=f$fit$convergence,messages=f$fit$message,positive_definite_hessian=f$sdr$pdHess,max_gradient=max(abs(f$obj$gr(f$fit$par))),logLik=as.numeric(logLik(f)))
 }
 f
}
coefrows <- function(f,key,target) {cc<-as.data.frame(if(inherits(f,'merMod'))coef(summary(f)) else summary(f)$coefficients$cond);data.frame(target=target,model=key,term=rownames(cc),estimate=cc[[1]],std_error=cc[[2]],lower=cc[[1]]-1.96*cc[[2]],upper=cc[[1]]+1.96*cc[[2]],inference='model-based weighted pseudo-likelihood Wald',row.names=NULL)}
varrows <- function(f,key,target) {v<-varframe(f);v$target<-target;v$model<-key;v}
predictmixed <- function(f,test,known) {
  test$fit_weight <- 1
  lp<-as.numeric(predict(f,newdata=test,type='link',allow.new.levels=TRUE)); unseen<-!(test$iso3 %in% known);
  if(any(unseen)){sd<-varframe(f)$sdcor[1];grid<-qnorm(((1:101)-.5)/101);lp0<-lp[unseen]; pp<-rowMeans(plogis(outer(lp0,sd*grid,'+')));lp<-plogis(lp);lp[unseen]<-pp;return(lp)};plogis(lp)
}
prediction_suite <- function(train,test,target,setting,key) {
  z<-augment(train,test);tr<-z$train;te<-z$test
  pooled<-suppressWarnings(fitlog(tr,core)); centred<-suppressWarnings(fitlog(tr,paste0('w_',core)));context<-suppressWarnings(fitlog(tr,c('gdp_z','institution_z')));combined<-suppressWarnings(fitlog(tr,c(core,'gdp_z','institution_z')))
  mixed<-fitmixed(tr,core,key=paste(target,key,sep='_'))
  ww<-wts(tr);grand<-weighted.mean(tr$y,ww);rates<-tapply(seq_len(nrow(tr)),tr$iso3,function(i) (sum(ww[i]*tr$y[i])+.5)/(sum(ww[i])+1));prior<-as.numeric(rates[te$iso3]);prior[is.na(prior)]<-grand
  pred<-data.frame(row_id=te$row_id,iso3=te$iso3,weight_raw=te$weight_raw,y=te$y,pooled=predict(pooled,te,type='response'),country_only=prior,centred=predict(centred,te,type='response'),multilevel=predictmixed(mixed,te,unique(tr$iso3)),context_only=predict(context,te,type='response'),pooled_context=predict(combined,te,type='response'),global=grand,setting=setting,target=target,fold=key)
  if(setting=='loco'){tt<-te;for(x in core)tt[[paste0('w_',x)]]<-tt[[x]]-weighted.mean(tt[[x]],tt$weight_raw);pred$centred_transductive<-predict(centred,tt,type='response')}
  list(pred=pred,models=list(pooled=pooled,centred=centred,context=context,combined=combined,mixed=mixed),transform=z[c('means','grand','context_mean','context_sd')])
}
for(target in targets) {
  cat('START',target,format(Sys.time()),'\n');flush.console()
  if(predict_only)diagnostics<-fromJSON(file.path(out,paste0('fit_diagnostics_',target,'.json')),simplifyVector=FALSE)
  d<-a[complete.cases(a[,c(core,target,'log_gdp_pc','v2x_polyarchy')]),]; d$y<-d[[target]]; transform<-augment(d);dd<-transform$train;write.csv(data.frame(term=names(transform$between_sd),between_sd=transform$between_sd),file.path(out,paste0('between_scales_',target,'.csv')),row.names=FALSE);crows<-list();vrows<-list()
  wb<-c(paste0('w_',core),paste0('b_',core),'gdp_z','institution_z')
  specifications<-list(null=list(terms='1',random='(1|iso3)',nodes=9,mode='equal'),within_between=list(terms=wb,random='(1|iso3)',nodes=1,mode='equal'),random_slope=list(terms=wb,random='(1+w_education||iso3)',nodes=1,mode='equal'),moderation=list(terms=c(wb,'w_education:institution_z'),random='(1+w_education||iso3)',nodes=1,mode='equal'),wb_unweighted=list(terms=wb,random='(1|iso3)',nodes=1,mode='unweighted'),wb_raw_weight=list(terms=wb,random='(1|iso3)',nodes=1,mode='raw'),null_laplace=list(terms='1',random='(1|iso3)',nodes=1,mode='equal'))
  if(!predict_only)for(key in names(specifications)){
    s<-specifications[[key]]; f<-fitmixed(dd,s$terms,s$random,s$nodes,s$mode,paste(target,key,sep='_'));saveRDS(f,file.path('models/geography_thesis',paste0(target,'_',key,'.rds')));crows[[key]]<-coefrows(f,key,target);vrows[[key]]<-varrows(f,key,target)
    if(key=='random_slope'){rr<-ranef(f)$cond$iso3;rr$iso3<-rownames(rr);write.csv(rr,file.path(out,paste0('random_effects_',target,'.csv')),row.names=FALSE)}
    write.csv(do.call(rbind,crows),file.path(out,paste0('multilevel_coefficients_',target,'.csv')),row.names=FALSE);write.csv(do.call(rbind,vrows),file.path(out,paste0('variance_components_',target,'.csv')),row.names=FALSE);write_json(diagnostics,file.path(out,paste0('fit_diagnostics_',target,'.json')),pretty=TRUE,auto_unbox=TRUE);cat('EXPLANATORY',target,key,'\n');flush.console()
  }
  set.seed(20261122);strata<-split(seq_len(nrow(d)),paste(d$iso3,d$y));teidx<-unlist(lapply(strata,function(i)sample(i,max(1,floor(length(i)*.25)))));tr<-d[-teidx,];te<-d[teidx,]
  z<-prediction_suite(tr,te,target,'known_country','split75');write.csv(z$pred,file.path(out,paste0('pred_known_',target,'.csv')),row.names=FALSE);saveRDS(z[c('models','transform')],file.path('models/geography_thesis',paste0(target,'_known.rds')))
  splits<-data.frame(row_id=d$row_id,iso3=d$iso3,role=ifelse(seq_len(nrow(d)) %in% teidx,'test','train'));write.csv(splits,file.path(out,paste0('split_',target,'.csv')),row.names=FALSE)
  lc<-list();for(country in sort(unique(d$iso3))) {
    zz<-prediction_suite(d[d$iso3!=country,],d[d$iso3==country,],target,'loco',country);lc[[country]]<-zz$pred;write.csv(do.call(rbind,lc),file.path(out,paste0('pred_loco_',target,'.csv')),row.names=FALSE);saveRDS(zz[c('models','transform')],file.path('models/geography_thesis',paste0(target,'_loco_',country,'.rds')));cat('LOCO',target,country,'\n');flush.console()
  }
  if(target %in% c('cultural','political')){ee<-b[complete.cases(b[,c(core,target,'log_gdp_pc','v2x_polyarchy')]),];ee$y<-ee[[target]];zz<-prediction_suite(d,ee,target,'temporal','R10');zz$pred$cohort<-ifelse(zz$pred$iso3 %in% unique(a$iso3),'common','new');write.csv(zz$pred,file.path(out,paste0('pred_temporal_',target,'.csv')),row.names=FALSE);saveRDS(zz[c('models','transform')],file.path('models/geography_thesis',paste0(target,'_temporal.rds')))}
  write_json(diagnostics,file.path(out,paste0('fit_diagnostics_',target,'.json')),pretty=TRUE,auto_unbox=TRUE);cat('TARGET_COMPLETE',target,format(Sys.time()),'\n');flush.console()
}
capture.output(sessionInfo(),file=file.path(out,paste0('R_session_',paste(targets,collapse='_'),'.txt')))
