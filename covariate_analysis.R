library(vioplot)

peer_pheno =read.csv('/home/json/Desktop/peer/Peer_pheno.csv')
rownames(peer_pheno) <- peer_pheno$SubID

sub_list = read.csv('/home/json/Desktop/peer/et_qap.csv')
sub_list = as.character(sub_list$Subjects)

peer_median = read.csv('/home/json/Desktop/peer/peer_group_median.csv')
et_median = read.csv('/home/json/Desktop/peer/et_group_median.csv')

x_peer_group = peer_median$x
x_et_group = et_median$x
y_peer_group = peer_median$y
y_et_group = et_median$y

x_corr_et = c()
y_corr_et = c()
x_corr_peer = c()
y_corr_peer = c()
age = c()
iq = c()
swan_in = c()
swan_hyp = c()
cbcl_tot = c()
cbcl_int = c()
cbcl_ext = c()

for (sub in sub_list) {
  
  if (file.exists(paste('/data2/Projects/Jake/Human_Brain_Mapping/', sub, '/et_device_pred.csv', sep="")) & file.exists(paste('/data2/Projects/Jake/Human_Brain_Mapping/', sub, '/gsr0_train1_model_tp_predictions.csv', sep=""))) {
    
    et_df = read.csv(paste('/data2/Projects/Jake/Human_Brain_Mapping/', sub, '/et_device_pred.csv', sep=""))
    peer_df = read.csv(paste('/data2/Projects/Jake/Human_Brain_Mapping/', sub, '/gsr0_train1_model_tp_predictions.csv', sep=""))
    
    x_et = et_df$x_pred
    x_peer = peer_df$x_pred
    y_et = et_df$y_pred
    y_peer = peer_df$y_pred
    
    if (length(x_et) == 250 & length(x_peer) == 250) {
      
      sub = substr(sub, 5, 11)
      
      x_corr_et = rbind(x_corr_et, cor(x_et, x_et_group, method='pearson'))
      y_corr_et = rbind(y_corr_et, cor(y_et, y_et_group, method='pearson'))
      x_corr_peer = rbind(x_corr_peer, cor(x_peer, x_peer_group, method='pearson'))
      y_corr_peer = rbind(y_corr_peer, cor(y_peer, y_peer_group, method='pearson'))
      age = rbind(age, peer_pheno[sub, "Age"])
      iq = rbind(iq, peer_pheno[sub, "FSIQ"])
      swan_in = rbind(swan_in, peer_pheno[sub, "SWAN_IN"])
      swan_hyp = rbind(swan_hyp, peer_pheno[sub, "SWAN_HY"])
      cbcl_tot = rbind(cbcl_tot, peer_pheno[sub, "CBCL_Total"])
      cbcl_int = rbind(cbcl_int, peer_pheno[sub, "CBCL_Int"])
      cbcl_ext = rbind(cbcl_ext, peer_pheno[sub, "CBCL_Ext"])
      
    }
    
  }
  
}

cov_df = data.frame(x_corr_et, x_corr_peer, y_corr_et, y_corr_peer,
                    age, iq, swan_in, swan_hyp, cbcl_tot, cbcl_int, cbcl_ext)

for (cov in colnames(cov_df)[5:11]) {

  temp_df = cov_df[, c("x_corr_et", "x_corr_peer", 'y_corr_et', 'y_corr_peer', cov)]
  temp_df = temp_df[complete.cases(temp_df), ]
  
  for (comp in c("x_corr_et", "x_corr_peer", 'y_corr_et', 'y_corr_peer')) {
    
    linearMod <- lm(temp_df[, comp] ~ temp_df[, cov])
  
    p_val = summary(linearMod)$coefficients[2, 4]
    
    print(c(cov, comp, p_val))
    
    plot(temp_df[[cov]], temp_df$x_corr_et, col='red', ylim=c(0:1))
    abline(linearMod, col='blue')
    legend("topleft", legend=c(p_val), lty=1:1.5, col=c('red'), cex=.6)
    
  }
  
}








