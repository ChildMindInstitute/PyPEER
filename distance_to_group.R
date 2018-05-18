library(vioplot)

sub_list = read.csv('/home/json/Desktop/peer/et_qap.csv')
sub_list = sub_list$Subjects

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

for (sub in sub_list) {
  
  if (file.exists(paste('/data2/Projects/Jake/Human_Brain_Mapping/', sub, '/et_device_pred.csv', sep="")) & file.exists(paste('/data2/Projects/Jake/Human_Brain_Mapping/', sub, '/gsr0_train1_model_tp_predictions.csv', sep=""))) {
  
  et_df = read.csv(paste('/data2/Projects/Jake/Human_Brain_Mapping/', sub, '/et_device_pred.csv', sep=""))
  peer_df = read.csv(paste('/data2/Projects/Jake/Human_Brain_Mapping/', sub, '/gsr0_train1_model_tp_predictions.csv', sep=""))
  
  x_et = et_df$x_pred
  x_peer = peer_df$x_pred
  y_et = et_df$y_pred
  y_peer = peer_df$y_pred
  
  if (length(x_et) == 250 & length(x_peer) == 250) {
  
  x_corr_et = rbind(x_corr_et, cor(x_et, x_et_group, method='pearson'))
  y_corr_et = rbind(y_corr_et, cor(y_et, y_et_group, method='pearson'))
  x_corr_peer = rbind(x_corr_peer, cor(x_peer, x_peer_group, method='pearson'))
  y_corr_peer = rbind(y_corr_peer, cor(y_peer, y_peer_group, method='pearson'))
  
  }
  
  }
  
}

vioplot(x_corr_et, x_corr_peer, ylim=c(-1, 1), names=c("Eye-Tracking", "PEER"))
title('Correlation of subject to group median in x')

vioplot(y_corr_et, y_corr_peer, ylim=c(-1, 1), names=c("Eye-Tracking", "PEER"))
title('Correlation of subject to group median in y')

vioplot(sqrt(2*(1-x_corr_et)), sqrt(2*(1-x_corr_peer)), ylim=c(0, 2), names=c("Eye-Tracking", "PEER"))
title('Distance of subject to group median in x')

vioplot(sqrt(2*(1-y_corr_et)), sqrt(2*(1-y_corr_peer)), ylim=c(0, 2), names=c("Eye-Tracking", "PEER"))
title('Distance of subject to group median in y')












