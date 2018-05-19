sub_list = read.csv('/home/json/Desktop/peer/et_qap.csv')
sub_list = sub_list$Subjects

peer_median = read.csv('/home/json/Desktop/peer/peer_group_median.csv')
et_median = read.csv('/home/json/Desktop/peer/et_group_median.csv')

x_peer_group = peer_median$x
x_et_group = et_median$x
y_peer_group = peer_median$y
y_et_group = et_median$y

x_subjects = c()
x_correlations = c()
y_subjects = c()
y_correlations = c()

for (sub in sub_list) {
  
  if (file.exists(paste('/data2/Projects/Jake/Human_Brain_Mapping/', sub, '/et_device_pred.csv', sep="")) & file.exists(paste('/data2/Projects/Jake/Human_Brain_Mapping/', sub, '/gsr0_train1_model_tp_predictions.csv', sep=""))) {
    
    et_df = read.csv(paste('/data2/Projects/Jake/Human_Brain_Mapping/', sub, '/et_device_pred.csv', sep=""))
    peer_df = read.csv(paste('/data2/Projects/Jake/Human_Brain_Mapping/', sub, '/gsr0_train1_model_tp_predictions.csv', sep=""))
    
    x_et = et_df$x_pred
    x_peer = peer_df$x_pred
    y_et = et_df$y_pred
    y_peer = peer_df$y_pred
    
    if (length(x_et) == 250 & length(x_peer) == 250) {
      
      x_subjects = rbind(subjects, sub, sub)
      y_subjects = rbind(subjects, sub, sub)
      x_correlations = rbind(correlations, cor(x_et, x_et_group, method='pearson'), cor(x_peer, x_peer_group, method='pearson'))
      y_correlations = rbind(correlations, cor(y_et, y_et_group, method='pearson'), cor(y_peer, y_peer_group, method='pearson'))
    }
    
  }
  
}

x_df = data.frame(x_subjects, x_correlations)
y_df = data.frame(y_subjects, y_correlations)

x_icc = ICCbare(c('x_subjects'), c('x_correlations'), x_df)
y_icc = ICCbare(c('y_subjects'), c('y_correlations'), y_df)



