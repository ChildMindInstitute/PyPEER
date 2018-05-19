sub_list = read.csv('/home/json/Desktop/peer/et_qap.csv')
sub_list = sub_list$Subjects

peer_median = read.csv('/home/json/Desktop/peer/peer_group_median.csv')
et_median = read.csv('/home/json/Desktop/peer/et_group_median.csv')

x_peer_group = peer_median$x
x_et_group = et_median$x
y_peer_group = peer_median$y
y_et_group = et_median$y

subjects = c()
correlations = c()

for (sub in sub_list) {
  
  if (file.exists(paste('/data2/Projects/Jake/Human_Brain_Mapping/', sub, '/et_device_pred.csv', sep="")) & file.exists(paste('/data2/Projects/Jake/Human_Brain_Mapping/', sub, '/gsr0_train1_model_tp_predictions.csv', sep=""))) {
    
    et_df = read.csv(paste('/data2/Projects/Jake/Human_Brain_Mapping/', sub, '/et_device_pred.csv', sep=""))
    peer_df = read.csv(paste('/data2/Projects/Jake/Human_Brain_Mapping/', sub, '/gsr0_train1_model_tp_predictions.csv', sep=""))
    
    x_et = et_df$x_pred
    x_peer = peer_df$x_pred
    y_et = et_df$y_pred
    y_peer = peer_df$y_pred
    
    if (length(x_et) == 250 & length(x_peer) == 250) {
      
      subjects = rbind(subjects, sub, sub)
      correlations = rbind(correlations, cor(x_et, x_et_group, method='pearson'), cor(y_et, y_et_group, method='pearson'))
      
    }
    
  }
  
}

df = data.frame(subjects, correlations)

icc_val = ICCbare(c('subjects'), c('correlations'), df)
