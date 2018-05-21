sub_list = read.csv('/home/json/Desktop/peer/et_qap.csv')
sub_list = sub_list$Subjects

peer_median = read.csv('/home/json/Desktop/peer/peer_group_median.csv')
et_median = read.csv('/home/json/Desktop/peer/et_group_median.csv')

x_peer_group = scale(peer_median$x, center=TRUE, scale=TRUE)
x_et_group = scale(et_median$x, center=TRUE, scale=TRUE)
y_peer_group = scale(peer_median$y, center=TRUE, scale=TRUE)
y_et_group = scale(et_median$y, center=TRUE, scale=TRUE)

x_vol_icc = c()
y_vol_icc = c()

for (vol in c(1:250)) {

  x_subjects = c()
  x_fixations = c()
  y_subjects = c()
  y_fixations = c()
  
for (sub in sub_list) {
  
  if (file.exists(paste('/data2/Projects/Jake/Human_Brain_Mapping/', sub, '/et_device_pred.csv', sep="")) & file.exists(paste('/data2/Projects/Jake/Human_Brain_Mapping/', sub, '/gsr0_train1_model_tp_predictions.csv', sep=""))) {
    
    et_df = read.csv(paste('/data2/Projects/Jake/Human_Brain_Mapping/', sub, '/et_device_pred.csv', sep=""))
    peer_df = read.csv(paste('/data2/Projects/Jake/Human_Brain_Mapping/', sub, '/gsr0_train1_model_tp_predictions.csv', sep=""))
    
    x_et = scale(et_df$x_pred, center=TRUE, scale=TRUE)
    x_peer = scale(peer_df$x_pred, center=TRUE, scale=TRUE)
    y_et = scale(et_df$y_pred, center=TRUE, scale=TRUE)
    y_peer = scale(peer_df$y_pred, center=TRUE, scale=TRUE)
    
    if (length(x_et) == 250 & length(x_peer) == 250) {
      
      x_subjects = rbind(x_subjects, sub, sub)
      y_subjects = rbind(y_subjects, sub, sub)
      
      x_fixations = rbind(x_fixations, x_et[vol], x_peer[vol])
      y_fixations = rbind(y_fixations, y_et[vol], y_peer[vol])
      
    }
    
  }
  
}
  
  x_df = data.frame(x_subjects, x_fixations)
  y_df = data.frame(y_subjects, y_fixations)
  
  x_icc = ICCest(c('x_subjects'), c('x_fixations'), x_df)
  y_icc = ICCest(c('y_subjects'), c('y_fixations'), y_df)
  
  x_vol_icc = rbind(x_vol_icc, x_icc$ICC)
  y_vol_icc = rbind(y_vol_icc, y_icc$ICC)

}

plot(c(1:250), x_vol_icc, type='l', col='red')
lines(c(1:250), y_vol_icc, col='blue')
legend(1, -.2, legend=c('x-dir', 'y-dir'), lty=1:1.5, col=c('red', 'blue'), cex=.6)

