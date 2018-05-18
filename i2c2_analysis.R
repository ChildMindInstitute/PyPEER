options(unzip='internal')
devtools::install_github("muschellij2/I2C2")
library('I2C2')

sub_list = read.csv('/home/json/Desktop/peer/et_qap.csv')
sub_list = sub_list$Subjects

joint_matrix = matrix(, nrow=1, ncol=500)
ids = c()
modality = c()

for (sub in sub_list) {
  
  if (file.exists(paste('/data2/Projects/Jake/Human_Brain_Mapping/', sub, '/et_device_pred.csv', sep="")) & file.exists(paste('/data2/Projects/Jake/Human_Brain_Mapping/', sub, '/gsr0_train1_model_tp_predictions.csv', sep=""))) {
    
    et_df = read.csv(paste('/data2/Projects/Jake/Human_Brain_Mapping/', sub, '/et_device_pred.csv', sep=""))
    peer_df = read.csv(paste('/data2/Projects/Jake/Human_Brain_Mapping/', sub, '/gsr0_train1_model_tp_predictions.csv', sep=""))
    
    x_et = scale(et_df$x_pred, center=TRUE, scale=TRUE)
    x_peer = scale(peer_df$x_pred, center=TRUE, scale=TRUE)
    y_et = scale(et_df$y_pred, center=TRUE, scale=TRUE)
    y_peer = scale(peer_df$y_pred, center=TRUE, scale=TRUE)
    
    if (length(x_et) == 250 & length(x_peer) == 250) {
      
      x_et = matrix(x_et, nrow=1, ncol=250)
      x_peer = matrix(x_peer, nrow=1, ncol=250)
      y_et = matrix(y_et, nrow=1, ncol=250)
      y_peer = matrix(y_peer, nrow=1, ncol=250)
      
      modality = rbind(modality, "ET", "PEER")
      ids = rbind(ids, sub, sub)
      
      joint_matrix = rbind(joint_matrix, cbind(x_et, x_peer))
      joint_matrix = rbind(joint_matrix, cbind(y_et, y_peer))
      
    }
    
  }
  
}

joint_matrix = joint_matrix[-1, ]

output = i2c2(joint_matrix, id=ids, visit=modality)
