require(devtools)
options(unzip='internal') # Allows you to download using install_github on Ubuntu
install_github('neurodata/fmriutils')
install_github('ebridge2/Discriminability')
library('discriminability') # loads the package
library('fmriutils')

sub_list = read.csv('/home/json/Desktop/peer/et_qap.csv')
sub_list = sub_list$Subjects

et_terciles = read.csv('/home/json/Desktop/peer/et_tercile_diversity_score.csv')
peer_terciles = read.csv('/home/json/Desktop/peer/peer_tercile_diversity_score.csv')

et_q1 = et_terciles[et_terciles$tercile == 'q1',]$volume + 1
et_q2 = et_terciles[et_terciles$tercile == 'q2',]$volume + 1
et_q3 = et_terciles[et_terciles$tercile == 'q3',]$volume + 1
peer_q1 = peer_terciles[peer_terciles$tercile == 'q1',]$volume + 1
peer_q2 = peer_terciles[peer_terciles$tercile == 'q2',]$volume + 1
peer_q3 = peer_terciles[peer_terciles$tercile == 'q3',]$volume + 1

disc_analysis <- function(sub_list, analysis_group=NULL) {

  if(is.null(analysis_group)) {
    num_volumes = 250 
    analysis_group = c(1:250)
  } else {
    num_volumes = length(analysis_group)
  }
  
  x_matrix = matrix(, nrow=1, ncol=num_volumes)
  y_matrix = matrix(, nrow=1, ncol=num_volumes)
  et_sub_ids = c()
  peer_sub_ids = c()

  for (sub in sub_list) {
    
    if (file.exists(paste('/data2/Projects/Jake/Human_Brain_Mapping/', sub, '/et_device_pred.csv', sep="")) & file.exists(paste('/data2/Projects/Jake/Human_Brain_Mapping/', sub, '/gsr0_train1_model_tp_predictions.csv', sep=""))) {
    
        et_df = read.csv(paste('/data2/Projects/Jake/Human_Brain_Mapping/', sub, '/et_device_pred.csv', sep=""))
        peer_df = read.csv(paste('/data2/Projects/Jake/Human_Brain_Mapping/', sub, '/gsr0_train1_model_tp_predictions.csv', sep=""))
        
        x_et = scale(et_df$x_pred[c(analysis_group)], center=TRUE, scale=TRUE)
        x_peer = scale(peer_df$x_pred[c(analysis_group)], center=TRUE, scale=TRUE)
        y_et = scale(et_df$y_pred[c(analysis_group)], center=TRUE, scale=TRUE)
        y_peer = scale(peer_df$y_pred[c(analysis_group)], center=TRUE, scale=TRUE)
        
        if (length(x_et) == num_volumes & length(x_peer) == num_volumes) {
          
          x_et = matrix(x_et, nrow=1, ncol=num_volumes)
          x_peer = matrix(x_peer, nrow=1, ncol=num_volumes)
          y_et = matrix(y_et, nrow=1, ncol=num_volumes)
          y_peer = matrix(y_peer, nrow=1, ncol=num_volumes)
        
          x_matrix = rbind(x_matrix, x_et, x_peer)
          y_matrix = rbind(y_matrix, y_et, y_peer)
          
          et_sub_ids = rbind(et_sub_ids, sub)
          peer_sub_ids = rbind(peer_sub_ids, sub)
          
          }
          
      }
      
  }
  
  sub_ids = c()
  
#   et_sub_ids = sample(et_sub_ids) # Randomize order of subjects
#   peer_sub_ids = sample(peer_sub_ids)
  
  for (item in c(1:length(et_sub_ids))) {
    
    sub_ids = rbind(sub_ids, et_sub_ids[item], peer_sub_ids[item])
    
  }
  
  x_matrix = x_matrix[-1,]
  y_matrix = y_matrix[-1,]
  
  n_scans = dim(x_matrix)[1]
  
  x_dist_graphs <- dist(x_matrix, diag=TRUE, upper=TRUE)
  y_dist_graphs <- dist(y_matrix, diag=TRUE, upper=TRUE)
  
  x_distance <- array(matrix(as.matrix(x_dist_graphs)), dim=c(n_scans, n_scans))
  y_distance <- array(matrix(as.matrix(y_dist_graphs)), dim=c(n_scans, n_scans))
  
  x_rdf = discr.rdf(x_distance, sub_ids)
  y_rdf = discr.rdf(y_distance, sub_ids)
  
  x_disc_score = discr.discr(x_rdf, remove_outliers=TRUE, thresh=0, output=TRUE)
  y_disc_score = discr.discr(y_rdf, remove_outliers=TRUE, thresh=0, output=TRUE)
  
}

disc_analysis(sub_list)





