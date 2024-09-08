import torch.nn as nn
import torch

class Predictor_onlyGlobalFeats_wdropout(nn.Module):
    
    def __init__(self, global_feats=0, n_tasks=1, predictor_hidden_feats=128, predictor_dropout=0., num_layers=1):
        super(Predictor_onlyGlobalFeats_wdropout, self).__init__();
        
        if num_layers == 1:
            mlp = [nn.Dropout(predictor_dropout), nn.Linear(global_feats, n_tasks-1)];
        else:
            mlp = [nn.Dropout(predictor_dropout), nn.Linear( global_feats, predictor_hidden_feats), nn.BatchNorm1d(predictor_hidden_feats), nn.ReLU()];
            for _ in range(num_layers - 2):
                mlp.extend([nn.Dropout(predictor_dropout), nn.Linear(predictor_hidden_feats, predictor_hidden_feats), nn.BatchNorm1d(predictor_hidden_feats), nn.ReLU()]);
            mlp.extend([nn.Linear(predictor_hidden_feats, n_tasks-1)]);
        
        self.predict = nn.Sequential(*mlp);

    def forward(self, global_feats):
                        
            logits = self.predict(global_feats);
            probas = torch.sigmoid(logits);
            
            return logits, probas;