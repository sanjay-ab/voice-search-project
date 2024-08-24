import torch
import json
import argparse
import sys
import os
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from utils.models import SSEmodel
from utils.metrics import compute_map
from pytorch_metric_learning import losses
from utils.data import *
import time
import tqdm
import datetime

class Loss(nn.Module):

    def __init__(self, temperature):
        super(Loss, self).__init__()
        self.mse = torch.nn.MSELoss(reduction='none')
        self.ntxent = losses.NTXentLoss(temperature=temperature)
        self.T=temperature   
        self.alpha=0.1
        print('temperature',self.T,'alpha',self.alpha)

    def compute_nce(self, x, pos, negs):
        neg_is_pos = (pos == negs).all(-1)
        pos = pos.unsqueeze(0)
        targets = torch.cat([pos, negs], dim=0)
        logits = torch.cosine_similarity(
            x.float(), targets.float(), dim=-1).type_as(x)
        logits /= self.T

        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float("-inf")
        logits = logits.transpose(0, 1)  # (num_x, num_cls+1)
        return logits

    def get_mu_cov(self,x,negs): #x: 500,512 , negs: 500,512
        mu=torch.mean(negs,0)  # (512)
        cov=torch.cov(torch.transpose(negs,0,1)) #512,512
        x_mu=torch.matmul(x,mu)/self.T # 500
        cov_x=torch.matmul(cov,torch.transpose(x,0,1))/(self.T*self.T) # 500,512
        x_cov_x=torch.sum(torch.mul(torch.transpose(cov_x,0,1),x),1) # 500
        log_N=torch.log(torch.tensor(negs.size(0)).cuda())
        targets=(x_mu+x_cov_x/2+log_N)
        return targets
    
    def forward(self,embs, negatives, negatives2, word_labels):
        ntx='ntx'
        if ntx=='ntx': 
            loss = self.ntxent(embs, word_labels)
            #loss/=embs.size(0)
        elif ntx=='nce':
            norm=torch.sqrt(torch.sum(torch.pow(embs,2),dim=1)).unsqueeze(1)
            embeddings=embs/norm # (500,512)
            batch_size,emb_dim=embeddings.size()
            batch_size=int(batch_size/2)
            x_labels=word_labels[:batch_size]
            y_labels=word_labels[batch_size:]
            assert (x_labels==y_labels).all(),(x_labels,y_labels)
            x=embeddings[:batch_size,:] # (250,512)
            y=embeddings[batch_size:,] # (250,512)
            
            # across loss
            negs=y.unsqueeze(1).repeat(1,y.size(0),1) 
           
            logits=self.compute_nce(x,y,negs) # 250 251
            targets = logits.new_zeros(logits.size(0), dtype=torch.long)
            loss=logits[:,0]
            targets=torch.log(torch.sum(torch.exp(logits[:,1:]),1))
            loss_across=-torch.mean(loss-targets,0)/2

            negs=x.unsqueeze(1).repeat(1,x.size(0),1) 
            
            logits=self.compute_nce(y,x,negs) # 250 251
            targets = logits.new_zeros(logits.size(0), dtype=torch.long)
            loss=logits[:,0]
            targets=torch.log(torch.sum(torch.exp(logits[:,1:]),1))
            loss_across-=torch.mean(loss-targets,0)/2
           
            if self.alpha>0 and negatives is not None: 
                # within loss
                negs=negatives.reshape(-1,y.size(0),x.size(1))
                logits=self.compute_nce(x,y,negs) # 250 251
                targets = logits.new_zeros(logits.size(0), dtype=torch.long)
                loss=logits[:,0]
                targets=torch.log(torch.sum(torch.exp(logits[:,1:]),1))
                loss_within=-torch.mean(loss-targets,0)
                
                #negs=negatives2.reshape(-1,y.size(0),x.size(1))
                #logits=self.compute_nce(y,x,negs) # 250 251
                #targets = logits.new_zeros(logits.size(0), dtype=torch.long)
                #loss=logits[:,0]
                #targets=torch.log(torch.sum(torch.exp(logits[:,1:]),1))
                #loss_within-=torch.mean(loss-targets,0)/2
            
                # total loss
                loss=loss_across+self.alpha*loss_within
            else:
                loss=loss_across

        else:
            norm=torch.sqrt(torch.sum(torch.pow(embs,2),dim=1)).unsqueeze(1)
            embeddings=embs/norm # (500,512)
            batch_size,emb_dim=embeddings.size()
            batch_size=int(batch_size/2)
            x_labels=word_labels[:batch_size]
            y_labels=word_labels[batch_size:]
            assert (x_labels==y_labels).all(),(x_labels,y_labels)
            x=embeddings[:batch_size,:] # (250,512)
            y=embeddings[batch_size:,] # (250,512)
            cossim=torch.sum(torch.mul(x,y),1)/self.T # 250
            targets1=self.get_mu_cov(x,y) 
            targets2=self.get_mu_cov(y,x)
 
            loss1=-torch.mean(cossim-targets1,0) # 250
            loss2=-torch.mean(cossim-targets2,0) # 250
            loss=(loss1+loss2)/2
            #print(cossim.size(),torch.sum(torch.abs(cossim))/torch.sum(torch.abs(targets)),torch.sum(cossim).data,torch.sum(targets).data,loss.data)
        return loss

def train_one_epoch(encoder,
                    optimizer,
                    train_loader,
                    train_loss,
                    epoch_size=200,
                    device=None,
                    training=True):
    if training:
        encoder.train()
    else:
        encoder.eval()
    loss_train=0
    count=0
    pbar = tqdm.tqdm(total = epoch_size+1)
    negative_embs,negative_embs2=None,None
    for batch_idx, (frames,word_labels,negs,negs2) in enumerate(train_loader): 
        frames=frames.to(device,dtype= torch.float32)
        frames=torch.cuda.FloatTensor(frames)
        embs=encoder(frames)
        if negs is not None:
            negs=negs.to(device,dtype= torch.float32)
            negs=torch.cuda.FloatTensor(negs)
            negative_embs=encoder(negs) 
            #negs2=negs2.to(device,dtype= torch.float32)
            #negs2=torch.cuda.FloatTensor(negs2)
            #negative_embs2=encoder(negs2)
        loss = train_loss(embs,negative_embs,negative_embs2,word_labels)
        if training:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        loss_train+=loss.data.cpu().numpy()
        count+=1
        pbar.update(1)
        if count>epoch_size: # handle big dataset
            break
    pbar.close()
    loss_train/=count
    return loss_train

def test_on_test_data(encoder,
                     test_loader,
                     device=None):
    encoder.eval()
    embeddings = []
    labels = []
    count=0
    with torch.no_grad():
        for batch_idx, (frames,word_labels) in enumerate(test_loader):
            frames=frames.to(device,dtype= torch.float32)
            frames=torch.cuda.FloatTensor(frames)
            embs = encoder(frames)
            embeddings.append(embs)
            labels.append(word_labels)
            count+=1
    embeddings = torch.cat(embeddings)
    embeddings = np.float32(embeddings.cpu().detach())
    print('computing MAP on',embeddings.shape)
    labels=np.hstack(labels)
    labels= np.float32(labels)
    map_value=compute_map(embeddings, labels)
    print('map value:',np.around(map_value,3))
    return map_value

class TrainGridsearch():
    """
    A tune.Trainable object to train the model on a specified config.
    """
    def getListOfFiles(self,dirName):
        # create a list of file and sub directories
        # names in the given directory
        listOfFile = os.listdir(dirName)
        allFiles = list()
        # Iterate over all the entries
        for entry in listOfFile:
            # Create full path
            fullPath = os.path.join(dirName, entry)
            # If entry is a directory then get the list of files in this directory
            if os.path.isdir(fullPath):
                allFiles = allFiles + self.getListOfFiles(fullPath)
            else:
                allFiles.append(fullPath)
    
        return allFiles

    def setup(self, config):
        """
        Initialize the training procedure given the config to test, i.e, initialize:
            - the train and val dataloaders
            - the model to train
            - the optimizer
            - the loss to use
        """
        self.device=config['device']
        self.mode=config['mode']
        self.output_dir=config['output_dir']
        self.valid_on_map=config['valid_on_map']
        self.ssl_dim=config['ssl_dim']
        self.max_patience=config['max_patience']
        self.config=config
        self.patience=0
        # Loading Datasets
        print('mode=',self.mode)
        if self.mode=='vad_aug':
            path_features=None
            assert config['path_wavs'] is not None
            dataset_train=UnsupDataset(path_item=config["path_train_item"],gpu_id=self.device,max_nb_frames=config["max_nb_frames"],ssl_path=config['ssl_path'],ssl_layer=config['ssl_layer'],path_wavs=config['path_wavs'])
            collate_fn_train=get_collate_fn('train_unsup')
            train_batch_size=1
            if not self.valid_on_map: 
                dataset_test=UnsupDataset(path_item=config["path_test_item"],gpu_id=self.device,max_nb_frames=config["max_nb_frames"],ssl_path=config['ssl_path'],ssl_layer=config['ssl_layer'],path_wavs=config['path_wavs'])
                collate_fn_test=get_collate_fn('train_unsup')
                test_batch_size=1 
        else:
            path_features=self.getListOfFiles(config["path_features"])
            dataset_train=GoldDataset(path_features=path_features,path_items=config["path_train_item"],mode=config['mode'],max_nb_frames=config["max_nb_frames"])
            collate_fn_train=get_collate_fn('train')
            train_batch_size=256
            if not self.valid_on_map:
                dataset_test=GoldDataset(path_features=config['path_test_features'],path_items=config["path_test_item"],mode=config['mode'],max_nb_frames=config["max_nb_frames"],preloaded_frames=dataset_train.dict_features)
                collate_fn_test=get_collate_fn('train')
                test_batch_size=250
        
        if not self.valid_on_map:
            self.test_loader = DataLoader(
                dataset_test,
                collate_fn=collate_fn_test,
                batch_size=test_batch_size,
                shuffle=True,
                drop_last=False,
            )  
        else:
            path_map_features=self.getListOfFiles(config["path_map_features"])
            if path_map_features==path_features:
                preloaded_frames=dataset_train.dict_features
            else:
                preloaded_frames=None
            dataset_map=TestDataset(path_features=path_map_features,path_item=config["path_map_item"],max_nb_frames=config["max_nb_frames"],preloaded_frames=preloaded_frames)
            collate_fn_map=get_collate_fn('test')
            map_batch_size=400 
            
            self.test_loader = DataLoader(
                dataset_map,
                collate_fn=collate_fn_map,
                batch_size=map_batch_size,
                shuffle=True,
                drop_last=False,
            )  
        
        self.train_loader = DataLoader(
            dataset_train,
            collate_fn=collate_fn_train,
            batch_size=train_batch_size,
            shuffle=True,
            drop_last=True,
        )
        
        
        # get the input size
        path_checkpoint=config["path_checkpoint"]
        if not os.path.isdir(path_checkpoint):
            print('TRAINING NEW MODEL') 
            self.encoder = SSEmodel(
                input_size=self.ssl_dim,
                n_conv_layers=config["n_conv_layers"],
                transformer_dim=config["transformer_dim"],
                n_heads=config["n_heads"],
                n_transformer_layers=config["n_transformer_layers"],
                device=config['device'])

            # initialize the optimizer
            self.encoder.to(self.device)
            self.optimizer = optim.Adam(self.encoder.parameters(), lr=config["lr"],weight_decay=config["weight_decay"])
            #lambda1 = lambda epoch: 0.9 ** epoch
            self.scheduler = None
            #self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda1)
            
        else:
            path_model=os.path.join(path_checkpoint,'model_checkpoint.tar')
            path_config=os.path.join(path_checkpoint,'params.json') 
            assert os.path.isfile(path_model),path_model
            assert os.path.isfile(path_config),path_config
            saved_states=torch.load(path_model,map_location=torch.device(self.device))
            print('RELOADING TRAINED MODEL AT',path_checkpoint)
            print('RELOADED MODEL HAS MAPR=',saved_states['map_at_r_eval'])
            with open(path_config) as buf:
                reloaded_config=json.load(buf)
            self.encoder = SSEmodel(
                input_size=self.ssl_dim,
                n_conv_layers=reloaded_config["n_conv_layers"],
                transformer_dim=reloaded_config["transformer_dim"],
                n_heads=reloaded_config["n_heads"],
                n_transformer_layers=reloaded_config["n_transformer_layers"],
                device=self.device,
            )
            #reloading state dict 
            self.encoder.load_state_dict(saved_states['model_state_dict'])
            self.encoder.to(self.device)
            self.optimizer = optim.Adam(self.encoder.parameters(), lr=reloaded_config["lr"],weight_decay=reloaded_config["weight_decay"])
            self.optimizer.load_state_dict(saved_states['optimizer_state_dict'])
       
        # distinguish between the case where we train for only 1 run or if the gridsearch is launched
        self.train_loss = Loss(
            temperature=config["temperature"]
        )
        self.count = 0
        self.max_n_iterations = config["max_t"]
        self.best_eval = -10000
        self.epoch_size = config["epoch_size"]
        self.path_test_item = config["path_test_item"]
        self.path_map_item = config["path_map_item"]
        self.log=[]
        self.config['device']='x'
        
    def step(self):

        loss_train= train_one_epoch(
            encoder=self.encoder,
            optimizer=self.optimizer,
            train_loader=self.train_loader,
            train_loss=self.train_loss,
            epoch_size=self.epoch_size,
            device=self.device
        )
        
        self.count += 1
        nb_updates=self.epoch_size*self.count
        loss_train=np.around(loss_train.item(),5)

        if not self.valid_on_map:
            metric= train_one_epoch(
                encoder=self.encoder,
                optimizer=None,
                train_loader=self.test_loader,
                train_loss=self.train_loss,
                epoch_size=self.epoch_size,
                device=self.device,
                training=False
            )
            metric=-metric
            metric_name='test loss'
        else:
            metric = test_on_test_data(
                encoder=self.encoder,
                test_loader=self.test_loader,
                device=self.device
            )
            self.log.append('MAP:'+str(metric))
            metric_name='MAP'

        metric=np.around(metric,5)
        self.log.append(' '.join([str(s) for s in ['nb updates',nb_updates,'train loss:',loss_train,metric_name,':',np.abs(metric)]]))
        print(self.log[-1])
        
        # saves the model if there is an increase in the MAP@R
        if metric >= self.best_eval:
            self.patience=0
            if self.best_eval==-10000:
                e = datetime.datetime.now()
                self.output_dir=os.path.join(self.output_dir,'-'.join(str(e).split('.')[0].split(' ')))
                if not os.path.isdir(self.output_dir):
                    os.makedirs(self.output_dir)
            self.best_eval = metric
            self.best_nb_updates=nb_updates
            self.best_loss_train=loss_train
            model_path = os.path.join(self.output_dir,"model_checkpoint.tar")
            self.log.append(' '.join([str(s) for s in ['---BEST','nb updates',self.best_nb_updates,'train loss:',self.best_loss_train,metric_name,':',np.abs(metric)]]))
            print(self.log[-1])
            torch.save(
                {
                    "model_state_dict": self.encoder.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss_train": loss_train,
                    "map": metric
                },
                model_path,
            )
            with open(os.path.join(self.output_dir,'params.json'),'w') as buf: 
                json.dump(self.config, buf)
        else:
            self.patience+=1
            print('patience',self.patience)
        with open(os.path.join(self.output_dir,'log'),'w') as buf:
             buf.write('\n'.join(self.log)+'\n') 
        return {"map": metric}

def parse_args(argv):
    
    parser = argparse.ArgumentParser(
        description="Train one model or launch a hyperparameter search"
    )
    # Add the paths
    group_paths = parser.add_argument_group("Paths")
    group_paths.add_argument(
        "--mode",
        type=str,
        default='vad_aug',
        help='can be vad_aug (split vads and use timestrechting, pairs (use list of pairs) or gold (use list of items with transcription)'
       )
    group_paths.add_argument(
        "--valid_on_map",
        action='store_true',
        help='early stopping based on the MAP value'
       )
    group_paths.add_argument(
        "--output_dir",
        type=str,
        help="Path to save checkpoint",
        required=True,
    )
    group_paths.add_argument(
        "--path_wavs",
        type=str,
        help="Path to main wav folder",
    )
    group_paths.add_argument(
        "--path_features",
        type=str,
        help="Path to the folder containing the preextracted train features, required only for gold training and knn-iterations",
    )
    group_paths.add_argument(
        "--path_test_features",
        type=str,
        help="Path to the folder containing the test features",
        default=None
    )
    group_paths.add_argument(
        "--path_map_features",
        type=str,
        help="Path to the folder containing the test features",
        default=None
    )
    group_paths.add_argument(
        "--path_train_item",
        type=str,
        help="Path to the training item file.",
        required=True,
    )
    group_paths.add_argument(
        "--path_test_item",
        type=str,
        help="Path to the test item file.",
        default=None,
    )
    group_paths.add_argument(
        "--path_map_item",
        type=str,
        help="Path to the map item file.",
        default=None,
    )
    group_paths.add_argument(
        "--ssl_dim",
        type=int,
        help='dimension of the ssl model',
        default=768,
        #default=1024,
    )
    group_paths.add_argument(
        "--ssl_layer",
        type=int,
        help='layer to extract from ssl model',
        default=8,
    )
    group_paths.add_argument(
        "--path_ssl_model",
        type=str,
        help="Pretrained ssl model",
        #default='pretrained/xlsr2_300m.pt',
        default='pretrained/wav2vec_small.pt',
    )
    group_paths.add_argument(
        "--path_checkpoint",
        type=str,
        help="path to model to restore",
        default='',
    )
    # Add the parameters of the hyperparameter search
    run_params = parser.add_argument_group("Runs params")
    run_params.add_argument(
        "--max_t",
        type=int,
        help="Maximum number of epochs of one run.",
        default=15,
    )
    run_type = parser.add_argument_group("Runs type")
    run_type.add_argument(
        "--max_patience",
        type=int,
        help="The number of waiting steps before stopping training.",
        default=5,
    )
    run_type.add_argument(
        "--epoch_size",
        type=int,
        help="The number of optimization step that constitutes an epoch.",
        default=1000,
    )
    run_type.add_argument(
        "--max_nb_frames",
        type=int,
        help="The maximum number of frames an input segment can be constituted of.",
        default=100,
    )
    # Add the hyperparameters (only usefull to specify in the case where only one run is performed, i.e, num_samples=1)
    hyperparam = parser.add_argument_group("Hyperparameters")
    hyperparam.add_argument(
        "--n_conv_layers",
        type=int,
        help="The number of convolutional layers in the Transformer.",
        default=1,
    )
    hyperparam.add_argument(
        "--transformer_dim",
        type=int,
        help="The hidden dimension of the transformer layers.",
        default=512,
    )
    hyperparam.add_argument(
        "--n_heads",
        type=int,
        help="The number of head for each transformer layer.",
        default=4,
    )
    hyperparam.add_argument(
        "--n_transformer_layers",
        type=int,
        help="The number of transformer layers to use.",
        default=1,
    )
    hyperparam.add_argument(
        "--temperature",
        type=float,
        help="The temperature value to use in the NTXent loss",
        default=0.15,
    )
    hyperparam.add_argument(
        "--weight_decay",
        type=float,
        help="weight decay param",
        default=0,
    )
    hyperparam.add_argument(
        "--learning_rate",
        type=float,
        help="The learning rate to use in the optimizer.",
        default=0.0001,
    )

    args = parser.parse_args(argv)

    return args

def main(argv):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args(argv)
    assert args.mode in ['vad_aug','pairs','gold'],('mode:',args.mode)
    assert os.path.isfile(args.path_train_item),(args.path_train_item)
    print('USE MAP FOR VALIDATION:',args.valid_on_map)
    parameters = {
        "device":device,
        "mode": args.mode,
        "valid_on_map": args.valid_on_map,
        "path_checkpoint": args.path_checkpoint,
        "path_train_item": args.path_train_item,
        "path_features": args.path_features,
        "path_test_features": args.path_test_features,
        "path_test_item": args.path_test_item,
        "path_map_item": args.path_map_item,
        "path_map_features": args.path_map_features,
        "max_t": args.max_t,
        "epoch_size": args.epoch_size,
        "max_nb_frames": args.max_nb_frames,
        "weight_decay": args.weight_decay,
        "n_conv_layers": args.n_conv_layers,
        "n_transformer_layers": args.n_transformer_layers,
        "transformer_dim":args.transformer_dim,
        "lr":args.learning_rate,
        "temperature":args.temperature,
        "ssl_path":args.path_ssl_model,
        "ssl_layer":args.ssl_layer,
        "ssl_dim":args.ssl_dim,
        "path_wavs":args.path_wavs,
        "n_heads":args.n_heads,
        "output_dir":args.output_dir,
        "max_patience":args.max_patience,
    }
     
    grid=TrainGridsearch() 
    grid.setup(config=parameters)
    for _ in range(args.max_t):
        grid.step()
        if grid.patience>=grid.max_patience:
            print('I can t wait anymore! early stopping')
            break 
    
if __name__ == "__main__":

    argv = sys.argv[1:]
    main(argv)
