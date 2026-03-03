import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from operator import truediv
import time
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import os
from sklearn.utils.class_weight import compute_class_weight
from get_cls_map import get_cls_map, classification_map, list_to_colormap, get_classification_map
import wandb
from GCPE import PEFTHyperspectralGCViT  # Import the PEFT-modified model
import os
os.environ["WANDB_API_KEY"] = "b68375e4a0cbcbc284700f0627c966e5d78181b6"
def loadData():
    #data = sio.loadmat('data\WHU_Hi_HongHu.mat')['WHU_Hi_HongHu']
    #labels = sio.loadmat('data\WHU_Hi_HongHu_gt.mat')['WHU_Hi_HongHu_gt']
    #data = sio.loadmat('data\Salinas_corrected.mat')['salinas_corrected']
    #labels = sio.loadmat('data\Salinas_gt.mat')['salinas_gt']
    data = sio.loadmat('data/WHU_Hi_LongKou.mat')['WHU_Hi_LongKou']
    labels = sio.loadmat('data/WHU_Hi_LongKou_gt.mat')['WHU_Hi_LongKou_gt']
    return data, labels

def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX

def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX

def createImageCubes(X, y, windowSize=5, removeZeroLabels=True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1  # Convert to 0-based indexing
    
    # Validate label range
    min_label = np.min(patchesLabels)
    max_label = np.max(patchesLabels)
    assert min_label >= 0 and max_label <= 21, \
        f"Labels must be in range [0, 15], found range [{min_label}, {max_label}]"
    
    # Check for non-finite values
    assert np.isfinite(patchesData).all(), "Patch data contains non-finite values"
    
    return patchesData, patchesLabels

def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                       test_size=testRatio,
                                                       random_state=randomState,
                                                       stratify=y)
    
    # Validate split results
    assert len(np.unique(y_train)) == len(np.unique(y)), "Training set missing some classes"
    assert len(np.unique(y_test)) == len(np.unique(y)), "Test set missing some classes"
    
    return X_train, X_test, y_train, y_test

class TrainDS(torch.utils.data.Dataset):
    def __init__(self, Xtrain, ytrain):
        self.len = Xtrain.shape[0]
        
        # Validate input shapes
        assert Xtrain.ndim == 4, f"Expected 4D input array, got shape {Xtrain.shape}"
        assert ytrain.ndim == 1, f"Expected 1D label array, got shape {ytrain.shape}"
        
        # Validate label range
        min_label = np.min(ytrain)
        max_label = np.max(ytrain)
        assert min_label >= 0 and max_label <= 21, \
            f"Labels must be in range [0, 15], found range [{min_label}, {max_label}]"
        
        # Validate data values
        assert np.isfinite(Xtrain).all(), "Training data contains non-finite values"
        
        self.x_data = torch.FloatTensor(Xtrain)
        self.y_data = torch.LongTensor(ytrain)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

class TestDS(torch.utils.data.Dataset):
    def __init__(self, Xtest, ytest):
        self.len = Xtest.shape[0]
        
        # Validate input shapes
        assert Xtest.ndim == 4, f"Expected 4D input array, got shape {Xtest.shape}"
        assert ytest.ndim == 1, f"Expected 1D label array, got shape {ytest.shape}"
        
        # Validate label range
        min_label = np.min(ytest)
        max_label = np.max(ytest)
        assert min_label >= 0 and max_label <= 21, \
            f"Labels must be in range [0, 15], found range [{min_label}, {max_label}]"
        
        # Validate data values
        assert np.isfinite(Xtest).all(), "Test data contains non-finite values"
        
        self.x_data = torch.FloatTensor(Xtest)
        self.y_data = torch.LongTensor(ytest)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len
'''
class PEFTTrainer:
    def __init__(self, model, train_loader, test_loader, config=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        
        # Default configuration
        self.config = {
            'num_epochs': 100,
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
            'warmup_epochs': 5,
            'grad_clip': 1.0,
            'label_smoothing': 0.1,
            'eval_interval': 1,
            'use_amp': True,
            'scheduler': {
                'T_0': 10,
                'T_mult': 2,
                'eta_min': 1e-6
            }
        }
        if config:
            self.config.update(config)
        # Add metrics tracking on GPU
        self.train_losses = torch.zeros(self.config['num_epochs'], device=self.device)
        self.train_accuracies = torch.zeros(self.config['num_epochs'], device=self.device)
        self.setup_training()
        
    def setup_training(self):
        # Split parameters into LoRA and non-LoRA groups
        lora_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if 'lora_' in name:
                lora_params.append(param)
            else:
                other_params.append(param)
        
        # Optimizer setup
        self.optimizer = optim.AdamW([
            {'params': lora_params, 'lr': self.config['learning_rate']},
            {'params': other_params, 'lr': self.config['learning_rate'] * 0.1}
        ], weight_decay=self.config['weight_decay'])
        
        # Scheduler setup
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config['scheduler']['T_0'],
            T_mult=self.config['scheduler']['T_mult'],
            eta_min=self.config['scheduler']['eta_min']
        )
        
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.config['label_smoothing'])
        self.scaler = GradScaler() if self.config['use_amp'] else None
        
        self.best_acc = 0.0
        self.best_epoch = 0
        
        os.makedirs('peft_checkpoints', exist_ok=True)
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["num_epochs"]}')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            inputs = (inputs - inputs.mean(dim=(2,3), keepdim=True)) / (inputs.std(dim=(2,3), keepdim=True) + 1e-8)
            
            self.optimizer.zero_grad()
            
            if self.config['use_amp']:
                with autocast(device_type='cuda'):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                self.scaler.scale(loss).backward()
                
                if self.config['grad_clip'] > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                
                if self.config['grad_clip'] > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
                
                self.optimizer.step()
            
            self.scheduler.step(epoch + batch_idx / len(self.train_loader))
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            total_loss += loss.item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%',
                'lr': f'{self.scheduler.get_last_lr()[0]:.6f}'
            })
        
        return total_loss / len(self.train_loader), 100. * correct / total

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss = 0
        predictions = []
        targets_all = []
        
        for inputs, targets in tqdm(self.test_loader, desc='Evaluating'):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            inputs = (inputs - inputs.mean(dim=(2,3), keepdim=True)) / (inputs.std(dim=(2,3), keepdim=True) + 1e-8)
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            predictions.extend(predicted.cpu().numpy())
            targets_all.extend(targets.cpu().numpy())
        
        predictions = np.array(predictions)
        targets_all = np.array(targets_all)
        acc = accuracy_score(targets_all, predictions) * 100
        kappa = cohen_kappa_score(targets_all, predictions) * 100
        
        return total_loss / len(self.test_loader), acc, kappa, predictions, targets_all

    def train(self):
        wandb.init(project="peft-gcvit-hyperspectral", config=self.config)
        # Initialize lists to store metrics
        train_losses = []
        train_accuracies = []
        eval_accuracies = []
        for epoch in range(self.config['num_epochs']):
            train_loss, train_acc = self.train_epoch(epoch)
            
            if (epoch + 1) % self.config['eval_interval'] == 0:
                eval_loss, eval_acc, kappa, predictions, targets = self.evaluate()
                
                wandb.log({
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'eval_loss': eval_loss,
                    'eval_acc': eval_acc,
                    'kappa': kappa,
                    'learning_rate': self.scheduler.get_last_lr()[0]
                })
                
                is_best = eval_acc > self.best_acc
                if is_best:
                    self.best_acc = eval_acc
                    self.best_epoch = epoch
                    
                    # Save best model
                    torch.save({
                        'epoch': epoch,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'scheduler': self.scheduler.state_dict(),
                        'accuracy': eval_acc,
                        'config': self.config
                    }, 'peft_checkpoints/model_best.pth')
                    
                    # Log classification report
                    report = classification_report(targets, predictions, output_dict=True,zero_division=0)
                    wandb.log({
                        'best_model': {
                            'accuracy': eval_acc,
                            'kappa': kappa,
                            'classification_report': report
                        }
                    })
                
                print(f'\nEpoch {epoch+1}:')
                print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
                print(f'Eval Loss: {eval_loss:.4f} | Eval Acc: {eval_acc:.2f}%')
                print(f'Kappa: {kappa:.2f}%')
                if is_best:
                    print(f'New best model! Previous best: {self.best_acc:.2f}%')
        plot_training_curves(train_losses, train_accuracies, eval_accuracies, self.config['num_epochs'])
        wandb.finish()
        return self.best_acc, self.best_epoch
'''
class PEFTTrainer:
    def __init__(self, model, train_loader, test_loader, config=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        
        # Default configuration
        self.config = {
            'num_epochs': 100,
            'learning_rate': 5e-5,  # Lower initial learning rate
            'weight_decay': 0.01,
            'warmup_epochs': 10,    # Longer warmup
            'grad_clip': 0.5,       # More aggressive gradient clipping
            'label_smoothing': 0.1,
            'eval_interval': 1,
            'use_amp': True,
            'lora_dropout': 0.2,    # Increased LoRA dropout
            'patience': 10,         # Early stopping patience
            'scheduler': {
                'T_0': 15,          # Longer initial cycle
                'T_mult': 2,
                'eta_min': 1e-6
            }
        }
        if config:
            self.config.update(config)
        
        # Initialize metrics tracking
        self.train_losses = []
        self.train_accuracies = []
        self.eval_accuracies = []
        self.learning_rates = []
        
        self.setup_training()
        
    def setup_training(self):
        # Split parameters into LoRA and non-LoRA groups
        lora_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if 'lora_' in name:
                lora_params.append(param)
            else:
                other_params.append(param)
        
        # Optimizer setup with different learning rates
        self.optimizer = optim.AdamW([
            {'params': lora_params, 'lr': self.config['learning_rate']},
            {'params': other_params, 'lr': self.config['learning_rate'] * 0.1}
        ], weight_decay=self.config['weight_decay'])
        
        # Scheduler setup
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config['scheduler']['T_0'],
            T_mult=self.config['scheduler']['T_mult'],
            eta_min=self.config['scheduler']['eta_min']
        )
        
        # Loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.config['label_smoothing'])
        
        # AMP scaler
        self.scaler = GradScaler() if self.config['use_amp'] else None
        
        # Training state
        self.best_acc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        
        # Create checkpoint directory
        os.makedirs('peft_checkpoints', exist_ok=True)
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Learning rate warmup
        if epoch < self.config['warmup_epochs']:
            warmup_factor = (epoch + 1) / self.config['warmup_epochs']
            current_lr = self.config['learning_rate'] * warmup_factor
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["num_epochs"]}')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Enhanced normalization
            inputs = (inputs - inputs.mean(dim=(2,3), keepdim=True)) / (inputs.std(dim=(2,3), keepdim=True) + 1e-8)
            
            self.optimizer.zero_grad()
            
            if self.config['use_amp']:
                with autocast(device_type='cuda'):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                self.scaler.scale(loss).backward()
                
                if self.config['grad_clip'] > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                
                if self.config['grad_clip'] > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
                
                self.optimizer.step()
            
            # Update learning rate
            if epoch >= self.config['warmup_epochs']:
                self.scheduler.step(epoch + batch_idx / len(self.train_loader))
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
            
            # Early convergence warning
            if correct/total > 0.99 and batch_idx > len(self.train_loader) // 2:
                print("\nWarning: Very high accuracy achieved mid-epoch. Consider adjusting learning rate.")
        
        # Store current learning rate
        self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
        
        return total_loss / len(self.train_loader), 100. * correct / total

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss = 0
        predictions = []
        targets_all = []
        
        for inputs, targets in tqdm(self.test_loader, desc='Evaluating'):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            inputs = (inputs - inputs.mean(dim=(2,3), keepdim=True)) / (inputs.std(dim=(2,3), keepdim=True) + 1e-8)
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            predictions.extend(predicted.cpu().numpy())
            targets_all.extend(targets.cpu().numpy())
        
        predictions = np.array(predictions)
        targets_all = np.array(targets_all)
        acc = accuracy_score(targets_all, predictions) * 100
        kappa = cohen_kappa_score(targets_all, predictions) * 100
        
        return total_loss / len(self.test_loader), acc, kappa, predictions, targets_all

    def train(self):
        wandb.init(project="peft-gcvit-hyperspectral", config=self.config)
        
        for epoch in range(self.config['num_epochs']):
            # Training phase
            train_loss, train_acc = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            
            # Evaluation phase
            if (epoch + 1) % self.config['eval_interval'] == 0:
                eval_loss, eval_acc, kappa, predictions, targets = self.evaluate()
                self.eval_accuracies.append(eval_acc)
                
                # Log metrics
                wandb.log({
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'eval_loss': eval_loss,
                    'eval_acc': eval_acc,
                    'kappa': kappa,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
                
                # Check for improvement
                is_best = eval_acc > self.best_acc
                if is_best:
                    self.best_acc = eval_acc
                    self.best_epoch = epoch
                    self.patience_counter = 0
                    
                    # Save best model
                    torch.save({
                        'epoch': epoch,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'scheduler': self.scheduler.state_dict(),
                        'accuracy': eval_acc,
                        'config': self.config
                    }, 'peft_checkpoints/model_best.pth')
                    
                    # Log detailed metrics for best model
                    report = classification_report(targets, predictions, output_dict=True, zero_division=0)
                    wandb.log({
                        'best_model': {
                            'accuracy': eval_acc,
                            'kappa': kappa,
                            'classification_report': report
                        }
                    })
                else:
                    self.patience_counter += 1
                
                # Print epoch summary
                print(f'\nEpoch {epoch+1}:')
                print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
                print(f'Eval Loss: {eval_loss:.4f} | Eval Acc: {eval_acc:.2f}%')
                print(f'Kappa: {kappa:.2f}%')
                if is_best:
                    print(f'New best model! Previous best: {self.best_acc:.2f}%')
                
                # Early stopping check
                if self.patience_counter >= self.config['patience']:
                    print(f'\nEarly stopping triggered after {self.config["patience"]} epochs without improvement')
                    break
        
        # Plot final training curves
        plot_training_curves(
            self.train_losses,
            self.train_accuracies,
            self.eval_accuracies,
            epoch + 1
        )
        
        wandb.finish()
        return self.best_acc, self.best_epoch
def create_data_loader(batch_size=64):
    # Load and validate raw data
    X, y = loadData()
    print('Hyperspectral data shape: ', X.shape)
    print('Label shape: ', y.shape)
    print('Original label range:', np.min(y), 'to', np.max(y))
    print('Unique labels:', np.unique(y))
    
    # Apply PCA
    test_ratio = 0.90
    patch_size = 15
    pca_components = 15
    X_pca = applyPCA(X, numComponents=pca_components)
    print('Data shape after PCA: ', X_pca.shape)
    
    # Create image cubes and validate labels
    X_pca, y_all = createImageCubes(X_pca, y, windowSize=patch_size)
    print('Data cube X shape: ', X_pca.shape)
    print('Processed label range:', np.min(y_all), 'to', np.max(y_all))
    print('Unique processed labels:', np.unique(y_all))
    
    # Split dataset
    Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X_pca, y_all, test_ratio)
    print('Xtrain shape: ', Xtrain.shape)
    print('Xtest shape: ', Xtest.shape)
    
    # Reshape data
    X = X_pca.reshape(-1, patch_size, patch_size, pca_components, 1)
    Xtrain = Xtrain.reshape(-1, patch_size, patch_size, pca_components, 1)
    Xtest = Xtest.reshape(-1, patch_size, patch_size, pca_components, 1)
    
    X = X.transpose(0, 4, 3, 1, 2).squeeze(1)
    Xtrain = Xtrain.transpose(0, 4, 3, 1, 2).squeeze(1)
    Xtest = Xtest.transpose(0, 4, 3, 1, 2).squeeze(1)
    
    # Create datasets
    trainset = TrainDS(Xtrain, ytrain)
    testset = TestDS(Xtest, ytest)
    allset = TestDS(X, y_all)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, 
                                             shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, 
                                            shuffle=False, num_workers=0)
    all_loader = torch.utils.data.DataLoader(allset, batch_size=batch_size, 
                                           shuffle=False, num_workers=0)
    
    return train_loader, test_loader, all_loader, y
def AA_andEachClassAccuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

def acc_reports(y_test, y_pred_test):


    '''target_names = ['Corn', 'Cotton', 'Sesame', 'Broad-leaf soybean',
                   'Narrow-leaf soybean', 'Rice', 'Water',
                   'Roads and houses', 'Mixed weed']'''
    
    target_names= ['Red roof', 'Road', 'Bare soil', 'Cotton'
        , 'Cotton firewood', 'Rape', 'Chinese cabbage',
                    'Pakchoi', 'Cabbage', 'Tuber mustard', 'Brassica parachinensis', 'Brassica chinensis', 'Small Brassica chinensis', 'Lactuca sativa', 
                    'Celtuce', 'Film covered lettuce', 'Romaine lettuce', 'Carrot', 'White radish', 'Garlic sprout	', 'Broad bean', 'Tree']
    '''target_names = ['Brocoli_green_weeds_1','Brocoli_green_weeds_2','Fallow','Fallow_rough_plow','Fallow_smooth',
                        'Stubble','Celery','Grapes_untrained','Soil_vinyard_develop','Corn_senesced_green_weeds',
                        'Lettuce_romaine_4wk','Lettuce_romaine_5wk','Lettuce_romaine_6wk','Lettuce_romaine_7wk',
                        'Vinyard_untrained','Vinyard_vertical_trellis']'''
    

    classification = classification_report(y_test, y_pred_test, digits=4, target_names=target_names,zero_division=0)
    oa = accuracy_score(y_test, y_pred_test)
    confusion = confusion_matrix(y_test, y_pred_test)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred_test)
    return classification, oa*100, confusion, each_acc*100, aa*100, kappa*100

@torch.no_grad()
def test(device, model, test_loader):
    model.eval()
    count = 0
    y_pred_test = 0
    y_test = 0
    
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        # Normalize inputs
        inputs = (inputs - inputs.mean(dim=(2,3), keepdim=True)) / (inputs.std(dim=(2,3), keepdim=True) + 1e-8)
        
        outputs = model(inputs)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        
        if count == 0:
            y_pred_test = outputs
            y_test = labels
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))
            y_test = np.concatenate((y_test, labels))
    
    return y_pred_test, y_test

def plot_training_curves(train_losses, train_accuracies, eval_accuracies, epoch):
    plt.figure(figsize=(15, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(eval_accuracies, label='Validation Accuracy')
    plt.title('Accuracy Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/training_curves_{epoch}.png')
    plt.close()

def main():
    # Create data loaders with slightly larger batch size for efficiency
    train_loader, test_loader, all_loader, y_all = create_data_loader(batch_size=64)
    
    # Create PEFT model with modified hyperparameters
    model = PEFTHyperspectralGCViT(
        in_channels=15,
        num_classes=22,
        dim=96,
        depths=[3, 4, 19],
        num_heads=[4, 8, 16],
        window_size=[7, 7, 7],
        r=19,           # Increased LoRA rank for better adaptation
        lora_alpha=32   # Keep original scaling factor
    )
    
    # Enhanced training configuration
    config = {
        'num_epochs': 100,
        'learning_rate': 5e-5,      # Lower initial learning rate
        'weight_decay': 0.01,
        'warmup_epochs': 10,        # Longer warmup period
        'grad_clip': 0.5,           # More conservative gradient clipping
        'label_smoothing': 0.1,
        'eval_interval': 1,
        'use_amp': True,
        'lora_dropout': 0.2,        # Increased dropout for regularization
        'patience': 15,             # Longer patience for early stopping
        'scheduler': {
            'T_0': 15,              # Longer initial cycle
            'T_mult': 2,
            'eta_min': 1e-6
        }
    }
    
    # Initialize trainer
    trainer = PEFTTrainer(model, train_loader, test_loader, config)
    
    # Training time measurement
    tic1 = time.perf_counter()
    
    # Train model
    best_acc, best_epoch = trainer.train()
    
    toc1 = time.perf_counter()
    
    # Load best model for testing with error handling
    try:
        checkpoint = torch.load('peft_checkpoints/model_best.pth', weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        print("Successfully loaded best model checkpoint")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
    
    # Testing time measurement
    tic2 = time.perf_counter()
    
    # Get predictions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nGenerating predictions...")
    y_pred_test, y_test = test(device, model, test_loader)
    
    toc2 = time.perf_counter()
    
    # Calculate metrics
    print("\nCalculating performance metrics...")
    classification, oa, confusion, each_acc, aa, kappa = acc_reports(y_test, y_pred_test)
    
    # Calculate times
    training_time = toc1 - tic1
    test_time = toc2 - tic2
    
    # Save results with more detailed information
    os.makedirs('cls_result', exist_ok=True)
    file_name = "cls_result/classification_report_peft_hong.txt"
    
    with open(file_name, 'w') as x_file:
        x_file.write(f'Model Configuration:\n')
        x_file.write(f'LoRA rank (r): 16\n')
        x_file.write(f'LoRA alpha: 32\n')
        x_file.write(f'Learning rate: {config["learning_rate"]}\n')
        x_file.write(f'Batch size: 64\n\n')
        
        x_file.write(f'Training Time (s): {training_time:.2f}\n')
        x_file.write(f'Test Time (s): {test_time:.2f}\n')
        x_file.write(f'Best epoch: {best_epoch}\n\n')
        
        x_file.write(f'Performance Metrics:\n')
        x_file.write(f'Overall Accuracy (%): {oa:.2f}\n')
        x_file.write(f'Average Accuracy (%): {aa:.2f}\n')
        x_file.write(f'Kappa Score (%): {kappa:.2f}\n\n')
        
        x_file.write(f'Per-Class Accuracies (%):\n')
        target_names= ['Red roof', 'Road', 'Bare soil', 'Cotton'
        , 'Cotton firewood', 'Rape', 'Chinese cabbage',
                    'Pakchoi', 'Cabbage', 'Tuber mustard', 'Brassica parachinensis', 'Brassica chinensis', 'Small Brassica chinensis', 'Lactuca sativa', 
                    'Celtuce', 'Film covered lettuce', 'Romaine lettuce', 'Carrot', 'White radish', 'Garlic sprout	', 'Broad bean', 'Tree']
        for name, acc in zip(target_names, each_acc):
            x_file.write(f'{name}: {acc:.2f}\n')
        x_file.write(f'\nDetailed Classification Report:\n{classification}\n')
        x_file.write(f'\nConfusion Matrix:\n{confusion}\n')
    
    print("\nTraining completed!")
    print(f"Best accuracy: {best_acc:.2f}% at epoch {best_epoch}")
    print(f"Kappa Score: {kappa:.2f}%")
    print(f"Results saved to {file_name}")
    
    # Generate classification maps
    print('\nGenerating classification maps...')
    cls_labels = get_cls_map(model, device, all_loader, y_all)
    print('Classification maps have been saved to the classification_maps directory')
    
    # Save the final model with comprehensive metadata
    torch.save({
        'state_dict': model.state_dict(),
        'config': config,
        'performance': {
            'accuracy': best_acc,
            'kappa': kappa,
            'training_time': training_time,
            'test_time': test_time,
            'best_epoch': best_epoch,
            'per_class_accuracy': each_acc.tolist(),
            'confusion_matrix': confusion.tolist()
        },
        'model_config': {
            'lora_rank': 16,
            'lora_alpha': 32,
            'dim': 96,
            'depths': [3, 4, 19],
            'num_heads': [4, 8, 16],
            'window_size': [7, 7, 7]
        }
    }, 'peft_checkpoints/final_model.pth')

if __name__ == '__main__':
    main()