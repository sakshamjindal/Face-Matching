import torch
from tqdm import tqdm

def train_epoch(train_dataloader,model,crit,optimizer):
    
    model.train()
    losses = []
    total_loss = 0
    
    for batch_idx, (data) in (enumerate(train_dataloader)):
        
        
        img1 = data[0].cuda()
        img2 = data[1].cuda()
        label = data[2].cuda()
            
        optimizer.zero_grad()
            
        output1 = model(img1)
        output2 = model(img2)
        loss = crit(output1,output2,label)
        
        total_loss += loss.item()
                
        loss.backward()
        optimizer.step()
        
    total_loss = total_loss/(1+batch_idx)
    
    return total_loss


def validate_epoch(test_dataloader,model,crit):
    
    with torch.no_grad():
        model.eval()
        val_loss = 0
        
        for batch_idx, (data) in (enumerate(test_dataloader)):
            if(len(data)==3):
                img1 = data[0].cuda()
                img2 = data[1].cuda()
                label = data[2].cuda()

            output1 = model(img1)
            output2 = model(img2)
            loss = crit(output1,output2,label)

            val_loss += loss.item()

    val_loss = val_loss/(1+batch_idx)
        
    return val_loss

def save_model(model,exp_name):
    experiment_name = "../experiments/best_{}.pth".format(exp_name)
    torch.save(model.state_dict(),experiment_name)
