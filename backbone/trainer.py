import torch
from tqdm import tqdm

def train_epoch(train_dataloader,model,crit,optimizer):
    
    model.train()
    losses = []
    total_loss = 0
    
    for batch_idx, (data) in (enumerate(train_dataloader)):
        
        
        img1 = data[0].cuda()
        img2 = data[1].cuda()
        temp = data[2].cuda()
            
        optimizer.zero_grad()
            
        output1 = model(img1)
        output2 = model(img2)
        
        if data[2].dim()==4:
            output3 = model(temp)
            loss = crit(output1,output2,output3)
        else:
            loss = crit(output1,output2,temp)
        
        total_loss += loss.item()
                
        loss.backward()
        optimizer.step()
        
    total_loss = total_loss/(1+batch_idx)
    
    return total_loss


def validate_epoch(test_dataloader,model,crit):
    
    model.eval()
    
    with torch.no_grad():
        model.eval()
        val_loss = 0
        
        for batch_idx, (data) in (enumerate(test_dataloader)):
            if(len(data)==3):
                img1 = data[0].cuda()
                img2 = data[1].cuda()
                temp = data[2].cuda()

                output1 = model(img1)
                output2 = model(img2)
                
                if data[2].dim()==4:
                    output3 = model(temp)
                    loss = crit(output1,output2,output3)
                else:
                    loss = crit(output1,output2,temp)

            val_loss += loss.item()

    val_loss = val_loss/(1+batch_idx)
        
    return val_loss

def save_model(model,exp_name):
    experiment_name = "../experiments/best_{}.pth".format(exp_name)
    torch.save(model.state_dict(),experiment_name)
