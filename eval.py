import random
import torch
from PIL import Image
import os
import argparse

import torchvision.transforms as transforms
import torch.nn as nn

from backbone.networks.inception_resnet_v1 import InceptionResnetV1

parser = argparse.ArgumentParser(description="Scoring similarity between 2 images")

parser.add_argument('--root_dir',default='lfw/')
parser.add_argument('--image1',required=True,help="Path to the first image")
parser.add_argument('--image2',required=True,help="Path to the second image")

args = parser.parse_args()

def main():
    
    root_dir = args.root_dir
    img1_path = args.image1
    img2_path = args.image2
    
    print("Loading the Model")
    Siamese_network = InceptionResnetV1(pretrained='vggface2')
    checkpoint = torch.load("pretrained/20180402-114759-vggface2.pt")
    Siamese_network.load_state_dict(checkpoint)
    Siamese_network = Siamese_network.cuda()
    Siamese_network.eval()
    
    with torch.no_grad():
        img1 = Image.open(os.path.join(root_dir, img1_path))
        img2 = Image.open(os.path.join(root_dir, img2_path))

        T_img1 = transforms.ToTensor()(img1).unsqueeze(0).cuda()
        T_img2 = transforms.ToTensor()(img2).unsqueeze(0).cuda()

        print("Generating Embeddings ....")
        embedding1 = Siamese_network(T_img1)
        embedding2 = Siamese_network(T_img2)

        print("Computing Similarity Score ... ")

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        cosine_similarity = cos(embedding1,embedding2).item()

        similarity = "similar" if cosine_similarity>0.65 else "dissimilar"

        print("Cosine_Similarity : {}".format(cosine_similarity))
        print("Detected : {}".format(similarity))
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    




