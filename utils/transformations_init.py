import torchvision.transforms as transforms

'''
On met ici les transformations qu'on utilisera ailleurs
'''

transform_pas_ouf = transforms.Compose([ # a changer bien sur 
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])



transform_trop_bien = None    # aled