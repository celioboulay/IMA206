from torchvision import transforms

ssl_transform = transforms.Compose([  # corriger, ajouter random crop et resize et gerer les transfos non chromatiques ou jsp comment ils appellent ca
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    #transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.GaussianBlur(2),
    transforms.ToTensor(),
])
