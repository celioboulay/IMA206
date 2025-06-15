def init(data_path, device, model):
    pass
'''    dataset = datasets.ImageFolder(data_path, transform=transform_local_center)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


    with torch.no_grad():
        features_list = []
        for images_batch, _ in dataloader:
            images_batch = images_batch.to(device)
            outputs = model.get_embedding(images_batch)
            features_list.append(outputs.cpu())
            
    features_tensor = torch.cat(features_list, dim=0)

    return features_tensor'''
