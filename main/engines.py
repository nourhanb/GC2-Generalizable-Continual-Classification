import os, sys
from libs import *

def compute_discrepancy(features_T, features_S, distance_metric='L2'):
    # Normalize features
    normalized_features_T = features_T / torch.norm(features_T, p=2, dim=1, keepdim=True)
    normalized_features_S = features_S / torch.norm(features_S, p=2, dim=1, keepdim=True)
    
    if distance_metric == 'L1':
        discrepancy = F.pairwise_distance(normalized_features_T, normalized_features_S, p=1).mean()
    elif distance_metric == 'L2':
        discrepancy = F.pairwise_distance(normalized_features_T, normalized_features_S, p=2).mean()
    elif distance_metric == 'cosine_similarity':
        discrepancy = 1 - F.cosine_similarity(normalized_features_T, normalized_features_S).mean()
    elif distance_metric == 'KL':
        prob_T = F.softmax(normalized_features_T, dim=1)
        prob_S = F.softmax(normalized_features_S, dim=1)
        prob_mean = torch.mean(prob_S, dim=0, keepdim=True)
        discrepancy = F.kl_div(torch.log(prob_T), prob_mean, reduction='batchmean')
    elif distance_metric == 'JS':
        prob_T = F.softmax(normalized_features_T, dim=1)
        prob_S = F.softmax(normalized_features_S, dim=1)
        prob_mean = 0.5 * (prob_T + prob_S)
        kl_T = F.kl_div(torch.log(prob_T), prob_mean, reduction='batchmean')
        kl_S = F.kl_div(torch.log(prob_S), prob_mean, reduction='batchmean')
        discrepancy = 0.5 * (kl_T + kl_S)
    else:
        raise ValueError("Invalid distance_metric value. Supported values are: 'L1', 'L2', 'cosine_similarity', 'KL', 'JS'.")

    return discrepancy


def train_fn(
    train_loaders, num_epochs, 
    models, 
    device = torch.device("cpu"), 
    save_ckps_dir = ".", 
):
    print("\nStart Training ...\n" + " = "*16)
    FT, FS,  = models["FT"].to(device), models["FS"].to(device), 
    GS = models["GS"].to(device)
    optimizer_FS = optim.AdamW(
        FS.parameters(), 
        lr = 1e-4, 
    )
    optimizer_GS = optim.AdamW(
        GS.parameters(), 
        lr = 1e-4, 
    )
    scheduler_FS = optim.lr_scheduler.StepLR(
        optimizer_FS, 
        step_size = 40, gamma = 0.1, 
    )
    scheduler_GS = optim.lr_scheduler.StepLR(
        optimizer_GS, 
        step_size = 40, gamma = 0.1, 
    )

    best_accuracy=0.0
    for epoch in range(1, num_epochs + 1):
        print("epoch {}/{}".format(epoch, num_epochs) + "\n" + " - "*16)

        with torch.autograd.set_detect_anomaly(True):
            FT, FS,  = FT.train(), FS.train(), 
            GS = GS.train()
            for images_T, labels_T, domains_T in tqdm.tqdm(train_loaders["train"]):
                images_T, labels_T, domains_T = images_T.to(device), labels_T.to(device), domains_T.to(device)
                images_S = GS(images_T)

                features_T, features_S,  = FT.backbone(images_T.float()), FS.backbone(images_S.float()), 
                discrepancy = compute_discrepancy(
                    features_T, features_S, distance_metric='L2'
                )

                pairwise_distances = []
                for input, other,  in itertools.combinations(range(1, 4), 2):
                    distance = torch.dist(
                        input = torch.mean(images_T[domains_T == input], axis = 0), other = torch.mean(images_T[domains_T == other], axis = 0), 
                        p = 2, 
                    )
                    pairwise_distances.append(distance)
                loss_FS = F.cross_entropy(FT.classifier(features_S), labels_T) \
                        + discrepancy
                #loss_FS += 0.001 * torch.norm(FT.classifier.weight)  # Add weight decay regularization
                loss_GS = F.cross_entropy(FT.classifier(features_S), labels_T) \
                        - torch.minimum(discrepancy - sum(pairwise_distances)/len(pairwise_distances), torch.zeros(1).cuda())

                for parameter in GS.parameters():
                    parameter.requires_grad = False
                loss_FS.backward(retain_graph = True)
                for parameter in GS.parameters():
                    parameter.requires_grad = True
                loss_GS.backward(retain_graph = False)

                # Domain Generalized Representation Learning
                optimizer_FS.step()
                state_dict_FT, state_dict_FS,  = FT.state_dict(), FS.state_dict(), 
                for parameter in state_dict_FS:
                    state_dict_FT[parameter] = 0.999*state_dict_FT[parameter] + (1 - 0.999)*state_dict_FS[parameter]
                FT.load_state_dict(state_dict_FT)

                # Learning to Generate Novel Domains
                optimizer_GS.step()

                optimizer_FS.zero_grad()
                optimizer_GS.zero_grad()

                #print('Loss GS=',loss_GS)
                #print('Loss FS=', loss_FS)

        with torch.no_grad():
            FT, FS,  = FT.eval(), FS.eval(), 
            GS = GS.eval()
            running_loss, running_corrects,  = 0.0, 0.0, 
            for images_T, labels_T, domains_T in tqdm.tqdm(train_loaders["val"]):
                images_T, labels_T, domains_T = images_T.to(device), labels_T.to(device), domains_T.to(device)

                logits = FT(images_T.float())
                loss = F.cross_entropy(logits, labels_T)

                running_loss, running_corrects,  = running_loss + loss.item()*images_T.size(0), running_corrects + torch.sum(torch.max(logits, 1)[1] == labels_T.data).item(), 
        val_loss, val_accuracy,  = running_loss/len(train_loaders["val"].dataset), running_corrects/len(train_loaders["val"].dataset), 
        print('running loss =', running_loss)
        print("{:<8} - loss:{:.4f}, accuracy:{:.4f}".format(
            "val", 
            val_loss, val_accuracy, 
        ))

        if val_accuracy>best_accuracy:
            torch.save(
                GS,
                "{}/best-all.ptl".format(save_ckps_dir)
            )

        scheduler_FS.step(), 
        scheduler_GS.step(), 

        torch.save(
            FT, 
            "{}/FT-all.ptl".format(save_ckps_dir), 
        )
        torch.save(
            GS, 
            "{}/GS-all.ptl".format(save_ckps_dir), 
        )
    print("\nFinish Training ...\n" + " = "*16)