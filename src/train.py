
from imports import *


# Plotting confusion matrix
def plot_confusion_matrix(cm, class_names):
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, fontsize=8, horizontalalignment='right')
    plt.yticks(tick_marks, class_names, fontsize=8)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color, fontsize=7)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure

#  Training and Validation
def train_epoch(model, device, dataloader, loss_fn, optimizer):
    train_loss, train_correct = 0.0, 0
    model.train()
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = loss_fn(output,labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
        scores, predictions = torch.max(output.data, 1)
        train_correct += (predictions == labels).sum().item()
                    
    return train_loss,train_correct
  
def valid_epoch(model,device,dataloader,loss_fn):
    valid_loss, val_correct = 0.0, 0
    model.eval()
    y_true, y_pred = [], []  # Use for Confusion Matrix
    y_t, y_p = [], []  # Use for Metrics (f1, precision, recall)
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        loss = loss_fn(output, labels)
        valid_loss += loss.item() * images.size(0)
        scores, predictions = torch.max(output.data, 1)

        val_correct += (predictions == labels).sum().item()

    return valid_loss, val_correct

# Testing and Inference
def test_inference(model, device, dataloader, loss_fn, class_names):
    test_loss, test_correct = 0.0, 0
    model.eval()
    y_true, y_pred = [], []  # Use for Confusion Matrix
    y_t, y_p = [], []  # Use for Metrics (f1, precision, recall)
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        loss = loss_fn(output, labels)
        test_loss += loss.item() * images.size(0)
        scores, predictions = torch.max(output.data, 1)
        test_correct += (predictions == labels).sum().item()
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predictions.cpu().numpy())
        y_t.append(labels.cpu().numpy())
        y_p.append(predictions.cpu().numpy())
    cf_matrix = confusion_matrix(y_true, y_pred)
    cf_figure = plot_confusion_matrix(cf_matrix, class_names)

    return test_loss, test_correct, cf_figure, cf_matrix


if __name__ == '__main__':

    args = args_parser()

    # Set device parameter
    if args.gpu:
        # Mac OS
        if os.name == 'posix' and torch.backends.mps.is_available():
            device = 'mps'
        
        # Windows OS
        elif os.name == 'nt' and torch.cuda.is_available():  # device is windows with cuda
            device = args.device
        
        # Any other OS with CUDA
        else:
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'

    
    # ======================= DATA ======================= #

    data_dir = '../data/'
    dataset = SkinCancer(data_dir, '../csv/train.csv', transform=None)
    dataset_size = len(dataset)
    test_dataset = SkinCancer(data_dir, '../csv/test.csv', transform=None)
    classes = np.unique(dataset.classes)

    # ======================= Model | Loss Function | Optimizer ======================= #

    if args.model == 'efficientnet':
        model = efficientnet()

    elif args.model == 'resnet':
        model = resnet()

    elif args.model == 'vit':
        model = vit()

    elif args.model == 'convnext':
        model = convnext()

    elif args.model == 'alexnet':
        model = alexnet()

    elif args.model == 'cnn':
        model = cnn()

    # copy weights
    MODEL_WEIGHTS = copy.deepcopy(model.state_dict())

    # ======================= Set Optimizer and loss Function ======================= #
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=0.9)
    elif args.optimizer == 'adamx':
        optimizer = torch.optim.Adamax(model.parameters(), lr=args.lr)

    criterion = nn.CrossEntropyLoss()
    
    batch_size = args.batch
    class_names = dataset.classes

    # ======================= Logger ======================= #      

    if args.logger == 'tb':
        tboard = SummaryWriter(log_dir=f'../logs/tensorboard/{model._get_name()}/{args.modality}_{args.epochs}_Epochs')
    else:
        # local logger
        logger = None

    # ======================= Start ======================= #
    start_t = time.time()
    best_acc = 0.0
    step = 0
    k = 5
    splits = KFold(n_splits=k, shuffle=True, random_state=42)
    args.finetune = 'finetune' if args.finetune else 'transfer'

    # ======================= Local Logger ======================= #

    exp_dir = f'../logs/{model._get_name()}_{args.epochs}/'
    os.makedirs(exp_dir, exist_ok=True)
    log_file = f"{exp_dir}/log.log"
    LOGGER = logging.getLogger(__name__)
    setup_logging(log_path=log_file, log_level='INFO', logger=LOGGER)

    # ======================= Local Logger ======================= #
    
    LOGGER.info(f'Device: {device}')
    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(dataset)))):

        LOGGER.info('Fold: {}, Model: {}'.format(fold, model._get_name()))
        
        # model.load_state_dict(MODEL_WEIGHTS) # uncomment to start fresh for each fold

        model.to(device)
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(val_idx)

        # train, will change for each fold
        train_loader = DataLoader(dataset, batch_size=batch_size,sampler=train_sampler)
        val_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)
        
        # hold out set, test once at the end of each fold
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        # ======================= Train per fold ======================= #
        for epoch in range(args.epochs):
            step += 1
            train_loss, train_correct = train_epoch(model, device, train_loader, criterion, optimizer)
            val_loss, val_correct = valid_epoch(model, device, val_loader, criterion)
            test_loss_epoch, test_acc_epoch, cf_figure, _ = test_inference(model, device, test_loader, criterion,
                                                                           class_names)
            tboard.add_figure("Confusion Matrix Epoch", cf_figure, step)

            train_loss = train_loss / len(train_loader.sampler)
            train_acc = train_correct / len(train_loader.sampler) * 100
            val_loss = val_loss / len(val_loader.sampler)
            val_acc = val_correct / len(val_loader.sampler) * 100

            LOGGER.info(f'Epoch: {epoch}/{args.epochs}')
            LOGGER.info(f'Average Training Loss: {train_loss}')
            LOGGER.info(f'Average Validation Loss: {val_loss}')
            LOGGER.info(f'Average Training Acc: {train_acc}')
            LOGGER.info(f'Average Validation acc: {val_acc}')

            test_loss_epoch = test_loss_epoch / len(test_loader.sampler)
            test_acc_epoch = test_acc_epoch / len(test_loader.sampler) * 100

            # ======================= Save per Epoch ======================================= #

            tboard.add_scalars('Loss', {'train': train_loss,
                                        'val': val_loss,
                                        'test': test_loss_epoch}, step)

            tboard.add_scalars('Acc', {'train': train_acc,
                                       'val': val_acc,
                                       'test': test_acc_epoch}, step)

            # ======================= Save model if new high accuracy ======================= #
            if test_acc_epoch > best_acc:
                LOGGER.info(f'New High Accuracy: <<<<< {test_acc_epoch} >>>>>')

                best_acc = test_acc_epoch
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(),
                           f'../models/{model._get_name()}_{args.modality}_{args.finetune}_{args.epochs}Epochs.pth')

                # # Save Scripted Model 
                # scripted_model = torch.jit.script(model)
                # torch.jit.save(scripted_model,
                #                f'../models/scripted_{model._get_name()}_{args.modality}_{args.finetune}_{args.epochs}Epochs.pt')

        # ======================= Test Model on HOS ======================= #

        # ======================= Test Model on HOS ======================= #

        test_loss, test_correct, cf_figure_fold, cf_matrix = test_inference(model, device, test_loader, criterion, class_names)  # noqa: E501

        tboard.add_figure("Confusion Matrix Fold", cf_figure_fold, fold)

        test_loss = test_loss / len(test_loader.sampler)
        test_acc = test_correct / len(test_loader.sampler) * 100

        # Save Confusion Matrix as numpy array
        np.save(f'../output_files/cf_matrix/{model._get_name()}_{args.modality}_{args.finetune}_Fold{fold}.npy', cf_matrix)  # noqa: E501

        tboard.add_scalar('Fold/Acc', test_acc, fold)
        tboard.add_scalar('Fold/Loss', test_loss, fold)

        # ======================= Save model if new high accuracy ======================= #
        if test_acc > best_acc:
            # print('#'*25)
            LOGGER.info(f'Top Accuracy: <<<<< {test_acc} >>>>>')
            # print('#'*25)
            best_acc = test_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(),
                       f'../models/{model._get_name()}_{args.modality}_{args.finetune}_{args.epochs}Epochs.pth')

    end_train = time.time()
    time_elapsed = start_t - end_train

    LOGGER.info(f'{model._get_name()} Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
