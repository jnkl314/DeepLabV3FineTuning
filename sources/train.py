import os
import torch
import numpy as np
import time
import copy
import cv2

def debug_export_before_forward(inputs, labels, idx):
    # im = inputs[0]*255;
    im = inputs[0];
    im = im.to('cpu').numpy()
    im[0, :, :] = im[0, :, :] * 0.229 + 0.485
    im[1, :, :] = im[1, :, :] * 0.224 + 0.456
    im[2, :, :] = im[2, :, :] * 0.225 + 0.406
    im = im * 255
    im = im.astype(np.uint8)
    la = labels[0].to(torch.uint8).to('cpu').numpy()
    im = im.transpose([1, 2, 0])
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{idx:06}_im.png", im)
    cv2.imwrite(f"{idx:06}_la.png", la)


def iou(pred, target, n_classes = 3):
  ious = []
  pred = pred.view(-1)
  target = target.view(-1)

  # Ignore IoU for background class ("0")
  for cls in range(1, n_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
    pred_inds = pred == cls
    target_inds = target == cls
    intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()  # Cast to long to prevent overflows
    union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection
    if union > 0:
        ious.append(float(intersection) / float(max(union, 1)))

  return np.array(ious)


def train_model(model, num_classes, dataloaders, criterion, optimizer, device, dest_dir, num_epochs=25):
    since = time.time()

    val_acc_history = []

    best_model_state_dict = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    counter = 0

    for epoch in range(1, num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_iou_means = []

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Security, skip this iteration if the batch_size is 1
                if 1 == inputs.shape[0]:
                    print("Skipping iteration because batch_size = 1")
                    continue

                # Debug
                # debug_export_before_forward(inputs, labels, counter)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)['out']
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                iou_mean = iou(preds, labels, num_classes).mean()
                running_loss += loss.item() * inputs.size(0)
                running_iou_means.append(iou_mean)

                # Increment counter
                counter = counter + 1

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            if running_iou_means is not None:
                epoch_acc = np.array(running_iou_means).mean()
            else:
                epoch_acc = 0.


            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_state_dict = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

            # Save current model every 25 epochs
            if 0 == epoch%25:
                current_model_path = os.path.join(dest_dir, f"checkpoint_{epoch:04}_DeepLabV3_Skydiver.pth")
                print(f"Save current model : {current_model_path}")
                torch.save(model.state_dict(), current_model_path)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    return best_model_state_dict, val_acc_history
