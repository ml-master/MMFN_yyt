import os
import torch
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, \
    classification_report
from transformers import logging
from torch.autograd import Variable
from torch.utils.data import DataLoader
from MMFN import MultiModal
from tqdm import tqdm
from myweibo_dataset import *
from gossipcop_dataset import *
from twitter_dataset import *

# Set logging verbosity to warning and error levels for transformers
logging.set_verbosity_warning()
logging.set_verbosity_error()

# Set CUDA_VISIBLE_DEVICES to control GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Check if CUDA is available, else use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"


# Define the training function
def train():
    batch_size = 32
    patience = 5  # 早停机制的耐心参数，如果验证集损失在连续5个epoch没有改善，就停止训练
    best_loss = np.inf
    patience_counter = 0

    # Load training and validation datasets
    # train_set = twitter_dataset(is_train=True)
    # validate_set = twitter_dataset(is_train=False)
    # train_set = weibo_dataset(is_train=True)
    # validate_set = weibo_dataset(is_train=False)
    train_set = gossipcop_dataset(is_train=True)
    validate_set = gossipcop_dataset(is_train=False)

    # Create data loaders for training and testing
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=8,
        collate_fn=collate_fn,  # A collate function that preprocesses batch elements
        shuffle=True
    )

    test_loader = DataLoader(
        validate_set,
        batch_size=batch_size,
        num_workers=8,
        collate_fn=collate_fn,
        shuffle=False
    )

    # Initialize the MultiModal model
    rumor_module = MultiModal()
    # rumor_module.forward = rumor_module.forward_no_image
    rumor_module.to(device)

    # Define the CrossEntropyLoss criterion for rumor classification
    loss_f_rumor = torch.nn.CrossEntropyLoss()

    # Extract parameters for optimizer groups
    base_params = list(map(id, rumor_module.bert.parameters()))
    base_params += list(map(id, rumor_module.swin.parameters()))

    # Define the optimizer with different learning rates for different parameter groups
    optim_task = torch.optim.Adam([
        {'params': filter(lambda p: p.requires_grad and id(p) not in base_params, rumor_module.parameters())},
        {'params': rumor_module.bert.parameters(), 'lr': 1e-5},
        {'params': rumor_module.swin.parameters(), 'lr': 1e-5}
    ], lr=1e-3)

    # Training loop
    for epoch in range(50):  # 假设训练最多50个epoch
        print("start to train")
        rumor_module.train()
        corrects_pre_rumor = 0
        loss_total = 0
        rumor_count = 0
        tk0 = tqdm(train_loader, desc="train", smoothing=0, mininterval=1.0)
        for i, (input_ids, attention_mask, token_type_ids, image, imageclip, textclip, label) in enumerate(tk0):
            # Transfer data to the appropriate device
            input_ids, attention_mask, token_type_ids, image, imageclip, textclip, label = to_var(input_ids), to_var(
                attention_mask), to_var(token_type_ids), to_var(image), to_var(imageclip), to_var(textclip), to_var(
                label)

            # Encode image and text data using pre-trained CLIP models
            with torch.no_grad():
                image_clip = clipmodel.encode_image(imageclip)
                text_clip = clipmodel.encode_text(textclip)

            # Forward pass through the MultiModal model
            pre_rumor = rumor_module(input_ids, attention_mask, token_type_ids, image, text_clip, image_clip)

            # Calculate the rumor loss
            loss_rumor = loss_f_rumor(pre_rumor, label)

            # Backpropagation and optimization
            optim_task.zero_grad()
            loss_rumor.backward()
            optim_task.step()

            # Calculate accuracy and update counters
            pre_label_rumor = pre_rumor.argmax(1)
            corrects_pre_rumor += pre_label_rumor.eq(label.view_as(pre_label_rumor)).sum().item()
            loss_total += loss_rumor.item() * input_ids.shape[0]
            rumor_count += input_ids.shape[0]

        # Calculate training accuracy and loss
        loss_rumor_train = loss_total / rumor_count
        acc_rumor_train = corrects_pre_rumor / rumor_count

        # Evaluate on the test set
        acc_rumor_test, precision_rumor_test, recall_rumor_test, f1_rumor_test, loss_rumor_test, conf_rumor = test(
            rumor_module, test_loader)

        # Print results
        print('-----------rumor detection----------------')
        print(
            "EPOCH = %d || acc_rumor_train = %.3f || acc_rumor_test = %.3f || loss_rumor_train = %.3f || loss_rumor_test = %.3f" %
            (epoch + 1, acc_rumor_train, acc_rumor_test, loss_rumor_train, loss_rumor_test))

        print('-----------rumor_confusion_matrix---------')
        print(conf_rumor)

        # 早停机制
        if loss_rumor_test < best_loss:
            best_loss = loss_rumor_test
            patience_counter = 0
            torch.save(rumor_module.state_dict(), 'best_model.pth')  # 保存最佳模型
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    # 加载最佳模型
    rumor_module.load_state_dict(torch.load('best_model.pth'))
    return rumor_module, test_loader


# Helper function to transfer data to the appropriate device
def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


# Helper function to test the model
def test(rumor_module, test_loader):
    rumor_module.eval()

    loss_f_rumor = torch.nn.CrossEntropyLoss()

    rumor_count = 0
    loss_total = 0
    rumor_label_all = []
    rumor_pre_label_all = []

    with torch.no_grad():
        for i, (input_ids, attention_mask, token_type_ids, image, imageclip, textclip, label) in enumerate(test_loader):
            # Transfer data to the appropriate device
            input_ids, attention_mask, token_type_ids, image, imageclip, textclip, label = to_var(input_ids), to_var(
                attention_mask), to_var(token_type_ids), to_var(image), to_var(imageclip), to_var(textclip), to_var(
                label)

            # Encode image and text data using pre-trained CLIP models
            image_clip = clipmodel.encode_image(imageclip)
            text_clip = clipmodel.encode_text(textclip)

            # Forward pass through the MultiModal model
            pre_rumor = rumor_module(input_ids, attention_mask, token_type_ids, image, text_clip, image_clip)
            loss_rumor = loss_f_rumor(pre_rumor, label)
            pre_label_rumor = pre_rumor.argmax(1)

            loss_total += loss_rumor.item() * input_ids.shape[0]
            rumor_count += input_ids.shape[0]

            # Collect predictions and labels
            rumor_pre_label_all.append(pre_label_rumor.detach().cpu().numpy())
            rumor_label_all.append(label.detach().cpu().numpy())

        # Calculate metrics
        loss_rumor_test = loss_total / rumor_count
        rumor_pre_label_all = np.concatenate(rumor_pre_label_all, 0)
        rumor_label_all = np.concatenate(rumor_label_all, 0)

        acc_rumor_test = accuracy_score(rumor_label_all, rumor_pre_label_all)
        precision_rumor_test = precision_score(rumor_label_all, rumor_pre_label_all, average=None)
        recall_rumor_test = recall_score(rumor_label_all, rumor_pre_label_all, average=None)
        f1_rumor_test = f1_score(rumor_label_all, rumor_pre_label_all, average=None)
        conf_rumor = confusion_matrix(rumor_label_all, rumor_pre_label_all)

        # Generate classification report
        classification_report_rumor = classification_report(rumor_label_all, rumor_pre_label_all,
                                                            target_names=['realnews', 'fakenews'], digits=4)

    print("Overall Accuracy:", acc_rumor_test)
    print("Precision per class:", precision_rumor_test)
    print("Recall per class:", recall_rumor_test)
    print("F1 Score per class:", f1_rumor_test)
    print("Confusion Matrix:\n", conf_rumor)
    print("Classification Report:\n", classification_report_rumor)

    return acc_rumor_test, precision_rumor_test, recall_rumor_test, f1_rumor_test, loss_rumor_test, conf_rumor


# Entry point
if __name__ == "__main__":
    model, test_loader = train()
    test(model, test_loader)
