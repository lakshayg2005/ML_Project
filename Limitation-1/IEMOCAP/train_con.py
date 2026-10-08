import argparse
import os
import numpy as np
import random
import time
import torch
import torch.optim as optim

from loss import Loss                                   # original CE / weighted loss (unchanged)
from contrastive_loss import ConfusionAwareContrastiveLoss
from dataloader import IEMOCAPRobertaDataset
from model_con import ECERC
from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score, classification_report
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import warnings
warnings.filterwarnings("ignore")


def seed_everything(seed=2007):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid * size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_IEMOCAP_bert_loaders(path=None, batch_size=32, num_workers=0, pin_memory=False, valid_rate=0.1):
    trainset = IEMOCAPRobertaDataset(train=True)
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid_rate)
    train_loader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler,
                              collate_fn=trainset.collate_fn, num_workers=num_workers, pin_memory=pin_memory)
    valid_loader = DataLoader(trainset, batch_size=batch_size, sampler=valid_sampler,
                              collate_fn=trainset.collate_fn, num_workers=num_workers, pin_memory=pin_memory)
    testset = IEMOCAPRobertaDataset(train=False)
    test_loader = DataLoader(testset, batch_size=batch_size, collate_fn=testset.collate_fn,
                             num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader


def train_or_eval_model(model, loss_f, dataloader, train_flag=False, optimizer=None, cuda_flag=False,
                        feature_type='text', target_names=None, con_loss_f=None, lambda_con=0.0):
    assert not train_flag or optimizer is not None
    losses, con_losses, preds, labels = [], [], [], []
    use_con = train_flag and con_loss_f is not None and lambda_con > 0

    model.train() if train_flag else model.eval()

    for step, data in enumerate(dataloader):
        if train_flag:
            optimizer.zero_grad()

        emo_roberta, sem_roberta, audio, vision, qmask, umask, label2 = \
            [d.cuda() for d in data[:-1]] if cuda_flag else data[:-1]
        seq_lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]

        if args.feature_type == "multi":
            emo_roberta = torch.cat([emo_roberta, audio, vision], dim=-1)
            sem_roberta = torch.cat([sem_roberta], dim=-1)

        label = torch.cat([label2[j][:seq_lengths[j]] for j in range(len(label2))])

        if use_con:
            log_prob, feats = model(emo_roberta, sem_roberta, qmask, umask, seq_lengths, return_feat=True)
            ce_loss = loss_f(log_prob, label)
            con_loss = con_loss_f(feats, label)
            loss = ce_loss + lambda_con * con_loss
            con_losses.append(con_loss.item())
        else:
            log_prob = model(emo_roberta, sem_roberta, qmask, umask, seq_lengths)
            ce_loss = loss_f(log_prob, label)
            loss = ce_loss

        preds.append(torch.argmax(log_prob, 1).cpu().numpy())
        labels.append(label.cpu().numpy())
        losses.append(ce_loss.item())          # CE only, comparable with the baseline

        if train_flag:
            loss.backward()
            optimizer.step()

    if preds != []:
        preds = np.concatenate(preds)
        labels = np.concatenate(labels)
    else:
        return float('nan'), float('nan'), float('nan'), [], [], '', float('nan')

    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_con = round(float(np.mean(con_losses)), 4) if con_losses else 0.0
    avg_accuracy = round(accuracy_score(labels, preds) * 100, 2)
    avg_fscore = round(f1_score(labels, preds, average='weighted') * 100, 2)

    all_matrix = [metrics.classification_report(labels, preds, target_names=target_names, digits=4),
                  metrics.confusion_matrix(labels, preds)]
    cr = classification_report(labels, preds, target_names=target_names, digits=4)

    return avg_loss, avg_accuracy, avg_fscore, all_matrix, [labels, preds], cr, avg_con


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_type', type=str, default='multi')
    parser.add_argument('--data_dir', type=str, default='../data/iemocap/iemocap_features_roberta.pkl')
    parser.add_argument('--output_dir', type=str, default='./contrastive_run',
                        help='NEW folder for checkpoints/results (originals are not touched)')
    parser.add_argument('--tag', type=str, default='CON', help='name tag for the saved files')
    parser.add_argument('--base_layer', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=200, metavar='E')
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64, metavar='BS')
    parser.add_argument('--valid_rate', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--l2', type=float, default=0.0002)
    parser.add_argument('--no_cuda', action='store_true', default=False)
    parser.add_argument('--class_weight', action='store_true', default=True)
    parser.add_argument('--Dataset', type=str, default="IEMOCAP")

    # contrastive regularizer hyper-parameters
    parser.add_argument('--lambda_con', type=float, default=0.1, help='weight of contrastive loss (lambda_1)')
    parser.add_argument('--con_temp', type=float, default=0.1, help='contrastive temperature')
    parser.add_argument('--conf_weight', type=float, default=2.0, help='extra weight on confusable negatives')
    parser.add_argument('--margin', type=float, default=0.5, help='cosine-distance margin between confusable prototypes')
    parser.add_argument('--proto_weight', type=float, default=0.5, help='weight of the prototype margin term')

    args = parser.parse_args()

    epochs, batch_size, output_path, data_path, base_layer, feature_type = \
        args.epochs, args.batch_size, args.output_dir, args.data_dir, args.base_layer, args.feature_type
    cuda_flag = torch.cuda.is_available() and not args.no_cuda
    os.makedirs(output_path, exist_ok=True)

    n_classes, n_speakers, hidden_size, input_size = 6, 2, 128, None
    target_names = ['hap', 'sad', 'neu', 'ang', 'exc', 'fru']
    # label ids: hap0 sad1 neu2 ang3 exc4 fru5  -> confusable: Happy/Excited, Angry/Frustrated
    confusable_pairs = [(0, 4), (3, 5)]
    class_weights = torch.FloatTensor([1 / 0.087178797, 1 / 0.145836136, 1 / 0.229786089,
                                       1 / 0.148392305, 1 / 0.140051123, 1 / 0.24875555])

    feat2dim = {'IS10': 1582, '3DCNN': 512, 'textCNN': 100, 'bert': 768, 'denseface': 342,
                'MELD_text': 600, 'MELD_audio': 300}
    D_audio = feat2dim['IS10'] if args.Dataset == 'IEMOCAP' else feat2dim['MELD_audio']
    D_visual = feat2dim['denseface']
    D_text = 1024

    if feature_type == 'text':
        input_size = D_text
    elif feature_type == "multi":
        input_size = D_text + D_audio + D_visual
    else:
        print('Error: feature_type not set.')
        exit(0)

    seed_everything()
    model = ECERC(args, d_t=D_text, d_a=D_audio, d_v=D_visual, base_layer=base_layer,
                  input_size=input_size, hidden_size=hidden_size, n_speakers=n_speakers,
                  n_classes=n_classes, cuda_flag=cuda_flag)
    con_loss_f = ConfusionAwareContrastiveLoss(confusable_pairs, n_classes=n_classes,
                                               temperature=args.con_temp, conf_weight=args.conf_weight,
                                               margin=args.margin, proto_weight=args.proto_weight)
    if cuda_flag:
        print('Running on GPU')
        class_weights = class_weights.cuda()
        model.cuda()
        con_loss_f.cuda()
    else:
        print('Running on CPU')

    print("The model has {} parameters in total".format(sum(x.numel() for x in model.parameters())))
    print('Running on the {} features........'.format(feature_type))

    ckpt_f1 = os.path.join(output_path, 'ECERC_{}_bestF1.pkl'.format(args.tag))
    ckpt_loss = os.path.join(output_path, 'ECERC_{}_bestLoss.pkl'.format(args.tag))
    report_path = os.path.join(output_path, 'report_{}.txt'.format(args.tag))
    for p in (ckpt_f1, ckpt_loss, report_path):
        if os.path.exists(p):
            raise FileExistsError('{} already exists - change --tag or --output_dir to avoid overwriting.'.format(p))

    train_loader, valid_loader, test_loader = get_IEMOCAP_bert_loaders(
        path=data_path, batch_size=batch_size, num_workers=0, valid_rate=args.valid_rate)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    loss_f = Loss(alpha=class_weights if args.class_weight else None)

    all_test_fscore, all_test_acc = [], []
    best_epoch, best_epoch2, patience, best_eval_fscore, best_eval_loss = -1, -1, 0, 0, None
    patience2 = 0
    best_test_cr, best_test_cm, best_test_lp = None, None, None

    for e in range(epochs):
        start_time = time.time()

        train_loss, train_acc, train_fscore, _, _, _, train_con = train_or_eval_model(
            model=model, loss_f=loss_f, dataloader=train_loader, train_flag=True, optimizer=optimizer,
            cuda_flag=cuda_flag, feature_type=feature_type, target_names=target_names,
            con_loss_f=con_loss_f, lambda_con=args.lambda_con)
        valid_loss, valid_acc, valid_fscore, _, _, _, _ = train_or_eval_model(
            model=model, loss_f=loss_f, dataloader=valid_loader, cuda_flag=cuda_flag,
            feature_type=feature_type, target_names=target_names)
        test_loss, test_acc, test_fscore, test_metrics, test_lp, test_cr, _ = train_or_eval_model(
            model=model, loss_f=loss_f, dataloader=test_loader, cuda_flag=cuda_flag,
            feature_type=feature_type, target_names=target_names)
        all_test_fscore.append(test_fscore)
        all_test_acc.append(test_acc)

        if args.valid_rate > 0:
            eval_loss, _, eval_fscore = valid_loss, valid_acc, valid_fscore
        else:
            eval_loss, _, eval_fscore = test_loss, test_acc, test_fscore

        if e == 0 or best_eval_fscore < eval_fscore:
            patience = 0
            best_epoch, best_eval_fscore = e, eval_fscore
            torch.save(model.state_dict(), ckpt_f1)
            best_test_cr, best_test_cm, best_test_lp = test_cr, test_metrics[1], test_lp
        else:
            patience += 1

        if best_eval_loss is None:
            best_eval_loss = eval_loss
            best_epoch2 = 0
            torch.save(model.state_dict(), ckpt_loss)
        else:
            if eval_loss < best_eval_loss:
                best_epoch2, best_eval_loss = e, eval_loss
                patience2 = 0
                torch.save(model.state_dict(), ckpt_loss)
            else:
                patience2 += 1

        print('epoch: {}, train_loss: {}, train_con: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, '
              'valid_acc: {}, valid_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'.format(
                  e, train_loss, train_con, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore,
                  test_loss, test_acc, test_fscore, round(time.time() - start_time, 2)))

        if patience >= args.patience and patience2 >= args.patience:
            break

    print('Early stopping...', patience)
    summary = 'Eval-metric: F1, Epoch: {}, best_eval_fscore: {}, Accuracy: {}, F1-Score: {}'.format(
        best_epoch, best_eval_fscore,
        all_test_acc[best_epoch] if best_epoch >= 0 else 0,
        all_test_fscore[best_epoch] if best_epoch >= 0 else 0)
    print(summary)

    # Save report, confusion matrix and predictions at the best-valid-F1 epoch
    with open(report_path, 'w') as f:
        f.write(str(vars(args)) + '\n\n' + summary + '\n\n')
        f.write(str(best_test_cr) + '\n')
        f.write('Confusion matrix (rows=true, cols=pred; order hap sad neu ang exc fru)\n')
        f.write(str(best_test_cm) + '\n')
    np.save(os.path.join(output_path, 'confusion_{}.npy'.format(args.tag)), best_test_cm)
    np.savez(os.path.join(output_path, 'preds_{}.npz'.format(args.tag)),
             labels=best_test_lp[0], preds=best_test_lp[1])
    print('Saved checkpoints/report to', output_path)