import torch
import numpy as np
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

parser = argparse.ArgumentParser(description='Test transferability', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model',       default='FP',          type=str,          help='Quantization transfer function-Q1 Q2 Q4 Q6 Q8 HT FP')
parser.add_argument('--arch',        default='resnet',      type=str,          help='Network architecture resnet, CIFARconv')
parser.add_argument('--analysis',    default='quant',       type=str     ,     help='Analysis type, Arch, Input Quant, Weight Quant or Activation Quant')
parser.add_argument('--dataset',     default='cifar10',     type=str,          help='Set dataset to use')
parser.add_argument('--attack',      default='pgd',         type=str,          help='Type of attack [PGD, CW]')
#parser.add_argument('--suffix',      default='',            type=str,          help='_adam or --- for SGD')
parser.add_argument('--opt_src',      default='',            type=str,          help='_adam or --- for SGD')
parser.add_argument('--opt_tgt',      default='',            type=str,          help='_adam or --- for SGD')
parser.add_argument('--save_path',   default='.\\outputs\\', type=str,          help='Save path for the output file')
parser.add_argument('--conf',        default='',            type=str,          help='Eg \'_30.0\' Confidence control for CW attack')

global args
args = parser.parse_args()
print(args)

args.save_path = os.path.join(args.save_path, args.dataset.lower(), 'transferability\\')

if args.analysis == 'seed':
    suffix = ["_1", "_2", "_3", "_4", "_5", "_6", "_7", "_8", "_9", "_10"]
    model = args.attack.lower() + args.conf + "_" + args.dataset.lower() + "_" + args.model + "_" + args.arch + args.opt_src + args.opt_tgt
    
            
    for s in suffix:
        if args.opt_tgt != args.opt_src:
            if args.opt_tgt =="" :
                model_name = model +s + "_src.pt"
            elif args.opt_src =="" :
                model_name = model + s +"_tgt.pt"
        filename = args.save_path + model_name
        data = torch.load(filename)
        if s == "_1":
            table = data["num_trans_images"][0].permute(1,0)
            gen_table = data["num_adv_images"].permute(1,0)
        else:
            table = torch.cat((table, data["num_trans_images"][0].permute(1,0)), dim=0)
            gen_table = torch.cat((gen_table, data["num_adv_images"].permute(1,0)), dim=0)

    mean = (table.sum() - torch.diagonal(table, 0).sum()) / 90
    std = torch.sqrt( (((table - mean)**2).sum() -  ((torch.diagonal(table, 0) - mean)**2).sum()) / 90)

    gen_mean = (gen_table.sum() - torch.diagonal(gen_table, 0).sum()) / 90
    gen_std = torch.sqrt( (((gen_table - gen_mean)**2).sum() -  ((torch.diagonal(gen_table, 0) - gen_mean)**2).sum()) / 90)
    
    print("Transferred Mean:{:.0f}     Std:{:.0f}".format(mean, std))
    print("Generated Mean:{:.0f}     Std:{:.0f}".format(gen_mean, gen_std))
    
    percent_tb = table / gen_table
    per_mean = (percent_tb.sum() - torch.diagonal(percent_tb, 0).sum()) / 90
    per_std = torch.sqrt( (((percent_tb - per_mean)**2).sum() -  ((torch.diagonal(percent_tb, 0) - per_mean)**2).sum()) / 90)
    print("Percent Trans {:.2f} \u00b1 {:.2f}".format(per_mean * 100, per_std * 100))
    row = (table - table * torch.eye(table.shape[0]).cuda())
    row_mean = row.sum(axis=0) / 9
    row_mean_2d = torch.stack( [row_mean] * row.shape[0], dim=0) * (1 - torch.eye(table.shape[0]).cuda())
    row_var = (row - row_mean_2d) ** 2
    row_std = torch.sqrt(row_var.sum(axis=0) / 9)
    row_mean = row_mean.unsqueeze(0)
    row_std = row_std.unsqueeze(0)
    np.set_printoptions(4, linewidth=200)
    print(table.cpu().numpy())

elif args.analysis == 'arch':
    suffix = [""]#"_1", "_2", "_3", "_4", "_5"]
    arch = ["resnet18", "resnet34", "resnet101", "vgg11", "vgg19", "densenet121", "wide_resnet50_2"]
    for ar in arch:
        for s in suffix:
            model = args.attack + "_" + args.analysis + args.conf +'_' + args.dataset.lower() + "_FP_torch_" + ar
            model_name = model + s + ".pt"
            data = torch.load(args.save_path+model_name)
            if s == "":
                table = data["num_trans_images"][0].permute(1,0)
            else:
                table = torch.cat((table,data["num_trans_images"][0].permute(1,0)), dim=0)
        
        if ar == "resnet18":
            CM_mean = table.mean(dim=0).unsqueeze(0)
            CM_std = table.std(dim=0).unsqueeze(0)
        else:
            CM_mean = torch.cat([CM_mean, table.mean(dim=0).unsqueeze(0)], dim=0)
            CM_std = torch.cat([CM_std, table.std(dim=0).unsqueeze(0)], dim=0)

    CM_mean = CM_mean.cpu().numpy()
    np.savetxt("temp.csv", CM_mean, delimiter=',')

    for i in range(CM_mean.shape[0]):
        CM_mean[i, :] = CM_mean[i, :] / CM_mean[i, :].max()

    np.set_printoptions(2)
    CM_mean = np.flipud(CM_mean)
    print("CM Row Mean:{}\nColumn Mean:{}".format(CM_mean.mean(axis=1)[::-1], CM_mean.mean(axis=0)))
    plt.figure(figsize=(7,6.8))
    ax = plt.subplot()
    sns.set(font_scale=1.25) # Adjust to fit
    sns.heatmap(CM_mean, annot=True, ax=ax, cmap='Reds', fmt="0.2f",  linewidths=0, cbar=False) 

    # Labels, title and ticks
    label_font = {'size':'15'}  # Adjust to fit
    ax.set_xlabel('Target Model', fontdict=label_font)
    ax.set_ylabel('Source Model', fontdict=label_font)

    ax.tick_params(axis='both', which='major', labelsize=14)  # Adjust to fit
    arch_labels = ['RN18', 'RN34', 'RN101', 'VGG11', 'VGG19', 'DN121', 'WRN50_2']
    ax.xaxis.set_ticklabels(arch_labels)
    ax.yaxis.set_ticklabels(reversed(arch_labels))
    ax.set_ylim([0, len(arch_labels)])
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_aspect(abs(x1-x0) / abs(y1-y0))
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.show()
    plt.savefig("arch_{}_5_mean.eps".format(args.attack), dpi=100)
    plt.savefig("arch_{}_5_mean.png".format(args.attack), dpi=100)

elif args.analysis == 'weight':
    suffix = ["_1", "_2", "_3", "_4", "_5"]
    input = ["w1", "w2", "w4", "w8", "w16", "w32"]
    for i in input:
        for s in suffix:
            model = args.attack+"_"+args.analysis+args.conf+ args.dataset.lower() +"_FP_"+ args.arch
            model_name = model+"_a32" + i + s + ".pt"
            try:
                data = torch.load(os.path.join(args.save_path, model_name))
            except:
                continue
            if s == "_1":
                table = data["num_trans_images"][0].permute(1,0)
            else:
                table = torch.cat((table,data["num_trans_images"][0].permute(1,0)), dim=0)
        
        if i == "w1":
            CM_mean = table.mean(dim=0).unsqueeze(0)
            CM_std = table.std(dim=0).unsqueeze(0)
        else:
            CM_mean = torch.cat([CM_mean, table.mean(dim=0).unsqueeze(0)], dim=0)
            CM_std = torch.cat([CM_std, table.std(dim=0).unsqueeze(0)], dim=0)

    CM_mean = CM_mean.cpu().numpy()
    #np.savetxt("temp.csv", CM_mean, delimiter=',')

    for i in range(CM_mean.shape[0]):
        CM_mean[i, :] = CM_mean[i, :] / CM_mean[i, :].max()

    np.set_printoptions(2)
    CM_mean = np.flipud(CM_mean)
    print("CM Row Mean:{}\nColumn Mean:{}".format(CM_mean.mean(axis=1)[::-1], CM_mean.mean(axis=0)))
    plt.figure(figsize=(7,6.8))
    ax = plt.subplot()
    sns.set(font_scale=1.25) # Adjust to fit
    sns.heatmap(CM_mean, annot=True, ax=ax, cmap='Blues', fmt="0.2f",  linewidths=0, cbar=False) 

    # Labels, title and ticks
    label_font = {'size':'15'}  # Adjust to fit
    ax.set_xlabel('Target Model', fontdict=label_font)
    ax.set_ylabel('Source Model', fontdict=label_font)

    ax.tick_params(axis='both', which='major', labelsize=14)  # Adjust to fit
    ax.yaxis.set_ticklabels(reversed(input))
    ax.xaxis.set_ticklabels(input)
    ax.set_ylim([0, len(input)])
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_aspect(abs(x1-x0) / abs(y1-y0))
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.show()
    plt.savefig("{}_weight_5_mean.png".format(args.attack), dpi=100)
    plt.savefig("{}_weight_5_mean.eps".format(args.attack), dpi=100)

elif args.analysis == 'act':
    suffix = ["_1", "_2", "_3", "_4", "_5"]
    input = ["a1", "a2", "a4", "a8", "a16", "a32"]
    for i in input:
        for s in suffix:
            model = args.attack+"_"+args.analysis+args.conf+ args.dataset.lower() +"_FP_"+ args.arch
            model_name = model+"_" + i + "w32" + s + ".pt"
            try:
                data = torch.load(os.path.join(args.save_path, model_name))
            except:
                continue
            if s == "_1":
                table = data["num_trans_images"][0].permute(1,0)
            else:
                table = torch.cat((table,data["num_trans_images"][0].permute(1,0)), dim=0)
        
        if i == "a1":
            CM_mean = table.mean(dim=0).unsqueeze(0)
            CM_std = table.std(dim=0).unsqueeze(0)
        else:
            CM_mean = torch.cat([CM_mean, table.mean(dim=0).unsqueeze(0)], dim=0)
            CM_std = torch.cat([CM_std, table.std(dim=0).unsqueeze(0)], dim=0)

    CM_mean = CM_mean.cpu().numpy()
    #np.savetxt("temp.csv", CM_mean, delimiter=',')

    for i in range(CM_mean.shape[0]):
        CM_mean[i, :] = CM_mean[i, :] / CM_mean[i, :].max()

    np.set_printoptions(2)
    CM_mean = np.flipud(CM_mean)
    print("CM Row Mean:{}\nColumn Mean:{}".format(CM_mean.mean(axis=1)[::-1], CM_mean.mean(axis=0)))
    plt.figure(figsize=(7,6.8))
    ax = plt.subplot()
    sns.set(font_scale=1.25) # Adjust to fit
    sns.heatmap(CM_mean, annot=True, ax=ax, cmap='Blues', fmt="0.2f",  linewidths=0, cbar=False) 

    # Labels, title and ticks
    label_font = {'size':'15'}  # Adjust to fit
    ax.set_xlabel('Target Model', fontdict=label_font)
    ax.set_ylabel('Source Model', fontdict=label_font)

    ax.tick_params(axis='both', which='major', labelsize=14)  # Adjust to fit
    ax.yaxis.set_ticklabels(reversed(input))
    ax.xaxis.set_ticklabels(input)
    ax.set_ylim([0, len(input)])
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_aspect(abs(x1-x0) / abs(y1-y0))
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.show()
    plt.savefig("{}_weight_5_mean.png".format(args.attack), dpi=100)
    plt.savefig("{}_weight_5_mean.eps".format(args.attack), dpi=100)

else:
    suffix = ["_1", "_2", "_3", "_4", "_5"]
    input = ["HT", "Q1", "Q2", "Q4", "Q6", "Q8", "FP"]
    for i in input:
        for s in suffix:
            model = args.attack.upper()+"_"+args.analysis+ args.conf + "_" + args.dataset.lower() +"_" + i +"_"+ args.arch
            model_name = model+s+".pt"
            data = torch.load(args.save_path+model_name)
            if s == "_1":
                table = data["num_trans_images"][0].permute(1,0)
            else:
                table = torch.cat((table,data["num_trans_images"][0].permute(1,0)), dim=0)
        
        if i == "HT":
            CM_mean = table.mean(dim=0).unsqueeze(0)
            CM_std = table.std(dim=0).unsqueeze(0)
        else:
            CM_mean = torch.cat([CM_mean, table.mean(dim=0).unsqueeze(0)], dim=0)
            CM_std = torch.cat([CM_std, table.std(dim=0).unsqueeze(0)], dim=0)

    CM_mean = CM_mean.cpu().numpy()
    #np.savetxt("temp.csv", CM_mean, delimiter=',')

    for i in range(CM_mean.shape[0]):
        CM_mean[i, :] = CM_mean[i, :] / CM_mean[i, :].max()

    np.set_printoptions(2)
    CM_mean = np.flipud(CM_mean)
    print("CM Row Mean:{}\nColumn Mean:{}".format(CM_mean.mean(axis=1)[::-1], CM_mean.mean(axis=0)))
    plt.figure(figsize=(7,6.8))
    ax = plt.subplot()
    sns.set(font_scale=1.25) # Adjust to fit
    sns.heatmap(CM_mean, annot=True, ax=ax, cmap='Blues', fmt="0.2f",  linewidths=0, cbar=False) 

    # Labels, title and ticks
    label_font = {'size':'15'}  # Adjust to fit
    ax.set_xlabel('Target Model', fontdict=label_font)
    ax.set_ylabel('Source Model', fontdict=label_font)

    ax.tick_params(axis='both', which='major', labelsize=14)  # Adjust to fit
    ax.yaxis.set_ticklabels(reversed(input))
    ax.xaxis.set_ticklabels(input)
    ax.set_ylim([0, len(input)])
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_aspect(abs(x1-x0) / abs(y1-y0))
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.show()
    plt.savefig("Quant_5_mean.eps", dpi=100)
