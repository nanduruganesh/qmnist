"""
MIT License

Copyright (c) 2020-present TorchQuantum Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse
import random
import numpy as np
import os
from tqdm import tqdm
import wandb

from torchquantum.dataset import MNIST, NoisyMNIST
from torch.optim.lr_scheduler import CosineAnnealingLR

import os, json, datetime
from models import *

from classical_models import ClassicalNN

def timestamp():
    return datetime.datetime.now().strftime("%Y_%m_%dT%H_%M_%S")

def train(dataflow, model, device, optimizer):
    pbar = tqdm(dataflow["train"], desc="Training Progress")
    for feed_dict in pbar:
        inputs = feed_dict["image"].to(device)
        targets = feed_dict["digit"].to(device)

        outputs = model(inputs)
        loss = F.nll_loss(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_description(f"Training Progress - Loss: {loss.item():.4f}")
        wandb.log({"Training Loss": loss.item()})


def valid_test(dataflow, split, model, device, qiskit=False):
    target_all = []
    output_all = []
    with torch.no_grad():
        for feed_dict in dataflow[split]:
            inputs = feed_dict["image"].to(device)
            targets = feed_dict["digit"].to(device)

            outputs = model(inputs, use_qiskit=qiskit)

            target_all.append(targets)
            output_all.append(outputs)
        target_all = torch.cat(target_all, dim=0)
        output_all = torch.cat(output_all, dim=0)

    _, indices = output_all.topk(1, dim=1)
    masks = indices.eq(target_all.view(-1, 1).expand_as(indices))
    size = target_all.shape[0]
    corrects = masks.sum().item()
    accuracy = corrects / size
    loss = F.nll_loss(output_all, target_all).item()

    print(f"{split} set accuracy: {accuracy}")
    print(f"{split} set loss: {loss}")

    return {split: {'accuracy': accuracy, 'loss': loss}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--static", action="store_true", help="compute with " "static mode"
    )
    parser.add_argument("--pdb", action="store_true", help="debug with pdb")
    parser.add_argument("--qiskit-simulation", action="store_true", help="run on a real quantum computer")
    parser.add_argument(
        "--wires-per-block", type=int, default=2, help="wires per block int static mode"
    )
    parser.add_argument(
        "--epochs", type=int, default=2, help="number of training epochs"
    )
    parser.add_argument(
        "--noise", type=float, default=0, help="std. dev of gaussian noise"
    )
    parser.add_argument(
        "--n_wires", type=int, default=10, help="number of wires in QNN components"
    )
    parser.add_argument(
        "--q_layers", type=int, default=5, help="number of quantum layers"
    )
    parser.add_argument(
        "--classical_layers", type=int, default=2, help="number of classical layers"
    )
    parser.add_argument(
        "--func", type=str, default='rx', help="quantum gate to run in layers"
    )
    parser.add_argument( # for slurm array
        "--mult-noise-by", type=float, default=1, help="multiply noise by this number"
    )
    parser.add_argument(
        "--model_name", type=str, default="QNN", help="Name of model to use, for now either QNN, EightQubitNN or just ClassicalNN"
    )
    parser.add_argument(
        "--save_to", type=str, default=f"runs/{timestamp()}", help="Path to save experiment results. This script will create the path if it doesn't exist."
    )
    parser.add_argument(
        "--api_key", type=str, default="", help="Path to api key if using IBMQ"
    )
    parser.add_argument(
        "--early_stopping", type=bool, default=True, help="Stop training early if accuracy drops twice"
    )
    parser.add_argument(
        "--group", type=str, default="", help="Name of wandb group to save to"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed"
    )
    parser.add_argument(
        "--wandb_name", type=str, default="", help="Overwrites the WandB model name"
    )
    parser.add_argument(
        "--no_noise_training", type=bool, default=False, help="Enables noiseless training, and noise is only added to test set"
    )

    args = parser.parse_args()
    if not os.path.exists(args.save_to):
        os.makedirs(args.save_to, exist_ok=True)

    args.noise *= args.mult_noise_by
    if not args.group:
        args.group = args.save_to

    print(args)
    if "classical" in args.model_name:
        wandb_run_name = f"{args.model_name}_noise:{args.noise}_n_layers:{args.classical_layers}"
    else:
        wandb_run_name = f"{args.model_name}_noise:{args.noise}_n_wires:{args.n_wires}_q_layers:{args.q_layers}_func:{args.func}"
    if args.wandb_name:
        wandb_run_name = args.wandb_name

    wandb.init(project="QMNIST", group=args.group, name=wandb_run_name)

    if args.pdb:
        import pdb

        pdb.set_trace()

    if args.no_noise_training:
        test_noise = args.noise
        args.noise = 0

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    digits_of_interest = list(range(10))
    dataset = NoisyMNIST(
        root="./mnist_data",
        train_valid_split_ratio=[0.9, 0.1],
        digits_of_interest=digits_of_interest,
        n_test_samples=75,
        std_dev=args.noise,
    )
    dataflow = dict()

    for split in dataset:
        sampler = torch.utils.data.RandomSampler(dataset[split])
        dataflow[split] = torch.utils.data.DataLoader(
            dataset[split],
            batch_size=256,
            sampler=sampler,
            num_workers=8,
            pin_memory=True,
        )

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.model_name == "QNN":
        model = QFCModel().to(device)
    elif args.model_name == "ClassicalNN":
        model = ClassicalNN(n_classes=len(digits_of_interest), n_middle_layers = args.classical_layers).to(device)
    elif args.model_name == "EightQNN":
        model = EightQNN().to(device)
    elif args.model_name == "TwentyQNN":
        model = TwentyQNN().to(device)
    elif args.model_name == "LayeredQNN":
        model = LayeredQNN(n_wires = args.n_wires, q_layers = args.q_layers).to(device)
    elif args.model_name == "HybridQNN":
        model = HybridQNN(n_wires = args.n_wires, q_layers = args.q_layers, func=args.func).to(device)
    else:
        raise ValueError(f"{args.model_name} not supported yet please add.")

    wandb.watch(model, log_freq=1)
    n_epochs = args.epochs
    optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
    results = {'noise': args.noise, 'epochs': args.epochs, 'model_name': args.model_name}
    results['valid'] = {'accuracy': 0}
    accuracy_decreasing = False

    if args.static:
        # optionally to switch to the static mode, which can bring speedup
        # on training
        model.q_layer.static_on(wires_per_block=args.wires_per_block)

    for epoch in range(1, n_epochs + 1):
        # train
        print(f"Epoch {epoch}:")
        train(dataflow, model, device, optimizer)
        print(optimizer.param_groups[0]["lr"])

        # valid
        old_accuracy = results['valid']['accuracy']
        results.update(valid_test(dataflow, "valid", model, device))
        new_accuracy = results['valid']['accuracy']
        wandb.log({"Validation Loss": results['valid']['loss']})
        wandb.log({"Validation Accuracy": results['valid']['accuracy']})

        
        if args.early_stopping and epoch > 10 and accuracy_decreasing and new_accuracy < old_accuracy:
            print("Accuracy decreased twice, stopping training")
            break
        accuracy_decreasing = (new_accuracy < old_accuracy)

        scheduler.step()

    # test
    if args.no_noise_training:
        dataset = NoisyMNIST(
            root="./mnist_data",
            train_valid_split_ratio=[0.9, 0.1],
            digits_of_interest=digits_of_interest,
            n_test_samples=75,
            std_dev=test_noise,
        )
        for split in ["test"]:
            sampler = torch.utils.data.RandomSampler(dataset[split])
            dataflow[split] = torch.utils.data.DataLoader(
                dataset[split],
                batch_size=256,
                sampler=sampler,
                num_workers=8,
                pin_memory=True,
            )

    print(args.noise)
    print(test_noise)
    results.update(valid_test(dataflow, "test", model, device, qiskit=False))
    wandb.log({"Test Loss": results['test']['loss']})
    wandb.log({"Test Accuracy": results['test']['accuracy']})

    model_id = os.urandom(5).hex() + '.pt'
    while model_id in os.listdir(args.save_to):
        model_id = os.urandom(5).hex() + '.pt'
    
    model_path = args.save_to + '/' + model_id
    torch.save(model, model_path)

    results['saved_model'] = model_id
    results_path = args.save_to + '/results.json'
    all_results = []
    if os.path.exists(results_path):
        with open(results_path) as f:
            try:
                all_results = json.load(f)
            except: pass # If file is improper ignore it

    all_results.append(results)
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=3)
    

    if args.qiskit_simulation:
        # run on Qiskit simulator and real Quantum Computers
        try:
            from qiskit import IBMQ
            from torchquantum.plugin import QiskitProcessor
        
            if not os.path.exists(args.api_key):
                raise Exception("provide api key filepath in command line args")
            api_key = open(args.api_key).read()
            IBMQ.save_account(api_key)

            # firstly perform simulate
            print(f"\nTest with Qiskit Simulator")
            processor_simulation = QiskitProcessor(use_real_qc=False)
            model.set_qiskit_processor(processor_simulation)
            valid_test(dataflow, "test", model, device, qiskit=True)

            return
            # then try to run on REAL QC   
            backend_name = "ibmq_lima"
            print(f"\nTest on Real Quantum Computer {backend_name}")
            # Please specify your own hub group and project if you have the
            # IBMQ premium plan to access more machines.
            processor_real_qc = QiskitProcessor(
                use_real_qc=True,
                backend_name=backend_name,
                hub="ibm-q",
                group="open",
                project="main",
            )
            model.set_qiskit_processor(processor_real_qc)
            valid_test(dataflow, "test", model, device, qiskit=True)
        except ImportError:
            print(
                "Please install qiskit, create an IBM Q Experience Account and "
                "save the account token according to the instruction at "
                "'https://github.com/Qiskit/qiskit-ibmq-provider', "
                "then try again."
            )


if __name__ == "__main__":
    main()
