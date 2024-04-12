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

import torchquantum as tq
from torchquantum.plugin import (
    tq2qiskit_measurement,
    qiskit_assemble_circs,
    op_history2qiskit,
    op_history2qiskit_expand_params,
)

from torchquantum.dataset import MNIST, NoisyMNIST
from torch.optim.lr_scheduler import CosineAnnealingLR

import os, json, datetime


class QFCModel(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(
                n_ops=50, wires=list(range(self.n_wires))
            )

            # gates with trainable parameters
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)

            # some trainable gates (instantiated ahead of time)
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=3)
            self.crx0(qdev, wires=[0, 2])

            # add some more non-parameterized gates (add on-the-fly)
            qdev.h(wires=3)  # type: ignore
            qdev.sx(wires=2)  # type: ignore
            qdev.cnot(wires=[3, 0])  # type: ignore
            qdev.rx(
                wires=1,
                params=torch.tensor([0.1]),
                static=self.static_mode,
                parent_graph=self.graph,
            )  # type: ignore

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_u3_h_rx"])

        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x, use_qiskit=False):
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=x.shape[0], device=x.device, record_op=True
        )

        bsz = x.shape[0]
        x = F.avg_pool2d(x, 6).view(bsz, 16)
        devi = x.device

        if use_qiskit:
            # use qiskit to process the circuit
            # create the qiskit circuit for encoder
            self.encoder(qdev, x)  
            op_history_parameterized = qdev.op_history
            qdev.reset_op_history()
            encoder_circs = op_history2qiskit_expand_params(self.n_wires, op_history_parameterized, bsz=bsz)

            # create the qiskit circuit for trainable quantum layers
            self.q_layer(qdev)
            op_history_fixed = qdev.op_history
            qdev.reset_op_history()
            q_layer_circ = op_history2qiskit(self.n_wires, op_history_fixed)

            # create the qiskit circuit for measurement
            measurement_circ = tq2qiskit_measurement(qdev, self.measure)

            # assemble the encoder, trainable quantum layers, and measurement circuits
            assembled_circs = qiskit_assemble_circs(
                encoder_circs, q_layer_circ, measurement_circ
            )

            # call the qiskit processor to process the circuit
            x0 = self.qiskit_processor.process_ready_circs(qdev, assembled_circs).to(  # type: ignore
                devi
            )
            x = x0

        else:
            # use torchquantum to process the circuit
            self.encoder(qdev, x)
            qdev.reset_op_history()
            self.q_layer(qdev)
            x = self.measure(qdev)

        # x = x.reshape(bsz, 2, 2).sum(-1).squeeze()
        x = x.reshape(bsz, 4)
            
        x = F.log_softmax(x, dim=1)

        return x

def timestamp():
    return datetime.datetime.now().strftime("%Y_%m_%dTH_%M_%S")

def train(dataflow, model, device, optimizer):
    for feed_dict in dataflow["train"]:
        inputs = feed_dict["image"].to(device)
        targets = feed_dict["digit"].to(device)

        outputs = model(inputs)
        loss = F.nll_loss(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"loss: {loss.item()}", end="\r")


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

    # FOR SLURM ARRAY HELP
    parser.add_argument(
        "--mult-noise-by", type=float, default=1, help="multiply noise by this number"
    )
    parser.add_argument(
        "--model_name", type=str, default="QNN", help="Name of model to use, for now either QNN or just ClassicalNN"
    )
    parser.add_argument(
        "--save_to", type=str, default=f"runs/{timestamp()}", help="Path to save experiment results. This script will create the path if it doesn't exist."
    )
    parser.add_argument(
        "--api_key", type=str, default="", help="Path to api key if using IBMQ"
    )

    args = parser.parse_args()
    if not os.path.exists(args.save_to):
        os.makedirs(args.save_to, exist_ok=True)

    args.noise *= args.mult_noise_by

    print(args)

    if args.pdb:
        import pdb

        pdb.set_trace()

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    digits_of_interest = [1,3,5,9]
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
    #elif args.model_name == "ClassicalNN":
        #model = ClasicalNN(n_classes=len(digits_of_interest)).to(device)
    else:
        raise ValueError(f"{args.model_name} not supported yet please add.")

    n_epochs = args.epochs
    optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)
    results = {'noise': args.noise, 'epochs': args.epochs, 'model': args.model_name}

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
        results.update(valid_test(dataflow, "valid", model, device))
        scheduler.step()

    # test
    results.update(valid_test(dataflow, "test", model, device, qiskit=False))

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
